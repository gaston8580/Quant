import torch.nn as nn
import matplotlib.pyplot as plt
import torch, os, time, argparse, warnings, datetime
import torch.ao.quantization as quant
from tools import common_utils
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tools.quantization_utils import convert_model_float2qat, convert_model_float2calibration

warnings.filterwarnings('ignore')
torch.set_printoptions(sci_mode=False)


def visualize_loss_acc(train_loss, val_loss, train_acc, val_acc, folder, details):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss value')
    plt.xlabel('epoch num')
    plt.title("loss")
    plt.savefig(f'{folder}/loss_{details}.png')
    plt.close()

    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc value')
    plt.xlabel('epoch num')
    plt.title("accuracy")
    plt.savefig(f'{folder}/acc_{details}.png')
    plt.close()


def build_dataloader(dist, data_dir, batch_size, workers=4, training=True):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 将[0,1]的像素值归一化到[-1,1]
    if training:
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.RandomVerticalFlip(),  # 随机垂直翻转, 增强数据
                                        transforms.ToTensor(),
                                        normalize])
        dataset = ImageFolder(data_dir + '/train', transform=transform)
    else:
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        normalize])
        dataset = ImageFolder(data_dir + '/val', transform=transform)
    
    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=workers, 
                            shuffle=(sampler is None) and training, sampler=sampler)
    return dataloader, sampler


def train_one_epoch(dataloader, model, optimizer, loss_fn):
    model.train()
    loss, current, n = 0.0, 0.0, 0
    for _, (x, y) in enumerate(dataloader):
        image, y = x.cuda(), y.cuda()
        output = model(image)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred) / output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n += 1

    train_loss = loss / n
    train_acc = current / n
    return train_loss, train_acc


def eval_one_epoch(args, dataloader, model, loss_fn):
    model = quant.convert(model) if args.stage == 'qat' else model
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for _, (x, y) in enumerate(dataloader):
            image, y = x.cuda(), y.cuda()
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n += 1

    val_loss = loss / n
    val_acc = current / n
    return val_loss, val_acc


def train_model(args, model):
    if args.launcher == 'none':
        dist = False
        total_gpus = 1
        args.without_sync_bn = True
        args.local_rank = 0
    else:
        dist = True
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        total_gpus, args.local_rank = getattr(common_utils, f'init_dist_{args.launcher}') \
                                        (args.tcp_port, args.local_rank, backend='nccl')
    
    batch_size = args.batch_size // total_gpus
    os.makedirs(f'{args.output_dir}/{args.model}/logs', exist_ok=True)
    details = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(args.output_dir, args.model, f'logs/log_train_{details}.txt') if not args.debug else None
    logger = common_utils.create_logger(log_file, rank=args.local_rank)

    logger.info('********************** Start logging **********************')
    logger.info(f'CUDA DEVICES = {[i for i in range(torch.cuda.device_count())]}')
    if dist:
        logger.info(f'batch size per gpu: {batch_size}')
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    
    train_loader, train_sampler = build_dataloader(dist, args.data_dir, batch_size, args.workers, training=True)
    val_loader, _ = build_dataloader(dist, args.data_dir, batch_size, args.workers, training=False)

    if args.stage == 'qat':
        args.epochs = args.qat_epochs
        convert_model_float2qat(args, model)
        ckpt_path = os.path.join(args.output_dir, args.model, 'calibration_model.pth')
        model.load_state_dict(torch.load(ckpt_path))
    
    model.cuda()
    if not args.without_sync_bn:
        # 将模型中的BN层转换为同步BN(SyncBatchNorm): 在分布式训练中保持BN层统计的一致性，可以提升模型在多GPU训练时的性能。
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # lr每10个epoch x0.5, TODO: 尝试cos学习率, 20240916    
    loss_fn = nn.CrossEntropyLoss()

    model.train()  # before wrap to DistributedDataParallel to support to fix some parameters
    if dist:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % torch.cuda.device_count()], 
                                                    find_unused_parameters=True)
    
    logger.info(model)
    num_total_params = sum([x.numel() for x in model.parameters()])
    logger.info(f'Total number of parameters: {num_total_params}')
    logger.info('********************** Start training **********************')

    # TODO: 加入tensorboard, 20240906
    train_loss_list, train_acc_list, val_loss_list, val_acc_list, max_acc = [], [], [], [], 0
    # train!!!
    for cur_epoch in range(args.epochs):
        start = time.time()
        torch.cuda.empty_cache()
        if train_sampler is not None:
            train_sampler.set_epoch(cur_epoch)
        
        train_loss, train_acc = train_one_epoch(train_loader, model, optimizer, loss_fn)
        val_loss, val_acc = eval_one_epoch(args, val_loader, model, loss_fn)
        scheduler.step()

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        logger.info(f"epoch ({cur_epoch + 1}/{args.epochs}) done in {time.time() - start:.3f} seconds, " +
                    f"train_loss={train_loss:.3f}, val_loss={val_loss:.3f}, train_acc={train_acc:.3f}, " + 
                    f"val_acc={val_acc:.3f}")
        # save model
        if val_acc > max_acc and args.local_rank == 0:
            max_acc = val_acc
            save_model = quant.convert(model) if args.stage == 'qat' else model
            if dist:
                torch.save(save_model.module.state_dict(), f'{args.output_dir}/{args.model}/{args.stage}_model.pth')
            else:
                torch.save(save_model.state_dict(), f'{args.output_dir}/{args.model}/{args.stage}_model.pth')
            logger.info(f"saved best model, epoch {cur_epoch + 1}")
    
    if args.local_rank == 0 and not args.debug:
        visualize_loss_acc(train_loss_list, val_loss_list, train_acc_list, val_acc_list, 
                           f'{args.output_dir}/{args.model}/logs', details)
    logger.info('********************** End training **********************')


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default='AlexNet', choices=['AlexNet', 'ResNet18'], required=False, 
                        help='model name')
    parser.add_argument('--stage', type=str, default='qat', choices=['float', 'qat'], required=False, 
                        help='the train stage')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, required=False, help='learning rate')
    parser.add_argument('--epochs', type=int, default=32, required=False, help='number of epochs to train')
    parser.add_argument('--qat_epochs', type=int, default=15, required=False, help='number of epochs to qat')
    parser.add_argument('--workers', type=int, default=10, help='number of workers for dataloader')
    parser.add_argument('--data_dir', type=str, default='/data/sfs_turbo/perception/animals/', help='data path')
    parser.add_argument('--output_dir', default='outputs', help='dir for saving ckpts and log files')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--debug', type=bool, default=False, help='whether in debug mode')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--without_sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=2, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--not_eval_with_train', action='store_true', default=False, help='')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--add_worker_init_fn', action='store_true', default=False, help='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_config()
    models = common_utils.get_model_map()
    model = models[args.model]()

    train_model(args, model)  # train float or qat model