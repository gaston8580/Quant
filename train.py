import torch.nn as nn
from Alexnet import AlexNet
from tools import common_utils
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.distributed as dist
import matplotlib.pyplot as plt
import torch, os, time, argparse, warnings, datetime

warnings.filterwarnings('ignore')

ROOT_TRAIN = r'/data/sfs_turbo/perception/animals/train'
ROOT_TEST = r'/data/sfs_turbo/perception/animals/val'


def build_dataloader():
    # 将图像的像素值归一化到[-1, 1]
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(),  # 随机垂直全展, 增强数据
        transforms.ToTensor(),
        normalize])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])

    train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
    val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet().cuda()

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD随机梯度下降

# 学习率每10个epoch变为原来的0.5
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 定义训练函数
def train_one_epoch(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        image, y = x.to(device), y.to(device)
        output = model(image)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred) / output.shape[0]

        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    train_loss = loss / n
    train_acc = current / n
    print('train_loss ' + str(train_loss))
    print('train_acc ' + str(train_acc))
    return train_loss, train_acc


def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

    val_loss = loss / n
    val_acc = current / n
    print('val_loss ' + str(val_loss))
    print('val_acc ' + str(val_acc))
    return val_loss, val_acc


def matplot_loss(train_loss, val_loss, folder):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss value')
    plt.xlabel('epoch num')
    plt.title("loss")
    plt.savefig(f'{folder}/loss.png')


def matplot_acc(train_acc, val_acc, folder):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc value')
    plt.xlabel('epoch num')
    plt.title("accuracy")
    plt.savefig(f'{folder}/acc.png')


def train_model():
    args = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
        args.without_sync_bn = True
    else:
        dist_train = True
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        total_gpus, args.local_rank = getattr(common_utils, f'init_dist_{args.launcher}') \
                                        (args.tcp_port, args.local_rank, backend='nccl')
    
    details = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(args.output_dir, f'log_train_{details}.txt')
    logger = common_utils.create_logger(log_file, rank=args.local_rank)

    # log to file
    logger.info('**********************Start logging**********************')
    logger.info(f'CUDA DEVICES = {[i for i in range(torch.cuda.device_count())]}')
    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))

    loss_train, acc_train, loss_val, acc_val = [], [], [], []

    folder = 'outputs'
    epoch = 128
    max_acc = 0
    for t in range(epoch):
        start = time.time()
        print(f"epoch {t + 1} / {epoch}")
        train_loss, train_acc = train_one_epoch(train_dataloader, model, loss_fn, optimizer)
        val_loss, val_acc = val(val_dataloader, model, loss_fn)

        loss_train.append(train_loss)
        acc_train.append(train_acc)
        loss_val.append(val_loss)
        acc_val.append(val_acc)

        lr_scheduler.step()

        # 保存最好的模型权重
        if val_acc > max_acc:
            if not os.path.exists(folder):
                os.mkdir(folder)
            max_acc = val_acc
            print(f"save best model, epoch {t + 1}")
            torch.save(model.state_dict(), f'{folder}/best_model.pth')

        if t == epoch - 1:
            torch.save(model.state_dict(), f'{folder}/latest_model.pth')
        print(f"epoch {t + 1} done in {time.time() - start:.3f} seconds\n-----------------------")

    matplot_loss(loss_train, loss_val, folder)
    matplot_acc(acc_train, acc_val, folder)
    print('Training Done!')


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=256, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=10, help='number of workers for dataloader')
    parser.add_argument('--output_dir', default='outputs', help='dir for saving ckpts and log files')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='pytorch')
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
    train_model()