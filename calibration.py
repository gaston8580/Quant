import torch.nn as nn
import matplotlib.pyplot as plt
import torch, os, argparse, time
from models.Alexnet import AlexNet
from tools import common_utils
from torch.quantization.observer import MinMaxObserver
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm


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


class SymmetricObserver(MinMaxObserver):
    def __init__(self, dtype=torch.qint8, quant_min=-128, quant_max=127):
        super(SymmetricObserver, self).__init__(dtype=dtype, quant_min=quant_min, quant_max=quant_max)

    def calculate_qparams(self):
        '''对称量化通常适用于正负范围对称的数据，非对称量化更适用于输入数据分布不均的数据。'''
        # 设置为对称量化
        scale = (self.max_val - self.min_val) / float(self.quant_max - self.quant_min)
        zero_point = 0
        return torch.tensor([scale]), torch.tensor([zero_point], dtype=torch.int32)


def calibration_one_epoch(args, dataloader, model):
    # model.qconfig = torch.quantization.default_qconfig
    model.qconfig = torch.quantization.QConfig(
        activation=SymmetricObserver.with_args(dtype=torch.qint8),
        weight=SymmetricObserver.with_args(dtype=torch.qint8)
        )
    model = torch.quantization.prepare(model)  # Insert observers

    model.eval()
    with torch.no_grad():
        start = time.time()
        for i, (image, label) in enumerate(dataloader):
            if i >= args.steps:
                break
            image, label = (image.cuda(), label.cuda()) if args.mode == 'cuda' else (image, label)
            model(image)
        print(f"Calibration time: {time.time() - start:.3f} seconds")
    model = torch.quantization.convert(model)  # Convert model
    return model

def eval_calibration_model(args, dataloader, model):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        pbar = tqdm(total=len(dataloader), desc="Evaluating", unit="batch")
        for _, (image, label) in enumerate(dataloader):
            image, label = (image.cuda(), label.cuda()) if args.mode == 'cuda' else (image, label)
            output = model(image)
            cur_loss = loss_fn(output, label)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(label == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n += 1
            pbar.set_postfix({"loss": loss / n, "accuracy": current / n})  # 设置进度条信息
            pbar.update(1)  # 更新进度条
    pbar.close()
    val_loss = loss / n
    val_acc = current / n
    return val_loss, val_acc


def calibration():
    args = parse_config()

    model = AlexNet()
    if args.mode == 'cuda':
        model = model.cuda()
    ckpt_path = os.path.join(args.output_dir, args.ckpt)
    if 'module' in list(torch.load(ckpt_path))[0]:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(ckpt_path))

    train_loader, _ = build_dataloader(False, args.data_dir, args.batch_size, args.workers, training=True)
    val_loader, _ = build_dataloader(False, args.data_dir, args.batch_size, args.workers, training=False)
    model = calibration_one_epoch(args, train_loader, model)
    val_loss, val_acc = eval_calibration_model(args, val_loader, model)
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--mode', type=str, default='cuda', required=False, help='gpu or cpu')
    parser.add_argument('--batch_size', type=int, default=32, required=False, help='batch size for training')
    parser.add_argument('--steps', type=int, default=10, required=False, help='step nums for calibration')
    parser.add_argument('--workers', type=int, default=10, help='number of workers for dataloader')
    parser.add_argument('--data_dir', type=str, default='/data/sfs_turbo/perception/animals/', help='data path')
    parser.add_argument('--output_dir', default='outputs', help='dir for saving ckpts and log files')
    parser.add_argument('--ckpt', type=str, default='best_model.pth', help='checkpoint to start from')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    calibration()