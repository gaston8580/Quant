# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
import numpy as np
import torch
import logging
import torch.distributed as dist
import random
import os
import subprocess
import pickle
import shutil
from models.Alexnet import AlexNet
from models.ResNet18 import ResNet18
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_model_map():
    return {'AlexNet': AlexNet, 'ResNet18': ResNet18}


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method('spawn')
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        # init_method='tcp://127.0.0.1:%d' % tcp_port,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def build_dataloader(dist, data_dir, batch_size, workers=4, training=True):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 将[0,1]的像素值归一化到[-1,1]
    if training:
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.RandomVerticalFlip(),  # 随机垂直翻转, 增强数据
                                        transforms.ToTensor(),
                                        normalize])
        dataset = ImageFolder(f'{data_dir}/train', transform=transform)
    else:
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        normalize])
        dataset = ImageFolder(f'{data_dir}/val', transform=transform)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = get_dist_info()
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=workers, 
                            shuffle=(sampler is None) and training, sampler=sampler)
    return dataloader, sampler


def visualize_image(image_tensor, title=None, cmap=None):
    """
    args:
        image_tensor (torch.Tensor): shape为 (C, H, W) 或 (H, W) 的图像张量
        title (str, optional): 图像标题
        cmap (str, optional): 用于灰度图像的颜色映射
    """
    plt.close()
    image_tensor = (image_tensor + 1) / 2  # 将[-1,1]的像素值恢复到[0,1]
    image_np = image_tensor.cpu().numpy()
    # 处理灰度图像
    if image_np.ndim == 2:
        plt.imshow(image_np, cmap=cmap)
    # 处理彩色图像
    elif image_np.ndim == 3:
        if image_np.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            image_np = image_np.transpose(1, 2, 0)
        plt.imshow(image_np)
    else:
        raise ValueError('The dimension of the image tensor must be 2 or 3')
    
    if title:
        plt.title(title)
    plt.axis('off')  # 隐藏坐标轴
    plt.show()