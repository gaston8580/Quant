import matplotlib.pyplot as plt
import torch, os, argparse, time
import torch.ao.quantization as quant
from tools import common_utils
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tools.common_utils import visualize_image
from tools.quantization_utils import convert_model_float2calibration, convert_model_float2qat


def build_dataloader(data_dir, batch_size, workers=4):
    # transforms.Normalize(mean, std)
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 将[0,1]的像素值归一化到[-1,1]
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])
    dataset = ImageFolder(data_dir + '/val', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=workers, 
                            shuffle=True, sampler=None)
    return dataloader


def eval_one_epoch(dataloader, model):
    classes = {0: "cat", 1: "dog"}
    model.eval()
    with torch.no_grad():
        for _, (x, y) in enumerate(dataloader):
            image, y = x.cuda(), y.item()
            pred = model(image)
            pred_class = torch.argmax(pred[0]).item()
            predicted, gt = classes[pred_class], classes[y]
            visualize_image(image[0], title=f'predicted: {predicted}, ground truth: {gt}')
            # time.sleep(0.5)


def eval_single_ckpt(args, model):
    args = parse_config()
    ckpt_path = os.path.join(args.output_dir, args.model, f'{args.stage}_model.pth')

    model.cuda()
    if args.stage != 'float':
        convert_model_float2qat(args, model)
    if args.stage == 'qat':
        quant.convert(model, inplace=True)
        model.load_state_dict(torch.load(ckpt_path))
    elif args.stage == 'calibration':
        model.load_state_dict(torch.load(ckpt_path))
        quant.convert(model, inplace=True)
    else:
        model.load_state_dict(torch.load(ckpt_path))

    val_loader = build_dataloader(args.data_dir, batch_size=1, workers=1)
    eval_one_epoch(val_loader, model)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default='AlexNet', choices=['AlexNet', 'ResNet18'], required=False, 
                        help='model name')
    parser.add_argument('--stage', type=str, default='qat', choices=['float', 'calibration', 'qat'], 
                        required=False, help="the predict stage")
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=10, help='number of workers for dataloader')
    parser.add_argument('--data_dir', type=str, default='/data/sfs_turbo/perception/animals/', help='data path')
    parser.add_argument('--output_dir', default='outputs', help='dir for saving ckpts and log files')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_config()
    models = common_utils.get_model_map()
    model = models[args.model]()

    eval_single_ckpt(args, model)