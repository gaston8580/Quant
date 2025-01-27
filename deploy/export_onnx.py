import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, argparse
import torch.ao.quantization as quant
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tools import common_utils
from tools.quantization_utils import convert_model_float2qat


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


def export_onnx(args, model):
    args = parse_config()
    ckpt_path = os.path.join(args.ckpt_dir, args.model, f'{args.stage}_model.pth')

    model.cuda()
    if args.stage != 'float':
        convert_model_float2qat(model)
    if args.stage == 'qat':
        quant.convert(model, inplace=True)
        model.load_state_dict(torch.load(ckpt_path))
    elif args.stage == 'calibration':
        model.load_state_dict(torch.load(ckpt_path))
        quant.convert(model, inplace=True)
    else:
        model.load_state_dict(torch.load(ckpt_path))

    model.eval()
    val_loader = build_dataloader(args.data_dir, batch_size=1, workers=1)
    dummy_input = None
    for _, (x, y) in enumerate(val_loader):
        image, y = x.cuda(), y.item()
        dummy_input = image
        break
    
    os.makedirs(f'{args.output_dir}/onnx', exist_ok=True)
    save_path = f'{args.output_dir}/onnx/{args.model}_{args.stage}.onnx'
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            # input_names=["image"],
            output_names=None,
            opset_version=13,
            do_constant_folding=False,
            export_params=True,
            verbose=True,
        )


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default='ResNet18', choices=['AlexNet', 'ResNet18'], required=False, 
                        help='model name')
    parser.add_argument('--stage', type=str, default='calibration', choices=['float', 'calibration', 'qat'], 
                        required=False, help="the predict stage")
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=10, help='number of workers for dataloader')
    parser.add_argument('--data_dir', type=str, default='/data/sfs_turbo/perception/animals/', help='data path')
    parser.add_argument('--ckpt_dir', default='outputs', help='dir for saving ckpts and log files')
    parser.add_argument('--output_dir', default='deploy', help='dir for saving ckpts and log files')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_config()
    models = common_utils.get_model_map()
    model = models[args.model]()

    export_onnx(args, model)