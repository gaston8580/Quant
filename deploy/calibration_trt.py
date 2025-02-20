import os, sys, argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from calibration import build_dataloader


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, cache_file):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data = calibration_data
        self.cache_file = cache_file
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.data[0].nbytes)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.current_index < len(self.data):
            batch = self.data[self.current_index]
            cuda.memcpy_htod(self.device_input, batch)
            self.current_index += 1
            return [int(self.device_input)]
        else:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def calibration_trt(args):
    # 校准数据
    train_loader, _ = build_dataloader(False, args.data_dir, args.batch_size, args.workers, training=True)
    calibration_data = []
    for i, (image, label) in enumerate(train_loader):
        if i >= args.steps:
            break
        calibration_data.append(image.numpy())

    # 创建校准器
    calibrator = Calibrator(calibration_data, f"deploy/onnx/{args.model}_calibration.cache")

    # 加载ONNX模型
    with open(f"deploy/onnx/{args.model}_float.onnx", "rb") as f:
        onnx_model = f.read()

    # 创建TensorRT构建器
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    parser.parse(onnx_model)

    # 设置构建器配置
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

    # 构建engine, 并生成calibration.cache
    engine = builder.build_engine(network, config)

    # 保存engine
    flag_use_trtexec = 0
    if not flag_use_trtexec:
        with open(f'deploy/onnx/{args.model}_int8.engine', 'wb') as f:
            f.write(engine.serialize())
    else:
        os.system(f'trtexec --onnx=deploy/onnx/{args.model}_float.onnx ' + \
                  f'--saveEngine=deploy/onnx/{args.model}_int8.engine ' + \
                  f'--calib=deploy/onnx/{args.model}_calibration.cache --int8')


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default='ResNet18', choices=['AlexNet', 'ResNet18'], required=False, 
                        help='model name')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--steps', type=int, default=300, required=False, help='step nums for calibration')
    parser.add_argument('--workers', type=int, default=10, help='number of workers for dataloader')
    parser.add_argument('--data_dir', type=str, default='/data/sfs_turbo/perception/animals/', help='data path')
    parser.add_argument('--output_dir', default='outputs', help='dir for saving ckpts and log files')
    parser.add_argument('--ckpt', type=str, default='float_model.pth', help='checkpoint to start from')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_config()
    calibration_trt(args)