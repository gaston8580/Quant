import os, sys, torch, argparse, time
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from tqdm import tqdm
from tools.common_utils import visualize_image, build_dataloader

# for compatibility with older numpy versions
# if not hasattr(np, 'bool'):
#     np.bool = np.bool_


def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def TRTEngineInit(trt_path):
    engine = load_engine(trt_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, input_name2idx_dict, output_name2idx_dict = [], [], [], {}, {}

    binding_idx = 0
    for tensor_name in engine:
        size = trt.volume(engine.get_tensor_shape(tensor_name))
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)  # 分配device内存
        bindings.append(device_mem)
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(host_mem)
            input_name2idx_dict[tensor_name] = binding_idx
        else:
            outputs.append(host_mem)
            output_name2idx_dict[tensor_name] = binding_idx
        binding_idx += 1
    for name in output_name2idx_dict.keys():
        output_name2idx_dict[name] = output_name2idx_dict[name] - len(inputs)
    return inputs, outputs, bindings, context, input_name2idx_dict, output_name2idx_dict, engine


def trt_engine_execute(context, input_name2idx_dict, output_name2idx_dict, bindings, onnx_inputs, inputs, outputs):
    result = {}
    for key in input_name2idx_dict.keys():
        input_data = onnx_inputs[key].cpu().numpy() if isinstance(onnx_inputs[key], torch.Tensor) else onnx_inputs[key]
        cp_idx = input_name2idx_dict[key]
        np.copyto(inputs[cp_idx], input_data.ravel())  # 可验证input精度是否正确
    for inp, src in zip(bindings[:len(inputs)], inputs):
        cuda.memcpy_htod_async(inp, src)  # cpu to gpu
    context.execute_v2(bindings)  # trt inference
    for out, dst in zip(outputs, bindings[len(inputs):]):
        cuda.memcpy_dtoh_async(out, dst)  # gpu to cpu
    for name in output_name2idx_dict.keys():
        idx = output_name2idx_dict[name]
        if 'onnx' not in name:
            result[name] = outputs[idx]
    return result


def eval_trt_engine(args, trt_path):
    inputs, outputs, bindings, context, input_name2idx_dict, output_name2idx_dict, _ = TRTEngineInit(trt_path)

    classes = {0: "cat", 1: "dog"}
    equal_num, time_sum, n = 0, 0.0, 0
    val_loader, _ = build_dataloader(False, args.data_dir, args.batch_size, args.workers, training=False)
    pbar = tqdm(total=len(val_loader), desc="Evaluating trt engine", unit="batch")
    for _, (image, label) in enumerate(val_loader):
        onnx_inputs = {'image': image}
        time_start = time.time()
        result = trt_engine_execute(context, input_name2idx_dict, output_name2idx_dict, bindings, onnx_inputs, \
                                    inputs, outputs)
        time_end = time.time()
        time_sum += (time_end - time_start) * 1000
        pred_class = np.argmax(result['class'])
        predicted, gt = classes[pred_class], classes[label.item()]
        if args.vis:
            visualize_image(image[0], title=f'predicted: {predicted}, ground truth: {gt}')
        pbar.set_postfix({"current infer time": f"{(time_end-time_start)*1000:.4f}ms"})  # 设置进度条信息
        pbar.update(1)
        equal_num += int(predicted == gt)
        n += 1
    
    pbar.close()
    acc = equal_num / n
    print('=' * 40)
    print(f'accuracy: {acc:.4f}')
    print(f'average time: {time_sum / n:.4f} ms')
    print('=' * 40)


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default='ResNet18', choices=['AlexNet', 'ResNet18'], required=False, 
                        help='model name')
    parser.add_argument('--quant', type=int, default=1, help='whether use quant engine')
    parser.add_argument('--mixed_precision', type=int, default=1, help='whether use mixed precision')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--vis', type=int, default=0, required=False, help='visualize the result')
    parser.add_argument('--workers', type=int, default=10, help='number of workers for dataloader')
    parser.add_argument('--data_dir', type=str, default='/data/sfs_turbo/perception/animals/', help='data path')
    parser.add_argument('--output_dir', default='deploy/onnx', help='dir for saving ckpts and log files')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    engine_suffix = '_int8_fp16' if args.mixed_precision else '_int8'
    engine_suffix = engine_suffix if args.quant else ''
    trt_path = f'{args.output_dir}/{args.model}{engine_suffix}.engine'

    eval_trt_engine(args, trt_path)
