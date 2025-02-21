import os, sys, torch, argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from tools.common_utils import visualize_image, build_dataloader

# for compatibility with older numpy versions
if not hasattr(np, 'bool'):
    np.bool = np.bool_


def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def TRTEngineInit(trt_path):
    engine = load_engine(trt_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, input_name2idx_dict, output_name2idx_dict = [], [], [], {}, {}

    num_bindings = engine.num_bindings
    for binding_idx in range(num_bindings):
        binding_name = engine.get_binding_name(binding_idx)
        size = trt.volume(engine.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding_idx))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)  # 分配device内存
        bindings.append(device_mem)
        if engine.binding_is_input(binding_idx):
            inputs.append(host_mem)
            input_name2idx_dict[binding_name] = binding_idx
        else:
            outputs.append(host_mem)
            output_name2idx_dict[binding_name] = binding_idx
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


def eval_one_epoch(args, dataloader, model, loss_fn):
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


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default='ResNet18', choices=['AlexNet', 'ResNet18'], required=False, 
                        help='model name')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--vis', type=int, default=1, required=False, help='visualize the result')
    parser.add_argument('--workers', type=int, default=10, help='number of workers for dataloader')
    parser.add_argument('--data_dir', type=str, default='/data/sfs_turbo/perception/animals/', help='data path')
    parser.add_argument('--output_dir', default='deploy/onnx', help='dir for saving ckpts and log files')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    trt_path = f'{args.output_dir}/{args.model}_int8.engine'
    inputs, outputs, bindings, context, input_name2idx_dict, output_name2idx_dict, _ = TRTEngineInit(trt_path)

    val_loader, _ = build_dataloader(False, args.data_dir, args.batch_size, args.workers, training=False)
    classes = {0: "cat", 1: "dog"}
    equal_num, n = 0, 0
    for i, (image, label) in enumerate(val_loader):
        onnx_inputs = {'image': image}
        result = trt_engine_execute(context, input_name2idx_dict, output_name2idx_dict, bindings, onnx_inputs, \
                                    inputs, outputs)
        pred_class = np.argmax(result['class'])
        predicted, gt = classes[pred_class], classes[label.item()]
        # visualize
        if args.vis:
            visualize_image(image[0], title=f'predicted: {predicted}, ground truth: {gt}')
        equal_num += int(predicted == gt)
        n += 1
    acc = equal_num / n
    print(f'accuracy: {acc}')
