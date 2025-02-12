import torch
from enum import Enum
import torch.ao.quantization as quant
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization import QConfig, default_weight_fake_quant, default_observer


class FakeQuantState(Enum):
    QAT = "qat"
    CALIBRATION = "calibration"
    VALIDATION = "validation"


class SymmetricObserver(MinMaxObserver):
    def __init__(self, dtype=torch.qint8, quant_min=-128, quant_max=127):
        super(SymmetricObserver, self).__init__(dtype=dtype, quant_min=quant_min, quant_max=quant_max)

    def calculate_qparams(self):
        '''对称量化通常适用于正负范围对称的数据，非对称量化更适用于输入数据分布不均的数据。'''
        # 设置为对称量化
        scale = (self.max_val - self.min_val) / float(self.quant_max - self.quant_min)
        zero_point = 0
        return torch.tensor([scale]), torch.tensor([zero_point], dtype=torch.int32)


def default_calibration_qconfig_setter():
    qconfig = QConfig(
        weight=SymmetricObserver.with_args(dtype=torch.qint8),
        activation=SymmetricObserver.with_args(dtype=torch.qint8),
        )
    return qconfig


def default_qat_qconfig_setter():
    qconfig = QConfig(
        weight=default_weight_fake_quant,
        activation=SymmetricObserver.with_args(dtype=torch.qint8),
        )
    return qconfig


def convert_model_float2calibration(model):
    model.fuse_model()
    model.qconfig = default_calibration_qconfig_setter()
    quant.prepare(model, inplace=True)  # Insert observers


def convert_model_float2qat(args, model):
    if args.model == 'ResNet18':
        model.fuse_model_qat()
    model.qconfig = default_qat_qconfig_setter()
    quant.prepare_qat(model, inplace=True)  # Insert observers and fake quantizers