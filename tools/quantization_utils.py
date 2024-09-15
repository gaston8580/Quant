import torch
from torch.quantization.observer import MinMaxObserver


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
    qconfig = torch.quantization.QConfig(
        weight=SymmetricObserver.with_args(dtype=torch.qint8),
        activation=SymmetricObserver.with_args(dtype=torch.qint8),
        )
    return qconfig