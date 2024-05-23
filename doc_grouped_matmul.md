# 功能描述

GroupedMatmul算子可以实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同，是一种灵活的支持方式。其主要输入与输出均为TensorList，其中输入数据x与输出结果y均支持切分及不切分的模式，根据参数split_item来确定x与y是否需要切分，在x需要切分的情况下使用参数group_list来描述对x的m轴进行切分的方式。
根据输入x、输入weight与输出y的Tensor数量不同，可以支持如下4种场景：
x、weight、y的Tensor数量等于组数，即每组的数据对应的Tensor是独立的。
x的Tensor数量为1，weight/y的Tensor数量等于组数，此时需要通过可选属性group_list说明x在行上的分组情况，如group_list[0]=10说明x的前10行参与第一组矩阵乘计算。
x、weight的Tensor数量等于组数，y的Tensor数量为1，此时每组矩阵乘的结果放在同一个Tensor中连续存放。
x、y的Tensor数量为1，weight数量等于组数，属于前两种情况的组合。
计算公式为：

# 接口原型

PyTorch 2.1及更高的版本中：
npu_grouped_matmul(Tensor[] x, Tensor[] weight, *, Tensor[]? bias=None, Tensor[]? scale=None, Tensor[]? offset=None, Tensor[]? antiquant_scale=None, Tensor[]? antiquant_offset=None, int[]? group_list=None, int? split_item=0, ScalarType? output_dtype=None) -> Tensor[]
PyTorch 1.11与2.0版本：
npu_grouped_matmul(Tensor[] x, Tensor[] weight, *, Tensor[] bias, Tensor[] scale, Tensor[] offset, Tensor[] antiquant_scale, Tensor[] antiquant_offset, int[]? group_list=None, int? split_item=0, ScalarType? output_dtype=None) -> Tensor[]

# 参数说明

x：必选参数，Device侧的TensorList，即输入参数中的x，在Ascend910B与Ascend910C上数据类型支持FLOAT16、BFLOAT16、INT8、FLOAT32，数据格式支持ND，在Ascend310P上数据类型支持FLOAT16，数据格式支持ND，支持的最大长度为128个，其中每个Tensor在split_item=0的模式下支持输入2至6维，其余模式下支持输入为2维。
weight：必选参数，Device侧的TensorList，即输入参数中matmul的weight输入，在Ascend910B与Ascend910C上数据类型支持FLOAT16、BFLOAT16、INT8、FLOAT32，数据格式支持ND，在Ascend310P上数据类型支持FLOAT16，数据格式支持FRACTAL_NZ，支持的最大长度为128个，其中每个Tensor支持输入为2维。
bias：在PyTorch 1.11与2.0版本中是必选参数，在PyTorch 2.1与更高的版本中是可选参数，Device侧的TensorList，即输入参数中matmul的bias输入，在Ascend910B与Ascend910C上数据类型支持FLOAT16、FLOAT32、INT32，数据格式支持ND，在Ascend310P上数据类型支持FLOAT16，数据格式支持ND，支持的最大长度为128个，其中每个Tensor支持输入为1维。
scale：可选参数，Device侧的TensorList，代表量化参数中的缩放因子，目前仅支持Ascend910B与Ascend910C，数据类型支持INT64，数据格式支持ND，长度与weight相同。
offset：可选参数，Device侧的TensorList，代表量化参数中的偏移量，目前仅支持Ascend910B与Ascend910C，数据类型支持FLOAT32，数据格式支持ND，长度与weight相同。
antiquantScale：可选参数，Device侧的TensorList，代表伪量化参数中的缩放因子，目前仅支持Ascend910B与Ascend910C，数据类型支持FLOAT16、BFLOAT16，数据格式支持ND，长度与weight相同。
antiquantOffset：可选参数，Device侧的TensorList，代表伪量化参数中的偏移量，目前仅支持Ascend910B与Ascend910C，数据类型支持FLOAT16、BFLOAT16，数据格式支持ND，长度与weight相同。
group_list：可选参数，Host侧的IntArray类型，是切分的索引，代表输入和输出M方向的matmul索引情况，数据类型支持INT64，数据格式支持ND，支持输入为1维，支持的最大长度为128个，默认为空。
split_item：可选属性，Int类型，切分模式的说明，数据类型支持INT32，可取的值有4个：0和1表示输出不需要切分，2和3表示输出需要进行切分。默认值为0。
output_dtype：可选属性，ScalarType类型，用于指定输出的数据类型，默认值为None，表明输出与输入是同一数据类型。

# 输出说明

Device侧的TensorList类型输出，代表GroupedMatmul的计算结果，当split_item取0或1时，其Tensor个数与weight相同，当split_item取2或3时，其Tensor个数为1。

# 约束说明

若x为多Tensor，group_list可以为空；当x为单Tensor，group_list的长度与weight的Tensor个数相同。
若bias不为空，其Tensor数量须与weight保持一致。
记一个matmul计算涉及的x、weight与y的维度分别为(m×k)、(k×n)和(m×n)，每一个matmul的输入与输出须满足[m, k]和[k, n]的k维度相等关系。
非量化场景支持的输入类型为：
x为FLOAT16、weight为FLOAT16、bias为FLOAT16、scale为空、offset为空、antiquant_scale为空、antiquant_offset为空、output_dtype为FLOAT16；
x为BFLOAT16、weight为BFLOAT16、bias为FLOAT32、scale为空、offset为空、antiquant_scale为空、antiquant_offset为空、output_dtype为BFLOAT16（当前仅在Ascend910B与Ascend910C上支持）；
x为FLOAT32、weight为FLOAT32、bias为FLOAT32、scale为空、offset为空、antiquant_scale为空、antiquant_offset为空、output_dtype为FLOAT32（当前仅在Ascend910B与Ascend910C上支持）；
当前仅在Ascend910B与Ascend910C上支持量化场景，支持的输入类型为：
x为INT8、weight为INT8、bias为INT32、scale为UINT64、offset为空、antiquant_scale为空、antiquant_offset为空、output_dtype为INT8；
当前仅在Ascend910B与Ascend910C上支持伪量化场景，支持的输入类型为：
x为FLOAT16、weight为INT8、bias为FLOAT16、scale为空，offset为空，antiquant_scale为FLOAT16、antiquant_offset为FLOAT16、output_dtype为FLOAT16；
x为BFLOAT16、weight为INT8、bias为FLOAT32、scale为空，offset为空，antiquant_scale为BFLOAT16、antiquant_offset为BFLOAT16、output_dtype为BFLOAT16；
对于实际无bias的场景，在PyTorch 1.11与2.0版本中，须手动指定“bias=[]”；在PyTorch 2.1及更高的版本中，可以直接不指定bias参数。scale、offset、antiquantScale、antiquantOffset四个参数在不同PyTorch版本中的约束与bias相同。
output_dtype的数据类型当前只支持None，或者与输入x的数据类型相同。

# 支持的PyTorch版本

PyTorch 2.3
PyTorch 2.2
PyTorch 2.1
PyTorch 2.0
PyTorch 1.11

# 支持的型号

Atlas A2 训练系列产品
昇腾910C AI处理器
Atlas 推理系列产品（Ascend 310P处理器）

# 调用示例

# 单算子调用模式，Torch2.1及更高的版本
import torch
import torch_npu
x1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
x2 = torch.randn(1024, 256, device='npu', dtype=torch.float16)
x3 = torch.randn(512, 1024, device='npu', dtype=torch.float16)
x = [x1, x2, x3]
weight1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
weight2 = torch.randn(256, 1024, device='npu', dtype=torch.float16)
weight3 = torch.randn(1024, 128, device='npu', dtype=torch.float16)
weight = [weight1, weight2, weight3]
bias1 = torch.randn(256, device='npu', dtype=torch.float16)
bias2 = torch.randn(1024, device='npu', dtype=torch.float16)
bias3 = torch.randn(128, device='npu', dtype=torch.float16)
bias = [bias1, bias2, bias3]
group_list = None
split_item = 0
npu_out = torch_npu.npu_grouped_matmul(x, weight, bias=bias, group_list=group_list, split_item=split_item)

# 图模式调用，Torch2.1及更高的版本
import torch
import torch.nn as nn
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)
class GMMModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, weight):
        return torch_npu.npu_grouped_matmul(x, weight)
def main():
    x1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
    x2 = torch.randn(1024, 256, device='npu', dtype=torch.float16)
    x3 = torch.randn(512, 1024, device='npu', dtype=torch.float16)
    x = [x1, x2, x3]
    weight1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
    weight2 = torch.randn(256, 1024, device='npu', dtype=torch.float16)
    weight3 = torch.randn(1024, 128, device='npu', dtype=torch.float16)
    weight = [weight1, weight2, weight3]
    model = GMMModel().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    custom_output = model(x, weight)

if __name__ == '__main__':
    main()