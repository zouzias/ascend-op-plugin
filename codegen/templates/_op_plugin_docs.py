import types

import torch._C
from torch._C import _add_docstr as add_docstr
 
import torch_npu


def _add_torch_npu_docstr(method, docstr):
    """Add doc to operator API.
    If implementing the Python side interface with pybind11, _add_docstr is needed to add doc.
    """
    func = getattr(torch_npu, method, None)
    if not func:
        return
    # PT1.11/2.0 requires the use of _add_doc
    if isinstance(func, types.BuiltinMethodType):
        add_docstr(func, docstr)
    else:
        getattr(torch_npu, method).__doc__ = docstr


_add_torch_npu_docstr(
    "fast_gelu",
    """
torch_npu.fast_gelu(self) -> Tensor

计算输入张量中fast_gelu的梯度。支持FakeTensor模式。

参数说明
self (Tensor) - 数据类型：float16、float32。

示例
示例一：

>>> x = torch.rand(2).npu()
>>> x
tensor([0.5991, 0.4094], device='npu:0')
>>> torch_npu.fast_gelu(x)
tensor([0.4403, 0.2733], device='npu:0')
示例二：

//FakeTensor模式
>>> from torch._subclasses.fake_tensor import FakeTensorMode
>>> with FakeTensorMode():
...     x = torch.rand(2).npu()
...     torch_npu.fast_gelu(x)
>>> FakeTensor(..., device='npu:0', size=(2,))

"""
)
