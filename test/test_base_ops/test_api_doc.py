import torch
import torch.nn.functional as F

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestMaxPool2d(TestCase):

    def test_fast_gelu_doc(self):
        torch_npu.fast_gelu.__doc__
        self.assertTrue(torch_npu.fast_gelu.__doc__, "torch_npu.fast_gelu no __doc__")


if __name__ == "__main__":
    run_tests()
