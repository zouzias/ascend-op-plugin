import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

class TestRFFT(TestCase):
    def test_npu_rfft_meta(self):
        shape = [64,64,1024]
        length = shape[-1]
        x = torch.randn(shape, dtype=torch.float32).npu()
        res = torch.fft.rfft(x, length, norm = "backward") 
        self.assertTrue(res.shape[2] == (length / 2 + 1))
            
if __name__ == "__main__":
    run_tests()