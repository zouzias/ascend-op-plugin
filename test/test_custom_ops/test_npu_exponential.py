import numpy as np
import torch
import torch_npu
from scipy.stats import kstest


def cal_reject_num(alpha, n):
    z = -3.0902
    rate = (1 - alpha) + z * pow((1 - alpha) * alpha / n, 0.5)
    reject_num = (1 - rate) * n
    return reject_num


def main():
    N = 100
    alpha = 0.01
    count = 0
    for i in range(N):
        k = np.random.randint(1, 5)
        shape = tuple(np.random.randint(10, 100, size=(k, )))
        tensor_cpu = torch.rand(size=shape)
        tensor_npu = tensor_cpu.npu()

        tensor_cpu = tensor_cpu.exponential_().numpy()
        tensor_npu = tensor_npu.exponential_().cpu().numpy()

        test_output = kstest(tensor_cpu.flatten(), tensor_npu.flatten())
        if test_output.pvalue < alpha:
            count += 1
    print(count)
    reject_num = cal_reject_num(alpha, N)
    print(reject_num)
    
    assert count <= reject_num
       


if __name__ == "__main__":
    main()



