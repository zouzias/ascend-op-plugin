import math
import unittest
import copy
import struct
from struct import pack, unpack
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestQuantScatter(TestCase):
    def supported_op_exec(self, tokenA, tokenB, indices, argmax, topRate):
        batch, seqlenA, hiddenSize = tokenA.size()
        _, seqlenB, _ = tokenB.size()
        TopR = int((seqlenA + seqlenB) * topRate)
        afterMergeSeqlenA = seqlenA - TopR
        heads = 8

        output1 = np.random.random((batch, afterMergeSeqlenA, hiddenSize)).astype(np.float16)
        output2 = np.zeros((batch, heads, seqlenB, hiddenSize)).astype(np.float16)
        output3 = np.zeros((batch, heads, seqlenB)).astype(np.float32)

        for b in range(batch):
            for i in range(afterMergeSeqlenA):
                idxA = indices[b][TopR + i]
                output1[b][i] = tokenA[b][idxA]
            output2[b][0] = tokenB[b]

            for i in range(seqlenB):
                output3[b][0][i] = 1.0

            colOneHeads = TopR // heads
            for h in range(heads):
                for col in range(colOneHeads):
                    idxA = indices[b][h * colOneHeads + col]
                    idxB = argmax[b][idxA]
                    output3[b][h][idxB] += 1
                    output2[b][h][idxB] += tokenA[b][idxA]
        output1Npu = torch.from_numpy(output1).to(torch.float16).npu()
        output2Npu = torch.from_numpy(output2).to(torch.float16).npu()
        output3Npu = torch.from_numpy(output3).to(torch.float32).npu()

        return [output1Npu, output2Npu, output3Npu]

    def custom_op_exec(self, tokenA, tokenB, indices, argmax, topRate):
        return torch_npu.npu_tome_merge(tokenA, tokenB, indices, argmax, topRate)

    def generate_input(self, batch, seqlenA, seqlenB, hiddenSize, topRate):
        tokenA = np.random.random((batch, seqlenA, hiddenSize)).astype(np.float16)
        tokenB = np.random.random((batch, seqlenB, hiddenSize)).astype(np.float16)
        indices = np.random.random((batch, seqlenA)).astype(np.int64)
        argmax = np.random.random((batch, seqlenA)).astype(np.int64)

        for i in range(batch):
            indices[i] = np.arange(seqlenA).astype(np.int64)
            argmax[i] = np.random.randint(0, seqlenB - 1, (seqlenA,)).astype(np.int64)

        tokenANpu = torch.from_numpy(tokenA).to(torch.float16).npu()
        tokenBNpu = torch.from_numpy(tokenB).to(torch.float16).npu()
        indicesNpu = torch.from_numpy(indices).to(torch.int64).npu()
        argmaxNpu = torch.from_numpy(argmax).to(torch.int64).npu()
        return [tokenANpu, tokenBNpu, indicesNpu, argmaxNpu]

    @SupportedDevices(['Ascend910B'])
    def test_npu_tome_merge(self, device="npu"):
        tokenA, tokenB, indices, argmax = self.generate_input(4, 3072, 1024, 320, 0.5)

        supported_output = self.supported_op_exec(tokenA.clone(), tokenB.clone(), indices.clone(), argmax.clone())
        custom_output = self.custom_op_exec(tokenA.clone(), tokenB.clone(), indices.clone(), argmax.clone())
        self.assertRtolEqual(supported_output, custom_output, 0.0001)

if __name__ == "__main__":
    run_tests()