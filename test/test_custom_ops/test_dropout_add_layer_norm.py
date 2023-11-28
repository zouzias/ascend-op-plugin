import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device


class TestNPUDropoutAddLayerNorm(TestCase):

    def supported_rms_norm_exec(self, x, gamma, epsilon):
        x = x.detach().numpy()
        gamma = gamma.detach().numpy()
        variance = np.mean(np.power(x, 2), axis=-1, keepdims=True)
        std = np.sqrt(variance + epsilon)
        rstd = 1 / std
        result_mid = x * rstd
        result = result_mid * gamma
        return torch.from_numpy(result), torch.from_numpy(rstd)
        
    def dropout_add_layer_norm_prenorm_training(self, batch_size, seqlen, epsilon):
        input_training = [
            # [hidden_size, input_dtype, residual_dtype, weight_dtype, \
            # dropout_p, has_colscale, has_rowscale, has_residual]
            [192, torch.float32, torch.float32, torch.float32, 0.37, True, True, True],
            [192, torch.float32, torch.float32, torch.float32, 0.37, False, False, False],
            [768, torch.float32, torch.float32, torch.float32, 0.37, True, False, False],
            [768, torch.float32, torch.float32, torch.float32, 0.37, True, True, False], 
        ]

        for item in input_training:
            x0_pt = torch.randn(batch_size, seqlen, item[0], dtype=item[1], requires_grad=True)
            x0 = x0_pt.detach().clone().npu()
            x0.requires_grad_(True)
            x0_ref = x0_pt.detach().clone().float().requires_grad_()

            if item[5]: # has_colscale
                colscale_pt = torch.randn(item[0], dtype=item[3], requires_grad=True)
                colscale = colscale_pt.detach().clone().npu()
                colscale.requires_grad_(True)
                colscale_ref = colscale_pt.detach().clone().float().requires_grad_()
            else:
                colscale = None
            
            if item[7]: # has_residual
                res_pt = torch.randn_like(x0_pt, dtype=item[2], requires_grad=True)
                res = res_pt.detach().clone().npu()
                res.requires_grad_(True)
                res_ref = res_pt.detach().clone().float().requires_grad_()
            else:
                res = None
            
            if item[6]: # has_rowscale
                rowscale_pt = torch.empty(batch_size, seqlen, dtype=item[1])
                survival_rate = 0.87
                rowscale_pt = rowscale_pt.bernoulli_(survival_rate) / survival_rate
                rowscale = rowscale_pt.detach().clone().npu()
                x0_scaled_pt = x0_pt * rearrange(rowscale_pt, "... -> ... 1")
                x0_scaled_ref = x0_ref * rearrange(rowscale_pt, "... -> ... 1")
            else:
                rowscale = None
                x0_scaled_pt = x0_pt
                x0_scaled_ref = x0_ref
            
            if item[5]: # has_colscale
                x0_scaled_pt = x0_scaled_pt * colscale_pt
                x0_scaled_ref = x0_scaled_ref * colscale_ref

            layer_norm_cls = torch.nn.LayerNorm
            model_pt = layer_norm_cls(item[0]).to(dtype=item[3])
            model_ref = layer_norm_cls(item[0]).to(dtype=torch.float32)

            torch.nn.init.normal_(model_pt.weight)
            torch.nn.init.normal_(model_pt.bias)
            our_weight = model_pt.weight.detach().clone().npu()
            our_weight.requires_grad_(True)
            our_bias = model_pt.bias.detach().clone().npu()
            our_bias.requires_grad_(True)
            with torch.no_grad():
                model_ref.weight.copy_(model_pt.weight)
                model_ref.bias.copy_(model_pt.bias)
            
            # forward
            out, pre_norm, dmask = torch_npu.npu_dropout_add_layer_norm(
                x0,
                our_weight,
                residual=res,
                bias=our_bias,
                p=item[4],
                eps=epsilon,
                rowscale=rowscale,
                layerscale=colscale,
                residual_in_fp32=(True if item[2] == torch.float32 else False),
                prenorm=True,
                is_rms_norm=False,
                return_dropout_mask=True,
            )
            assert out.dtype == item[1]
            assert pre_norm.dtype == item[2]

            dmask_pt = dmask.detach().clone().cpu()
            if item[7]: # has_residual
                residual_pt = (
                    (x0_scaled_pt.float() * dmask_pt.float()) / (1 - item[4]) + res_pt.float()
                ).to(dtype=item[2])
                residual_ref = (x0_scaled_ref * dmask_pt.float()) / (1 - item[4]) + res_ref
            else:
                residual_pt = ((x0_scaled_pt.float() * dmask_pt.float()) / (1 - item[4])).to(dtype=item[2])
                residual_ref = (x0_scaled_ref * dmask_pt.float()) / (1 - item[4])
            assert (pre_norm.cpu() - residual_ref).abs().max() <= 4 * (
                residual_pt - residual_ref
            ).abs().max() + 1e-4

            out_pt = model_pt(residual_pt.to(dtype=item[3])).to(dtype=item[1])
            out_ref = model_ref(residual_ref)
            assert (out.cpu() - out_ref).abs().max() <= 4 * (out_pt - out_ref).abs().max() + 1e-4

            # backward
            g_pt = torch.randn_like(out_pt) / batch_size
            (out_pt * F.sigmoid(residual_pt)).backward(g_pt)
            (out.npu() * F.sigmoid(pre_norm.npu())).backward(g_pt.npu())
            (out_ref * F.sigmoid(residual_ref.to(dtype=item[2]))).backward(g_pt)
            assert (x0.grad.cpu() - x0_ref.grad).abs().max() <= 4 * (x0_pt.grad - x0_ref.grad).abs().max() + 1e-4

            if item[7]: # has_residual
                assert (res.grad.cpu() - res_ref.grad).abs().max() <= 4 * (
                    res_pt.grad - res_ref.grad
                ).abs().max() + 1e-4
            
            assert (our_weight.grad.cpu() - model_ref.weight.grad).abs().max() <= 2 * (
                model_pt.weight.grad - model_ref.weight.grad
            ).abs().max() + 2e-4
            
            assert (our_bias.grad.cpu() - model_ref.bias.grad).abs().max() <= 2 * (
                model_pt.bias.grad - model_ref.bias.grad
            ).abs().max() + 2e-4
            
            if item[5]: # has_colscale
                assert (colscale.grad.cpu() - colscale_ref.grad).abs().max() <= 2 * (
                    colscale_pt.grad - colscale_ref.grad
                ).abs().max() + 2e-4

    def dropout_add_layer_norm_eval(self, batch_size, seqlen, epsilon):
        input_eval = [
            # [hidden_size, input_dtype, residual_dtype, weight_dtype, dropout_p]
            [768, torch.float16, torch.float16, torch.float32, 0.0],
            [1024, torch.float32, torch.float32, torch.float32, 0.0]
        ]

        for item in input_eval:
            x0_pt = torch.randn(batch_size, seqlen, item[0], dtype=item[1], requires_grad=True)
            x0 = x0_pt.detach().clone().npu()
            x0.requires_grad_(True)
            x0_ref = x0_pt.detach().clone().float().requires_grad_()

            res_pt = torch.randn_like(x0_pt, dtype=item[2], requires_grad=True)
            res = res_pt.detach().clone().npu()
            res.requires_grad_(True)
            res_ref = res_pt.detach().clone().float().requires_grad_()

            layer_norm_cls = torch.nn.LayerNorm
            model_pt = layer_norm_cls(item[0]).to(dtype=item[3])
            model_ref = layer_norm_cls(item[0]).to(dtype=torch.float32)

            torch.nn.init.normal_(model_pt.weight)
            torch.nn.init.normal_(model_pt.bias)
            our_weight = model_pt.weight.detach().clone().npu()
            our_weight.requires_grad_(True)
            our_bias = model_pt.bias.detach().clone().npu()
            our_bias.requires_grad_(True)
            with torch.no_grad():
                model_ref.weight.copy_(model_pt.weight)
                model_ref.bias.copy_(model_pt.bias)
            
            model_pt.eval()
            model_ref.eval()

            residual_pt = (x0_pt.float() + res_pt.float()).to(dtype=item[2])
            residual_ref = x0_ref + res_ref

            out_pt = model_pt(residual_pt.to(dtype=item[3])).to(item[1])
            out_ref = model_ref(residual_ref)
            out, _, _ = torch_npu.npu_dropout_add_layer_norm(
                x0,
                our_weight,
                residual=res,
                bias=our_bias,
                p=item[4],
                eps=epsilon,
                rowscale=None,
                layerscale=None,
                residual_in_fp32=(True if item[2] == torch.float32 else False),
                prenorm=False,
                is_rms_norm=False,
                return_dropout_mask=False,
            )
            assert (out.cpu() - out_ref).abs().max() <= 4 * (out_pt - out_ref).abs().max() + 1e-4

    def dropout_add_rms_norm_eval(self, batch_size, seqlen, epsilon):
        input_eval = [
            # [hidden_size, input_dtype, residual_dtype, weight_dtype, dropout_p]
            [768, torch.float16, torch.float16, torch.float32, 0.0],
            [1024, torch.float32, torch.float32, torch.float32, 0.0]
        ]

        for item in input_eval:
            x0_pt = torch.randn(batch_size, seqlen, item[0], dtype=item[1], requires_grad=True)
            x0 = x0_pt.detach().clone().npu()
            x0_ref = x0_pt.detach().clone().float()

            res_pt = torch.randn_like(x0_pt, dtype=item[2], requires_grad=True)
            res = res_pt.detach().clone().npu()
            res_ref = res_pt.detach().clone().float()

            weight_pt = torch.nn.Parameter(torch.ones(item[0], dtype=item[3]))
            weight_ref = weight_pt.detach().clone().float()
            our_weight = weight_pt.detach().clone().npu()

            residual_pt = (x0_pt.float() + res_pt.float()).to(dtype=item[2])
            residual_ref = x0_ref + res_ref

            out_pt, _ = self.supported_rms_norm_exec(residual_pt.to(dtype=item[3]), weight_pt, epsilon)
            out_ref, _ = self.supported_rms_norm_exec(residual_ref, weight_ref, epsilon)
            out, _, _ = torch_npu.npu_dropout_add_layer_norm(
                x0,
                our_weight,
                residual=res,
                bias=None,
                p=item[4],
                eps=epsilon,
                rowscale=None,
                layerscale=None,
                residual_in_fp32=(True if item[2] == torch.float32 else False),
                prenorm=False,
                is_rms_norm=True,
                return_dropout_mask=False,
            )
            assert (out.cpu() - out_ref).abs().max() <= 4 * (out_pt.to(item[1]) - out_ref).abs().max() + 1e-4

    def test_dropout_add_layer_norm(self, device="npu"):
        if device is None:
            device = get_npu_device()
        
        rtol, atol = (1e-3, 2e-4)
        # set seed
        torch.random.manual_seed(0)
        batch_size = 8
        seqlen = 51
        epsilon = 1e-5

        self.dropout_add_layer_norm_prenorm_training(batch_size, seqlen, epsilon)
        self.dropout_add_layer_norm_eval(batch_size, seqlen, epsilon)
        self.dropout_add_rms_norm_eval(batch_size, seqlen, epsilon)


if __name__ == "__main__":
    run_tests()
