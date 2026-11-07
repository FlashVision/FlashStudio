"""Export helper tests for the dict-returning FlashDet forward."""

import torch


def test_torchscript_wrapper_returns_preds_tensor():
    from flashstudio.pages.export.weights.exporter import _TorchScriptWrapper

    class Dummy(torch.nn.Module):
        def forward(self, x):
            return {"preds": x * 2, "feat_sizes": [(1, 2)]}

    model = _TorchScriptWrapper(Dummy()).eval()
    x = torch.randn(1, 3, 8, 8)
    traced = torch.jit.trace(model, x)
    out = traced(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
