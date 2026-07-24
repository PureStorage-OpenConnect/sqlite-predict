"""Export amazon/chronos-bolt-small to a single ONNX graph the extension can
serve through predict-onnx.c as a `forecast()` backend.

Chronos-Bolt is a *direct* multi-horizon quantile forecaster (a T5 encoder-
decoder that emits all horizons and quantiles in one forward pass, no
autoregressive decoding), which makes it a clean single-graph ONNX target. The
catch the community hit (github.com/amazon-science/chronos-forecasting/
discussions/272) is that the legacy TorchScript exporter chokes on the model's
NaN-based instance normalization (`aten::nanmean`/`nansum`, used to ignore the
left-padding that rounds a context up to the patch size). The modern **dynamo
exporter** decomposes those ops natively, so no monkeypatching and no numerical
drift: onnxruntime reproduces the Python pipeline to ~1e-6.

Tensor contract (what predict-onnx.c's `sequence` io_spec targets):
  input   context   float32 [1, ctx]   raw series values; the model normalizes
                                        internally (instance norm)
  output  quantiles float32 [1, 9, 64] 9 deciles (0.1..0.9) x up to 64 horizons

CONTEXT LENGTH MUST BE A MULTIPLE OF 16 (the patch size). Chronos-Bolt left-pads
a non-multiple context with NaN and normalizes with nanmean to ignore it; that
NaN path does not survive export faithfully. Feeding a multiple of 16 means zero
padding, so the export is numerically exact. The serving side (predict-onnx.c)
therefore truncates each series to its last floor(L/16)*16 values before the
call -- dropping at most 15 of the oldest points, negligible for any real
context. The gate below only ever feeds multiples of 16, matching that contract.

Chronos-Bolt is Apache-2.0, but the exported graph is a large derivative; keep
it out of git and regenerate with this script.

Run: uv run --with "setuptools<81" --with chronos-forecasting --with torch \
       --with onnx --with onnxscript --with onnxruntime --with numpy \
       python scripts/export_chronos_onnx.py [OUT.onnx]
"""
import sys
import numpy as np
import torch
from chronos import BaseChronosPipeline

MODEL = "amazon/chronos-bolt-small"
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class QuantileHead(torch.nn.Module):
    """Return the bare quantile tensor; ONNX needs a tensor output, not the
    ChronosBoltOutput dataclass the model returns."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, context):
        return self.model(context=context).quantile_preds  # [b, 9, 64]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "chronos_bolt_small.onnx"
    pipe = BaseChronosPipeline.from_pretrained(
        MODEL, device_map="cpu", torch_dtype=torch.float32)
    inner = getattr(pipe, "inner_model", None) or pipe.model
    wrapped = QuantileHead(inner.eval()).eval()

    sample = torch.randn(1, 512, dtype=torch.float32) * 10 + 50
    torch.onnx.export(
        wrapped, (sample,), out,
        input_names=["context"], output_names=["quantiles"],
        dynamic_shapes={"context": {1: torch.export.Dim("ctx", min=16, max=2048)}},
        dynamo=True, opset_version=18)
    print(f"exported {out}")

    # Fidelity gate: onnxruntime must match the reference pipeline across a
    # range of multiple-of-16 context lengths (the serving contract) so a silent
    # export bug can't slip through.
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    worst = 0.0
    for ctx_len in (160, 336, 512, 720, 2048):
        c = torch.randn(1, ctx_len, dtype=torch.float32) * 7 + 30
        got = sess.run(None, {"context": c.numpy()})[0][0, :, :48].T  # [48, 9]
        ref, _ = pipe.predict_quantiles(c[0], prediction_length=48,
                                        quantile_levels=QUANTILES)
        worst = max(worst, float(np.abs(ref[0].numpy() - got).max()))
    print(f"onnxruntime vs pipeline: max abs diff {worst:.2e} "
          f"({'OK' if worst < 1e-3 else 'FAIL'})")
    if worst >= 1e-3:
        sys.exit(1)


if __name__ == "__main__":
    main()
