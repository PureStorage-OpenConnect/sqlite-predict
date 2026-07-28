# TabFM → ONNX export eval

Can the real Google TabFM run through `sqlite-predict`'s ONNX backend?
Short answer: **not as-is.** It runs in PyTorch, but it does not fit the
extension's `in_context` contract, and it does not cleanly export to ONNX.

Measured locally on CPU (macOS, Apple silicon), float32, against the
`google/tabfm-1.0.0-pytorch` classification checkpoint loaded from disk.
The weights and any export are a non-commercial Derivative and are not
committed; `benchmarks/tabfm_onnx_eval.py` reproduces these numbers from a
local checkpoint.

## It runs

| thing | value |
| --- | --- |
| parameters | 1.64B |
| load (fp32, local weights) | 5.1 s |
| one ensemble-member forward (CPU, fp32, 200-ctx + 100-query, 6 feat) | 949 ms |
| full ensemble `predict` (measured earlier, `comparison-tabular.md`) | 33–42 s |
| raw-model accuracy, easy 2-class task, no quantile transform | 1.000 |
| output shape | `[B, T, 10]` (`max_classes = 10`) |

The 949 ms is a single ensemble member on a tiny context. The
`TabFMClassifier` runs a whole ensemble of feature-permuted views and
blends them with NNLS weights, which is where the tens-of-seconds figure
comes from. Accuracy is a floor here: the easy task is linearly separable,
so the model is right even without the preprocessing it normally leans on.

## It does not fit the `in_context` contract

The forward is:

```
model(x, y, train_size, cat_mask=, d=)
  x          [B, T, H]   ensemble batch, train+query packed into ONE sequence
  y          [B, T]      labels (query positions masked out via train_size)
  train_size [B]         where the context ends and the queries begin
  cat_mask   [B, H]      which columns are categorical
  d          [B]         real feature count (H is padded)
```

The extension's `in_context` layout feeds three separate tensors
(`x_train`, `y_train`, `x_query`) of raw feature rows. TabFM instead wants a
single packed `X` with a `train_size` delimiter, plus a categorical mask and
a feature count, and it expects those features to have already been through
an sklearn pipeline (categorical ordinal encoding, an RTDL quantile
transform, outlier handling) and expanded into an *ensemble* of views whose
outputs are blended. None of that preprocessing or ensembling can live
inside a SQL extension.

## Exporting it takes work, but it does run

Getting a loadable ONNX out of TabFM took two fixes, both reproduced in
`benchmarks/tabfm_onnx_external.py`:

1. **One unexportable op.** The torch→ONNX translator cannot convert
   `aten.repeat_interleave` with no `dim` (the model's
   `train_size.repeat_interleave(hc)` in `ColEmbedding`, model.py:353):
   `NotImplementedError: No conversion available yet when dim is None`. A
   naive `dim=0` exports but leaves `ts`'s shape metadata inconsistent, so
   onnxruntime rejects the downstream `arange < ts` with a `Less`
   `[ShapeInferenceError] Incompatible dimensions`. Rewriting that one call
   as `view(-1,1).expand(-1,hc).reshape(-1)` (identical values, clean
   `Expand`+`Reshape` shape inference) fixes it. It is a one-line model edit,
   not something the extension can do.
2. **External-data format.** At 1.64B params, fp32 weights are ~6.5 GB, past
   ONNX's 2 GB single-file protobuf limit. A self-contained `.onnx` is
   impossible; the export splits into an 8.5 MB graph plus a 6.56 GB external
   weights blob. Neither the inline-BLOB student path nor a single-file
   `weights_uri` assumes that split.

With both fixes, the full model runs in onnxruntime and is numerically
faithful:

| thing | value |
| --- | --- |
| export (dynamo, external data) | ~35 s |
| graph / external data | 8.5 MB / 6.56 GB |
| onnxruntime session load | ~13 s (one-time) |
| forward (CPU, fp32, 1 ensemble member) | 798 ms |
| max &#124;pytorch − onnx&#124; logit diff | 1.3e-2 |
| query-label agreement pytorch vs onnx | **100%** |

The ~0.01 logit gap is ordinary cross-runtime float variance from operator
fusion; every prediction is identical, so this is not the silent MPS-style
divergence. The 798 ms is one ensemble member on a 150-row context; the full
`TabFMClassifier` ensemble is still the tens-of-seconds figure.

## Verdict

TabFM *can* be exported to ONNX and runs faithfully, but serving it directly
through the extension would still require: a one-line model patch to export,
an external-data ONNX over 2 GB, an `io_spec` expanded with
`train_size`/`cat_mask`/`d` (or a wrapper graph that internalizes the
packing), and the sklearn preprocessing + ensemble pushed into SQL or the
graph. That is a large amount of work and machinery to serve a model that
still costs tens of seconds per call.

This is exactly the case the teacher/student split was built for. The
in-context ONNX path we shipped is proven on a synthetic 1-NN teacher and is
the right shape for a *distilled* student; TabFM's role is the offline
teacher, run with its full Python pipeline, whose skill we compress into a
small vector or in-context student that fits the extension and serves in
microseconds. Nothing here changes that conclusion — it sharpens it.
