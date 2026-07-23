"""Run the full TabFM through onnxruntime via external-data ONNX.

Follow-up to tabfm_onnx_eval.py: at 1.64B params / ~6.5 GB fp32, TabFM is
past ONNX's 2 GB single-file protobuf limit, so a servable export must use
external-data format (a small graph .onnx plus a sibling weights blob). This
script exports that way, then loads and runs it in onnxruntime and checks the
logits against PyTorch.

Run (heavy: torch + a 6.5 GB export written to ~/.cache, cleaned up after):

    uv run --with torch --with tabfm --with onnx --with onnxruntime \
        --with onnxscript --with numpy python benchmarks/tabfm_onnx_external.py

The weights and any export are a non-commercial Derivative — never committed.
"""

import gc
import glob
import os
import time
import traceback

import numpy as np
import torch

CKPT = os.path.expanduser("~/.cache/sqlite-predict/tabfm")
OUT = os.path.expanduser("~/.cache/sqlite-predict/tabfm_core.onnx")
N_TRAIN, N_QUERY, H = 100, 50, 6


def patch_repeat_interleave():
    """The one export blocker: train_size.repeat_interleave(hc) (model.py:353)
    has no dim, which the ONNX translator can't convert. A naive dim=0 exports
    but leaves ts's shape metadata inconsistent, so onnxruntime rejects the
    downstream `arange < ts` Less. Rewriting the 1-D no-dim case as
    view/expand/reshape gives the identical values [a,a,b,b,...] with clean
    Expand+Reshape shape inference. The RoPE/mask sites already pass a dim, so
    they pass through untouched."""
    orig = torch.Tensor.repeat_interleave

    def ri(self, repeats, dim=None, **kw):
        if dim is None and self.dim() == 1 and isinstance(repeats, int):
            return self.view(-1, 1).expand(-1, repeats).reshape(-1)
        return orig(self, repeats, dim, **kw) if dim is not None \
            else orig(self, repeats, **kw)

    torch.Tensor.repeat_interleave = ri


def make_inputs():
    rng = np.random.default_rng(0)
    n = N_TRAIN + N_QUERY
    X = rng.standard_normal((n, H)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    args = (torch.from_numpy(X)[None], torch.from_numpy(y)[None],
            torch.tensor([N_TRAIN]), torch.zeros((1, H), dtype=torch.bool),
            torch.tensor([H]))
    feed = {"x": X[None], "y": y[None], "train_size": np.array([N_TRAIN], np.int64),
            "cat_mask": np.zeros((1, H), bool), "d": np.array([H], np.int64)}
    return args, feed, y


def cleanup():
    for f in glob.glob(OUT + "*"):
        os.remove(f)


def main():
    cleanup()
    from tabfm.src.pytorch import tabfm_v1_0_0
    print("# TabFM external-data ONNX run\n")

    patch_repeat_interleave()
    t0 = time.time()
    model = tabfm_v1_0_0.load("classification", checkpoint_path=CKPT,
                              dtype=None)
    model.eval()
    print(f"loaded fp32 in {time.time() - t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")

    args, feed, y = make_inputs()
    with torch.no_grad():
        ref = model(*args).float().numpy()
    print(f"pytorch reference output {list(ref.shape)}")

    # ---- export with external data ----
    print("\n## export")
    try:
        t0 = time.time()
        torch.onnx.export(
            model, args, OUT, dynamo=True,
            input_names=["x", "y", "train_size", "cat_mask", "d"],
            output_names=["logits"])
        graph_mb = os.path.getsize(OUT) / 1e6
        data = sorted(f for f in glob.glob(OUT + "*") if f != OUT)
        data_gb = sum(os.path.getsize(f) for f in data) / 1e9
        print(f"exported in {time.time() - t0:.0f}s: graph {graph_mb:.1f} MB + "
              f"external data {data_gb:.2f} GB across {len(data)} file(s)")
    except Exception:  # noqa: BLE001
        print("EXPORT FAILED:")
        traceback.print_exc()
        cleanup()
        return

    # free the torch model before loading the onnx (peak-memory hygiene)
    del model
    gc.collect()

    # ---- run through onnxruntime ----
    print("\n## onnxruntime run")
    try:
        import onnxruntime as ort
        t0 = time.time()
        sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
        load_s = time.time() - t0
        # warm + timed
        sess.run(None, feed)
        t0 = time.time()
        out = sess.run(None, feed)[0]
        run_ms = (time.time() - t0) * 1000
        out = np.asarray(out).astype(np.float32).reshape(ref.shape)
        max_diff = float(np.abs(out - ref).max())
        # accuracy agreement on the query rows
        q = slice(N_TRAIN, N_TRAIN + N_QUERY)
        n_cls = int(y.max()) + 1
        pt_pred = ref[0, q, :n_cls].argmax(-1)
        ort_pred = out[0, q, :n_cls].argmax(-1)
        agree = float((pt_pred == ort_pred).mean())
        print(f"session load {load_s:.1f}s; forward {run_ms:.0f} ms")
        print(f"max |pytorch - onnx| logit diff: {max_diff:.2e}")
        print(f"query-label agreement pytorch vs onnx: {agree:.3f}")
        # predictions are the product; a small logit diff from op fusion is
        # expected cross-runtime float variance, not a broken export.
        faithful = agree == 1.0 and max_diff < 0.05
        verdict = ("MATCH: runs faithfully (predictions identical)" if faithful
                   else "DIVERGENCE: predictions differ")
        print(f"\n=> {verdict} via external-data ONNX.")
    except Exception:  # noqa: BLE001
        print("ONNXRUNTIME RUN FAILED:")
        traceback.print_exc()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
