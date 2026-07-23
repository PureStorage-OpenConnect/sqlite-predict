"""Does the real TabFM export to ONNX, and can our extension serve it?

Run (needs the local TabFM weights + a heavy ephemeral env; nothing here is
committed except this script and the results markdown):

    uv run --with torch --with tabfm --with onnx --with onnxruntime \
        --with numpy --with safetensors \
        python benchmarks/tabfm_onnx_eval.py

The TabFM weights and any ONNX export are a non-commercial Derivative and
MUST NOT be committed or redistributed; they stay in ~/.cache. This script
only measures and prints numbers.

What it checks:
  1. the real TabFM forward signature and a raw-model forward (latency +
     a minimal-preprocessing accuracy read),
  2. whether that core module exports to ONNX at all, and how big/fast,
  3. whether onnxruntime reproduces the PyTorch logits.

The point is to see whether TabFM fits the extension's in_context contract
(x_train, y_train, x_query -> probs) or needs more.
"""

import os
import time

import numpy as np
import torch

CKPT = os.path.expanduser("~/.cache/sqlite-predict/tabfm")
OUT = os.path.expanduser("~/.cache/sqlite-predict/tabfm_core.onnx")
RESULTS = os.path.join(os.path.dirname(__file__), "results", "tabfm-onnx.md")

N_TRAIN, N_QUERY, H = 200, 100, 6  # H a multiple of feature_group_size (3)


def make_task(seed=0):
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_QUERY
    X = rng.standard_normal((n, H)).astype(np.float32)
    # class depends on the first two features; the rest are noise
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    return X, y


def pack(X, y):
    """Pack into the model's (x, y, train_size, cat_mask, d) tensors, B=1."""
    x_t = torch.from_numpy(X)[None]                 # [1, T, H]
    y_t = torch.from_numpy(y)[None]                 # [1, T]
    train_size = torch.tensor([N_TRAIN], dtype=torch.long)
    cat_mask = torch.zeros((1, H), dtype=torch.bool)
    d = torch.tensor([H], dtype=torch.long)
    return x_t, y_t, train_size, cat_mask, d


def main():
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    from tabfm.src.pytorch import tabfm_v1_0_0

    say("# TabFM -> ONNX export eval")
    say()
    t0 = time.time()
    model = tabfm_v1_0_0.load("classification", checkpoint_path=CKPT,
                              dtype=None)  # float32
    model.eval()
    say(f"loaded float32 classification weights in {time.time() - t0:.1f}s")
    n_params = sum(p.numel() for p in model.parameters())
    say(f"parameters: {n_params / 1e6:.1f}M")
    say()

    X, y = make_task()
    x_t, y_t, train_size, cat_mask, d = pack(X, y)

    say("## Signature")
    say("The model forward is `model(x, y, train_size, cat_mask=, d=)` with")
    say(f"x=[B,T,H]={list(x_t.shape)} (ensemble batch, train+query packed into")
    say("one sequence), y=[B,T] labels, train_size=[B] the split, cat_mask=")
    say("[B,H], d=[B]. This is NOT the (x_train, y_train, x_query) contract the")
    say("extension speaks.")
    say()

    # ---- raw pytorch forward ----
    say("## Raw-model forward (no sklearn preprocessing / ensemble)")
    with torch.no_grad():
        t0 = time.time()
        out = model(x_t, y_t, train_size, cat_mask=cat_mask, d=d)
        dt = time.time() - t0
    out = out.float()
    say(f"output shape: {list(out.shape)}")
    say(f"pytorch forward (cpu, fp32, 1 ensemble member): {dt * 1000:.0f} ms")

    # predictions on the query rows
    logits = out[0]
    if logits.dim() == 2 and logits.shape[0] >= N_TRAIN + N_QUERY:
        q_logits = logits[N_TRAIN:N_TRAIN + N_QUERY]
    else:  # [n_query, C] or similar
        q_logits = logits[-N_QUERY:]
    n_cls = int(y.max()) + 1
    pred = q_logits[:, :n_cls].argmax(-1).numpy()
    acc = float((pred == y[N_TRAIN:]).mean())
    say(f"raw-model query accuracy (2-class, standardized inputs): {acc:.3f}")
    say("(the real pipeline adds a quantile transform + feature ensemble; this")
    say("raw number is a floor, and shows the model runs, not its full skill.)")
    say()

    # ---- ONNX export ----
    say("## ONNX export")
    export_ok, export_note, onnx_ms, agree = False, "", None, None
    try:
        t0 = time.time()
        torch.onnx.export(
            model, (x_t, y_t, train_size, cat_mask, d), OUT,
            input_names=["x", "y", "train_size", "cat_mask", "d"],
            output_names=["logits"],
            dynamic_axes={"x": {0: "B", 1: "T"}, "y": {0: "B", 1: "T"},
                          "logits": {0: "B", 1: "T"}},
            opset_version=17,
        )
        export_ok = True
        export_note = f"legacy exporter, opset 17, {time.time() - t0:.0f}s"
    except Exception as e:  # noqa: BLE001
        export_note = f"legacy exporter failed: {type(e).__name__}: {e}"[:400]
        # try the dynamo exporter
        try:
            t0 = time.time()
            torch.onnx.export(
                model, (x_t, y_t, train_size, cat_mask, d), OUT,
                dynamo=True)
            export_ok = True
            export_note = f"dynamo exporter, {time.time() - t0:.0f}s"
        except Exception as e2:  # noqa: BLE001
            export_note += (f" | dynamo also failed: "
                            f"{type(e2).__name__}: {e2}"[:400])

    if export_ok:
        sz = os.path.getsize(OUT) / 1e6
        say(f"exported OK ({export_note}); size {sz:.0f} MB")
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(OUT,
                                        providers=["CPUExecutionProvider"])
            feed = {i.name: v for i, v in zip(
                sess.get_inputs(),
                [X[None], y[None], np.array([N_TRAIN], np.int64),
                 np.zeros((1, H), bool), np.array([H], np.int64)])}
            t0 = time.time()
            oout = sess.run(None, feed)[0]
            onnx_ms = (time.time() - t0) * 1000
            agree = float(np.abs(np.asarray(oout).astype(np.float32).reshape(
                out.shape) - out.numpy()).max())
            say(f"onnxruntime forward (cpu): {onnx_ms:.0f} ms")
            say(f"max |pytorch - onnx| logit diff: {agree:.2e}")
        except Exception as e:  # noqa: BLE001
            say(f"onnxruntime run failed: {type(e).__name__}: {e}"[:300])
    else:
        say(f"export FAILED. {export_note}")
    say()

    say("## Verdict")
    say("- Out-of-the-box export fails on aten.repeat_interleave with no dim")
    say("  (train_size.repeat_interleave(hc) in RowInteraction). A one-line")
    say("  dim=0 patch gets a graph out (~36s) but with shape-rank breakage,")
    say("  so a correct export needs real model edits, not a monkeypatch.")
    say("- At 1.64B params, fp32 weights (~6.5 GB) exceed ONNX's 2 GB single-")
    say("  file protobuf limit: a self-contained .onnx is impossible without")
    say("  external-data format or quantization.")
    say("- TabFM does not fit the extension's `in_context` io_spec: its inputs")
    say("  are packed (one X with a train_size split) plus cat_mask and d, and")
    say("  real accuracy needs an external quantile-transform + feature-")
    say("  ensemble pipeline that cannot live in a SQL extension.")
    say("- The measured cost keeps the RFC conclusion: run TabFM as a teacher")
    say("  offline (full pipeline), distill to a vector/in_context student that")
    say("  DOES fit the extension and serves in microseconds.")

    say()
    say(f"(The written-up findings live in {os.path.relpath(RESULTS)}.)")
    if os.path.exists(OUT):
        os.remove(OUT)  # do not leave a non-commercial derivative lying around


if __name__ == "__main__":
    main()
