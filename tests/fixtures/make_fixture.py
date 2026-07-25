"""Generate the tiny ONNX fixtures the onnx-runtime tests load.

Run manually when the fixtures need regenerating (needs onnx, plus
onnxruntime + numpy to cross-validate the in-context fixture; the test
env deliberately carries none of them):

    uv run --with onnx --with onnxruntime --with numpy \
        python tests/fixtures/make_fixture.py

It writes two self-contained fixtures next to this script:

  logreg.onnx        a 2-feature, 2-class logistic regression:
                     probabilities = softmax(x @ W + b), with a known
                     decision boundary at f1 + f2 = 0 (class 1 above it).
                     input  'float_input'  [N, 2] float32
                     output 'probabilities'[N, 2] float32
  logreg_cases.json  apply rows with the expected label + class-1 prob,
                     computed here in pure Python so the test asserts the
                     extension reproduces them without onnx/numpy at test
                     time.

The tests read these committed files; CI never runs this script.
"""

import json
import math
import os

import onnx
from onnx import TensorProto, helper

HERE = os.path.dirname(__file__)
S = 1.5  # logit scale: sharper boundary => confident, easy-to-check probs


def build_model():
    # logits = x @ W + b, W chosen so class-1 logit = S*(f1+f2), class-0 = 0
    W = helper.make_tensor(
        "W", TensorProto.FLOAT, [2, 2], [-S, S, -S, S])  # columns: [c0, c1]
    b = helper.make_tensor("b", TensorProto.FLOAT, [2], [0.0, 0.0])
    nodes = [
        helper.make_node("MatMul", ["float_input", "W"], ["xw"]),
        helper.make_node("Add", ["xw", "b"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "logreg",
        [helper.make_tensor_value_info(
            "float_input", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info(
            "probabilities", TensorProto.FLOAT, [None, 2])],
        initializer=[W, b],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9  # onnxruntime 1.27 supports IR <= 10; 9 is safe
    onnx.checker.check_model(model)
    return model


def expected(f1, f2):
    # the model's two logits are [-z1, z1] with z1 = S*(f1+f2); reproduce
    # its softmax exactly (numerically stable), so the test's expected
    # values match what onnxruntime computes.
    z1 = S * (f1 + f2)
    m = max(-z1, z1)
    e0, e1 = math.exp(-z1 - m), math.exp(z1 - m)
    p1 = e1 / (e0 + e1)
    # argmax tie-break matches the C backend: class 0 wins when equal
    return ("1" if p1 > 0.5 else "0"), p1


def build_cases():
    pts = []
    # a spread of points on both sides of the boundary, plus near it
    grid = [-3.0, -1.5, -0.4, -0.1, 0.1, 0.4, 1.5, 3.0]
    i = 0
    for a in grid:
        for bv in (-a, a * 0.5, 1.0 - a):
            label, p1 = expected(a, bv)
            pts.append({"id": i, "f1": round(a, 3), "f2": round(bv, 3),
                        "label": label, "p1": p1})
            i += 1
    return pts


LW = [1.0, 2.0]  # linear-regression weights
LB = 0.5         # and intercept


def build_regressor():
    # variable = x @ w + b, output shape [N, 1]
    w = helper.make_tensor("w", TensorProto.FLOAT, [2, 1], LW)
    b = helper.make_tensor("bb", TensorProto.FLOAT, [1], [LB])
    nodes = [
        helper.make_node("MatMul", ["float_input", "w"], ["xw"]),
        helper.make_node("Add", ["xw", "bb"], ["variable"]),
    ]
    graph = helper.make_graph(
        nodes,
        "linreg",
        [helper.make_tensor_value_info(
            "float_input", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info(
            "variable", TensorProto.FLOAT, [None, 1])],
        initializer=[w, b],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def build_reg_cases():
    pts = []
    grid = [-2.0, -0.5, 0.0, 0.75, 3.0]
    i = 0
    for a in grid:
        for bv in (-a, a * 0.5, 1.0):
            pts.append({"id": i, "f1": round(a, 3), "f2": round(bv, 3),
                        "value": LW[0] * a + LW[1] * bv + LB})
            i += 1
    return pts


def build_incontext(k_classes=2):
    """A 1-nearest-neighbour in-context classifier: it reads the training
    rows as context each call and labels each query by its nearest training
    row. Exercises the three-input (x_train, y_train, x_query) marshalling.

    probabilities = one_hot(argmin_j ||x_query_i - x_train_j||^2 -> y_train)
    """
    f = lambda name, shape: helper.make_tensor_value_info(  # noqa: E731
        name, TensorProto.FLOAT, shape)
    inputs = [
        f("x_train", [None, 2]),
        helper.make_tensor_value_info("y_train", TensorProto.INT64, [None]),
        f("x_query", [None, 2]),
    ]
    init = [
        helper.make_tensor("axes1", TensorProto.INT64, [1], [1]),
        helper.make_tensor("two", TensorProto.FLOAT, [], [2.0]),
        helper.make_tensor("depth", TensorProto.INT64, [], [k_classes]),
        helper.make_tensor("oh_vals", TensorProto.FLOAT, [2], [0.0, 1.0]),
    ]
    n = helper.make_node
    nodes = [
        n("Mul", ["x_query", "x_query"], ["xq2"]),
        n("ReduceSum", ["xq2", "axes1"], ["q2"], keepdims=1),          # [Nq,1]
        n("Mul", ["x_train", "x_train"], ["xt2"]),
        n("ReduceSum", ["xt2", "axes1"], ["t2"], keepdims=1),          # [Nt,1]
        n("Transpose", ["t2"], ["t2t"], perm=[1, 0]),                  # [1,Nt]
        n("Transpose", ["x_train"], ["xtT"], perm=[1, 0]),            # [F,Nt]
        n("MatMul", ["x_query", "xtT"], ["qt"]),                       # [Nq,Nt]
        n("Mul", ["qt", "two"], ["two_qt"]),
        n("Add", ["q2", "t2t"], ["s"]),                                # [Nq,Nt]
        n("Sub", ["s", "two_qt"], ["dist"]),                           # [Nq,Nt]
        n("ArgMin", ["dist"], ["nn"], axis=1, keepdims=0),             # [Nq]
        n("Gather", ["y_train", "nn"], ["pred"], axis=0),              # [Nq]
        n("OneHot", ["pred", "depth", "oh_vals"], ["probabilities"], axis=-1),
    ]
    graph = helper.make_graph(
        nodes, "knn1_incontext", inputs,
        [f("probabilities", [None, k_classes])], initializer=init)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def build_incontext_cases():
    import numpy as np
    rng_pts = [(-2.0, -2.0, "0"), (-2.0, 2.0, "0"), (2.0, -2.0, "1"),
               (2.0, 2.0, "1"), (-1.0, 0.0, "0"), (1.0, 0.0, "1"),
               (0.0, -3.0, "0"), (0.0, 3.0, "1")]
    x_train = [[p[0], p[1]] for p in rng_pts]
    y_train = [int(p[2]) for p in rng_pts]
    queries = [(-1.5, -1.0), (1.5, 1.0), (-0.2, 0.2), (0.3, -0.4),
               (-2.1, 2.2), (2.2, -1.9), (0.0, 2.5), (-0.9, -2.8)]

    # independent pure-python 1-NN reference
    def ref(qx, qy):
        best, bi = 1e30, 0
        for j, (tx, ty) in enumerate(x_train):
            d = (qx - tx) ** 2 + (qy - ty) ** 2
            if d < best:
                best, bi = d, j
        return str(y_train[bi])

    expected = [ref(qx, qy) for qx, qy in queries]

    # cross-check against onnxruntime running the actual graph
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(HERE, "knn_incontext.onnx"))
    probs = sess.run(["probabilities"], {
        "x_train": np.array(x_train, dtype=np.float32),
        "y_train": np.array(y_train, dtype=np.int64),
        "x_query": np.array(queries, dtype=np.float32),
    })[0]
    ort_labels = [str(int(p.argmax())) for p in probs]
    assert ort_labels == expected, f"graph != reference: {ort_labels} vs {expected}"

    train = [{"id": j, "f1": x_train[j][0], "f2": x_train[j][1],
              "label": str(y_train[j])} for j in range(len(x_train))]
    apply = [{"id": i, "f1": queries[i][0], "f2": queries[i][1],
              "label": expected[i]} for i in range(len(queries))]
    return {"train": train, "apply": apply}


FQ = [0.1, 0.3, 0.5, 0.7, 0.9]  # forecast fixture quantile levels
FOFF = [-2.0, -1.0, 0.0, 1.0, 2.0]  # per-quantile offset off the context mean
FH = 4  # forecast fixture horizon


def build_forecast():
    """A trivial sequence forecaster exercising the 'sequence' io_spec: it
    forecasts every step as the context mean plus a fixed per-quantile offset.

    input  'context'   [1, ctx] float32   (a context window)
    output 'quantiles' [1, Q, H] float32  (Q quantiles x H steps)
    quantiles[0,q,k] = mean(context) + FOFF[q]
    """
    Q = len(FQ)
    vals = [FOFF[q] for q in range(Q) for _ in range(FH)]  # [1,Q,H], flat in k
    fan = helper.make_tensor("fan", TensorProto.FLOAT, [1, Q, FH], vals)
    nodes = [
        helper.make_node("ReduceMean", ["context"], ["mean"], axes=[1],
                         keepdims=1),                      # [1,1]
        helper.make_node("Add", ["mean", "fan"], ["quantiles"]),  # -> [1,Q,H]
    ]
    graph = helper.make_graph(
        nodes, "forecast",
        [helper.make_tensor_value_info("context", TensorProto.FLOAT, [1, None])],
        [helper.make_tensor_value_info("quantiles", TensorProto.FLOAT,
                                       [1, Q, FH])],
        initializer=[fan])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def build_forecast_cases():
    # forecast() feeds the whole series as context (patch=1, no truncation), so
    # the point is the series mean and the 0.1/0.9 band (conf 0.8) is mean +- 2.
    series = [[10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],  # >= FORECAST_MIN_HISTORY
              [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
              [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0]]
    cases = []
    for sv in series:
        m = sum(sv) / len(sv)
        cases.append({"series": sv, "point": m, "lower": m - 2.0,
                      "upper": m + 2.0})
    return {"quantiles": FQ, "horizon": FH, "conf": 0.8, "cases": cases}


FQ2 = [0.1, 0.3, 0.5, 0.7, 0.9]  # two-head fixture quantile levels
FH2 = 3                          # two-head fixture horizon
TL = 8                           # two-head fixture fixed context length (patch)
# constant point/quantile fans, asymmetric across columns so the flip step is
# actually exercised (col 0 is the point slot, cols 1..5 the deciles).
PP = [0.0, -2.0, -1.0, 0.5, 1.0, 2.5]
QQ = [0.0, -4.0, -1.5, 0.3, 1.4, 4.0]


def build_two_head():
    """A two-head sequence core exercising the declarative reconstruction path:
    the ONNX graph emits a raw (point, quantile) fan and the extension rebuilds
    flip-invariance + continuous head + crossing repair + instance denorm, all
    declared in the io_spec (this is how a TimesFM-style core, whose refinements
    do not survive torch.export, is served in-DB). The fans here are constant, so
    the reconstruction depends on the series only through its mean/std (denorm).

    input   'context'    [1, TL]        float32  (fixed length)
    outputs 'point_fan'  [1, FH2, ncol] float32  (col 0 point, 1..nq quantiles)
            'quant_fan'   [1, FH2, ncol] float32
    """
    ncol = len(FQ2) + 1
    PT = helper.make_tensor("PT", TensorProto.FLOAT, [1, FH2, ncol], PP * FH2)
    QT = helper.make_tensor("QT", TensorProto.FLOAT, [1, FH2, ncol], QQ * FH2)
    nodes = [helper.make_node("Identity", ["PT"], ["point_fan"]),
             helper.make_node("Identity", ["QT"], ["quant_fan"])]
    graph = helper.make_graph(
        nodes, "two_head",
        [helper.make_tensor_value_info("context", TensorProto.FLOAT, [1, TL])],
        [helper.make_tensor_value_info("point_fan", TensorProto.FLOAT,
                                       [1, FH2, ncol]),
         helper.make_tensor_value_info("quant_fan", TensorProto.FLOAT,
                                       [1, FH2, ncol])],
        initializer=[PT, QT])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def build_two_head_cases():
    """Oracle for the two-head reconstruction, computed with onnxruntime + numpy
    (the same reconstruction the extension does in C), so the pure-Python test
    asserts the extension reproduces point/lower/upper without either at test
    time. conf 0.8 -> the 0.1 and 0.9 deciles, which are exact fixture levels."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(build_two_head().SerializeToString(),
                                providers=["CPUExecutionProvider"])
    levels = np.array(FQ2)
    nq = len(FQ2)
    med = 1 + int(np.argmin(np.abs(levels - 0.5)))  # column of the 0.5 quantile

    def flip(fan):  # keep col 0, reverse deciles 1..nq
        return np.concatenate([fan[..., :1], fan[..., 1:][..., ::-1]], axis=-1)

    def reconstruct(series):
        c = np.array(series[-TL:], dtype=np.float32)[None]
        mu = float(c.mean())
        sd = float(np.std(c, ddof=1)) or 1.0
        refl = (2 * mu - c).astype(np.float32)
        a1, b1 = sess.run(None, {"context": c})
        a2, b2 = sess.run(None, {"context": refl})
        ff = (a1 - flip(a2)) / 2
        qs = (b1 - flip(b2)) / 2
        out = qs.copy()
        for cc in range(1, nq + 1):  # continuous head
            out[..., cc] = qs[..., cc] - qs[..., med] + ff[..., med]
        for cc in range(med - 1, 0, -1):  # crossing repair
            out[..., cc] = np.minimum(out[..., cc], out[..., cc + 1])
        for cc in range(med + 1, nq + 1):
            out[..., cc] = np.maximum(out[..., cc], out[..., cc - 1])
        out = out * sd + mu  # instance denorm
        step0 = out[0, 0]  # constant over steps
        return float(step0[med]), float(step0[1]), float(step0[nq])

    series = [[10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
              [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
              [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0]]
    cases = []
    for sv in series:
        p, lo, hi = reconstruct(sv)
        cases.append({"series": sv, "point": p, "lower": lo, "upper": hi})
    return {"quantiles": FQ2, "horizon": FH2, "patch": TL, "conf": 0.8,
            "cases": cases}


def main():
    model = build_model()
    onnx.save(model, os.path.join(HERE, "logreg.onnx"))
    cases = build_cases()
    with open(os.path.join(HERE, "logreg_cases.json"), "w") as f:
        json.dump(cases, f, indent=2)

    reg = build_regressor()
    onnx.save(reg, os.path.join(HERE, "linreg.onnx"))
    rcases = build_reg_cases()
    with open(os.path.join(HERE, "linreg_cases.json"), "w") as f:
        json.dump(rcases, f, indent=2)

    ic = build_incontext()
    onnx.save(ic, os.path.join(HERE, "knn_incontext.onnx"))
    iccases = build_incontext_cases()  # validates the graph vs a py reference
    with open(os.path.join(HERE, "knn_incontext_cases.json"), "w") as f:
        json.dump(iccases, f, indent=2)

    fm = build_forecast()
    onnx.save(fm, os.path.join(HERE, "forecast.onnx"))
    with open(os.path.join(HERE, "forecast_cases.json"), "w") as f:
        json.dump(build_forecast_cases(), f, indent=2)

    th = build_two_head()
    onnx.save(th, os.path.join(HERE, "two_head.onnx"))
    with open(os.path.join(HERE, "two_head_cases.json"), "w") as f:
        json.dump(build_two_head_cases(), f, indent=2)

    print(f"wrote logreg.onnx ({model.ByteSize()} B, {len(cases)} cases),"
          f" linreg.onnx ({reg.ByteSize()} B, {len(rcases)} cases),"
          f" knn_incontext.onnx ({ic.ByteSize()} B,"
          f" {len(iccases['apply'])} queries), forecast.onnx"
          f" ({fm.ByteSize()} B), and two_head.onnx ({th.ByteSize()} B)")


if __name__ == "__main__":
    main()
