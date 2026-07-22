"""Generate the tiny ONNX fixtures the onnx-runtime tests load.

Run manually when the fixtures need regenerating (needs the `onnx`
package, which the test env deliberately does not carry):

    uv run --with onnx python tests/fixtures/make_fixture.py

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

    print(f"wrote logreg.onnx ({model.ByteSize()} B, {len(cases)} cases)"
          f" and linreg.onnx ({reg.ByteSize()} B, {len(rcases)} cases)")


if __name__ == "__main__":
    main()
