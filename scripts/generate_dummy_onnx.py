"""
Generate a very simple ONNX model for testing jetson-toolbox.

Model:
  input:  (1, 3, 224, 224) float32
  output: (1, 3, 224, 224) float32
  op: Identity (just forwards the tensor)
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def build_identity_model() -> onnx.ModelProto:
    input_tensor = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, 3, 224, 224],  # NCHW
    )
    output_tensor = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1, 3, 224, 224],
    )

    node = helper.make_node(
        "Identity",
        inputs=["input"],
        outputs=["output"],
        name="IdentityNode",
    )

    graph = helper.make_graph(
        nodes=[node],
        name="DummyIdentityModel",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[],
    )

    opset = helper.make_operatorsetid("", 13)
    model = helper.make_model(graph, opset_imports=[opset], ir_version=11)
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "dummy_identity_1x3x224x224.onnx"
    model = build_identity_model()
    onnx.save(model, out_path)

    print(f"Saved dummy ONNX model to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
