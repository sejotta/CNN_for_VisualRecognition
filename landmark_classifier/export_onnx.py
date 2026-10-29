"""Export a trained checkpoint to ONNX for CPU / Raspberry Pi inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import build_model_from_checkpoint, load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a landmark classifier to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("exports/landmarks.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import onnx
    except ImportError as exc:
        raise SystemExit(
            "The 'onnx' package is required for export. "
            "Install it with: pip install -r requirements-deploy.txt"
        ) from exc

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    device = torch.device("cpu")
    model = build_model_from_checkpoint(checkpoint, device)
    model.eval()

    height, width = (int(value) for value in checkpoint["image_size"])
    dummy_input = torch.randn(1, 3, height, width, device=device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(args.output),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    # Bake the preprocessing contract into the ONNX file so deployment code
    # cannot silently use the wrong normalization.
    model_proto = onnx.load(str(args.output))
    metadata = {
        "model_name": str(checkpoint["model_name"]),
        "class_names": json.dumps(checkpoint["class_names"], ensure_ascii=False),
        "image_size": json.dumps([height, width]),
        "mean": json.dumps([float(value) for value in checkpoint["mean"]]),
        "std": json.dumps([float(value) for value in checkpoint["std"]]),
    }
    for key, value in metadata.items():
        prop = model_proto.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save(model_proto, str(args.output))

    print(f"ONNX model written to: {args.output.resolve()}")
    print(f"Input:  input  float32 [batch, 3, {height}, {width}]")
    print(f"Output: logits float32 [batch, {len(checkpoint['class_names'])}]")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed; skipping numeric verification.")
        return 0

    session = ort.InferenceSession(
        str(args.output),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: dummy_input.numpy()})[0]
    with torch.inference_mode():
        torch_logits = model(dummy_input).numpy()
    max_difference = float(np.abs(onnx_logits - torch_logits).max())
    print(f"ONNX verification max |difference|: {max_difference:.3e}")

    if max_difference > 1e-3:
        raise SystemExit("ONNX output differs from PyTorch by more than 1e-3.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
