"""Build TensorRT engine from ONNX model (Phase 4.2)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_engine(onnx_path: str, output_path: str, fp16: bool = True, max_batch: int = 1):
    """Build a TensorRT engine from an ONNX model."""
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not available. Install tensorrt package.")
        print(f"Would convert {onnx_path} -> {output_path}")
        return False

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX Parse Error: {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    profile = builder.create_optimization_profile()
    input_shape = network.get_input(0).shape
    if input_shape[0] == -1:
        # Dynamic batch
        min_shape = (1,) + tuple(input_shape[1:])
        opt_shape = (max_batch,) + tuple(input_shape[1:])
        max_shape = (max_batch,) + tuple(input_shape[1:])
        profile.set_shape(network.get_input(0).name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("Engine build failed")
        return False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(engine_bytes)

    print(f"TensorRT engine saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("--onnx", default="yolov8n.onnx", help="ONNX model path")
    parser.add_argument("--output", default="yolov8n.engine", help="TRT engine output path")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16")
    parser.add_argument("--batch", type=int, default=1, help="Max batch size")
    args = parser.parse_args()

    build_engine(args.onnx, args.output, fp16=args.fp16, max_batch=args.batch)


if __name__ == "__main__":
    main()
