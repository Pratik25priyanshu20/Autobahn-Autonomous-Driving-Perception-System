import argparse


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine (stub)")
    parser.add_argument("--onnx", default="models/model.onnx", help="ONNX path")
    parser.add_argument("--output", default="models/model.plan", help="Engine output path")
    args = parser.parse_args()
    print(f"Would convert {args.onnx} to TensorRT engine at {args.output}")


if __name__ == "__main__":
    main()
