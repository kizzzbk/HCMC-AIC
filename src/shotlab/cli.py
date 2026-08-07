import argparse
from pathlib import Path

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=["pyscenedetect", "transnetv2"], default=["pyscenedetect"])
    parser.add_argument("--primary", choices=["pyscenedetect", "transnetv2"], default="pyscenedetect")
    parser.add_argument("--pyscene-threshold", type=float, default=27.0)
    parser.add_argument("--transnet-threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--long-shot-seconds", type=float, default=3.0)
    parser.add_argument("--sample-every-seconds", type=float, default=1.0)
    values = vars(parser.parse_args())
    values["input_path"] = values.pop("input")
    values["output_root"] = values.pop("output")
    run_pipeline(**values)


def embed_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    
    args = parser.parse_args()
    from .embed import embed_keyframes
    embed_keyframes(
        output_root=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()

