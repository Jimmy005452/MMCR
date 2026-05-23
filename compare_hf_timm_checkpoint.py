import argparse
import json
from pathlib import Path

import torch

from mmcr.models import DEFAULT_ARCH, build_image_encoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="tanganke/clip-vit-large-patch14_sun397")
    parser.add_argument("--download-root", default="checkpoint_another")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--report", default=None)
    parser.add_argument("--show-keys", type=int, default=20)
    return parser.parse_args()


def safe_repo_dir(repo_id: str):
    return repo_id.replace("/", "__")


def download_repo(repo_id: str, download_root: Path):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Please install huggingface_hub: pip install huggingface_hub") from exc

    local_dir = download_root / safe_repo_dir(repo_id)
    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
        )
    )


def find_weight_file(repo_dir: Path):
    safetensors_files = sorted(repo_dir.rglob("*.safetensors"))
    if safetensors_files:
        return safetensors_files[0]

    bin_files = sorted(repo_dir.rglob("*.bin"))
    if bin_files:
        return bin_files[0]

    pt_files = sorted(repo_dir.rglob("*.pt"))
    if pt_files:
        return pt_files[0]

    raise FileNotFoundError(f"No .safetensors, .bin, or .pt weight file found under {repo_dir}")


def load_hf_state_dict(weight_path: Path):
    if weight_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Please install safetensors: pip install safetensors") from exc
        return load_file(str(weight_path))

    obj = torch.load(weight_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint object type: {type(obj)}")


def summarize_overlap(hf_state: dict, timm_state: dict):
    hf_keys = set(hf_state)
    timm_keys = set(timm_state)
    direct_overlap = sorted(hf_keys & timm_keys)
    missing_in_hf = sorted(timm_keys - hf_keys)
    extra_in_hf = sorted(hf_keys - timm_keys)

    same_shape_overlap = []
    shape_mismatch = []
    for key in direct_overlap:
        if hf_state[key].shape == timm_state[key].shape:
            same_shape_overlap.append(key)
        else:
            shape_mismatch.append(
                {
                    "key": key,
                    "hf_shape": list(hf_state[key].shape),
                    "timm_shape": list(timm_state[key].shape),
                }
            )

    return {
        "hf_num_keys": len(hf_keys),
        "timm_num_keys": len(timm_keys),
        "direct_overlap_count": len(direct_overlap),
        "same_shape_overlap_count": len(same_shape_overlap),
        "shape_mismatch_count": len(shape_mismatch),
        "direct_overlap": direct_overlap,
        "same_shape_overlap": same_shape_overlap,
        "shape_mismatch": shape_mismatch,
        "missing_in_hf": missing_in_hf,
        "extra_in_hf": extra_in_hf,
    }


def print_key_examples(title: str, keys, limit: int):
    print(f"\n{title}")
    for key in list(keys)[:limit]:
        print(f"  {key}")


def main():
    args = parse_args()
    download_root = Path(args.download_root)

    print(f"Downloading {args.repo_id} to {download_root} ...")
    repo_dir = download_repo(args.repo_id, download_root)
    weight_path = find_weight_file(repo_dir)
    print(f"Found weight file: {weight_path}")

    print("Loading Hugging Face checkpoint ...")
    hf_state = load_hf_state_dict(weight_path)

    print(f"Building timm model: {args.arch}")
    timm_model = build_image_encoder(arch=args.arch, pretrained=False)
    timm_state = timm_model.state_dict()

    report = summarize_overlap(hf_state, timm_state)
    report["repo_id"] = args.repo_id
    report["repo_dir"] = str(repo_dir)
    report["weight_path"] = str(weight_path)
    report["arch"] = args.arch

    print("\n=== Summary ===")
    print(f"HF keys: {report['hf_num_keys']}")
    print(f"timm keys: {report['timm_num_keys']}")
    print(f"Direct key overlap: {report['direct_overlap_count']}")
    print(f"Same-shape direct overlap: {report['same_shape_overlap_count']}")
    print(f"Shape mismatches in overlap: {report['shape_mismatch_count']}")

    print_key_examples("HF key examples:", sorted(hf_state), args.show_keys)
    print_key_examples("timm key examples:", sorted(timm_state), args.show_keys)
    print_key_examples("Direct overlap examples:", report["direct_overlap"], args.show_keys)

    if report["same_shape_overlap_count"] == len(timm_state):
        print("\nResult: likely directly loadable into the timm encoder.")
    else:
        print("\nResult: not directly loadable as-is. A key converter is likely needed.")

    if args.report is not None:
        report_path = Path(args.report)
    else:
        report_path = download_root / f"{safe_repo_dir(args.repo_id)}_compare_report.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
