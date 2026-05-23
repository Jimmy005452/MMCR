import argparse
import re
from pathlib import Path

import torch

from mmcr.models import DEFAULT_ARCH, build_image_encoder
from compare_hf_timm_checkpoint import download_repo, find_weight_file, load_hf_state_dict, safe_repo_dir


DEFAULT_REPOS = [
    "tanganke/clip-vit-large-patch14_sun397",
    "tanganke/clip-vit-large-patch14_stanford-cars",
    "tanganke/clip-vit-large-patch14_resisc45",
    "tanganke/clip-vit-large-patch14_eurosat",
    "tanganke/clip-vit-large-patch14_svhn",
    "tanganke/clip-vit-large-patch14_gtsrb",
    "tanganke/clip-vit-large-patch14_mnist",
    "tanganke/clip-vit-large-patch14_dtd",
]

DATASET_NAME_MAP = {
    "stanford-cars": "cars",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+", default=DEFAULT_REPOS)
    parser.add_argument("--download-root", default="checkpoint_another")
    parser.add_argument("--output-root", default="checkpoint_another/converted")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_name_from_repo(repo_id: str):
    suffix = repo_id.rsplit("_", maxsplit=1)[-1]
    return DATASET_NAME_MAP.get(suffix, suffix)


def require_key(state: dict, key: str):
    if key not in state:
        raise KeyError(f"Missing expected HF key: {key}")
    return state[key]


def convert_hf_clip_state_to_timm(hf_state: dict, timm_state: dict):
    converted = {}

    converted["cls_token"] = require_key(hf_state, "vision_model.embeddings.class_embedding").reshape(1, 1, -1)
    converted["pos_embed"] = require_key(hf_state, "vision_model.embeddings.position_embedding.weight").unsqueeze(0)
    converted["patch_embed.proj.weight"] = require_key(hf_state, "vision_model.embeddings.patch_embedding.weight")

    if "patch_embed.proj.bias" in timm_state:
        converted["patch_embed.proj.bias"] = torch.zeros_like(timm_state["patch_embed.proj.bias"])

    if "vision_model.pre_layrnorm.weight" in hf_state:
        converted["norm_pre.weight"] = hf_state["vision_model.pre_layrnorm.weight"]
        converted["norm_pre.bias"] = hf_state["vision_model.pre_layrnorm.bias"]

    if "vision_model.post_layernorm.weight" in hf_state:
        converted["norm.weight"] = hf_state["vision_model.post_layernorm.weight"]
        converted["norm.bias"] = hf_state["vision_model.post_layernorm.bias"]

    layer_pattern = re.compile(r"vision_model\.encoder\.layers\.(\d+)\.")
    layer_ids = sorted({int(match.group(1)) for key in hf_state for match in [layer_pattern.match(key)] if match})

    for layer_id in layer_ids:
        hf_prefix = f"vision_model.encoder.layers.{layer_id}"
        timm_prefix = f"blocks.{layer_id}"

        converted[f"{timm_prefix}.norm1.weight"] = require_key(hf_state, f"{hf_prefix}.layer_norm1.weight")
        converted[f"{timm_prefix}.norm1.bias"] = require_key(hf_state, f"{hf_prefix}.layer_norm1.bias")
        converted[f"{timm_prefix}.norm2.weight"] = require_key(hf_state, f"{hf_prefix}.layer_norm2.weight")
        converted[f"{timm_prefix}.norm2.bias"] = require_key(hf_state, f"{hf_prefix}.layer_norm2.bias")

        q_weight = require_key(hf_state, f"{hf_prefix}.self_attn.q_proj.weight")
        k_weight = require_key(hf_state, f"{hf_prefix}.self_attn.k_proj.weight")
        v_weight = require_key(hf_state, f"{hf_prefix}.self_attn.v_proj.weight")
        converted[f"{timm_prefix}.attn.qkv.weight"] = torch.cat([q_weight, k_weight, v_weight], dim=0)

        q_bias = require_key(hf_state, f"{hf_prefix}.self_attn.q_proj.bias")
        k_bias = require_key(hf_state, f"{hf_prefix}.self_attn.k_proj.bias")
        v_bias = require_key(hf_state, f"{hf_prefix}.self_attn.v_proj.bias")
        converted[f"{timm_prefix}.attn.qkv.bias"] = torch.cat([q_bias, k_bias, v_bias], dim=0)

        converted[f"{timm_prefix}.attn.proj.weight"] = require_key(hf_state, f"{hf_prefix}.self_attn.out_proj.weight")
        converted[f"{timm_prefix}.attn.proj.bias"] = require_key(hf_state, f"{hf_prefix}.self_attn.out_proj.bias")
        converted[f"{timm_prefix}.mlp.fc1.weight"] = require_key(hf_state, f"{hf_prefix}.mlp.fc1.weight")
        converted[f"{timm_prefix}.mlp.fc1.bias"] = require_key(hf_state, f"{hf_prefix}.mlp.fc1.bias")
        converted[f"{timm_prefix}.mlp.fc2.weight"] = require_key(hf_state, f"{hf_prefix}.mlp.fc2.weight")
        converted[f"{timm_prefix}.mlp.fc2.bias"] = require_key(hf_state, f"{hf_prefix}.mlp.fc2.bias")

    missing = sorted(set(timm_state) - set(converted))
    extra = sorted(set(converted) - set(timm_state))
    shape_mismatches = []
    for key in sorted(set(timm_state) & set(converted)):
        if timm_state[key].shape != converted[key].shape:
            shape_mismatches.append((key, tuple(converted[key].shape), tuple(timm_state[key].shape)))

    if shape_mismatches:
        details = "\n".join(f"{key}: HF {hf_shape} vs timm {timm_shape}" for key, hf_shape, timm_shape in shape_mismatches[:20])
        raise ValueError(f"Shape mismatches found:\n{details}")

    if extra:
        raise ValueError(f"Converted state has unexpected timm keys: {extra[:20]}")

    final_state = {}
    for key, value in timm_state.items():
        if key in converted:
            final_state[key] = converted[key].to(dtype=value.dtype)
        else:
            final_state[key] = value

    return final_state, missing


def convert_repo(repo_id: str, args, timm_state: dict):
    download_root = Path(args.download_root)
    repo_dir = download_repo(repo_id, download_root)
    weight_path = find_weight_file(repo_dir)
    hf_state = load_hf_state_dict(weight_path)
    converted_state, missing = convert_hf_clip_state_to_timm(hf_state, timm_state)

    dataset = dataset_name_from_repo(repo_id)
    output_path = Path(args.output_root) / dataset / "encoder.pt"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted_state, output_path)
    print(f"Saved {repo_id} -> {output_path}")
    if missing:
        print(f"  Filled {len(missing)} missing timm keys from the base timm initialization.")


def main():
    args = parse_args()
    timm_model = build_image_encoder(arch=args.arch, pretrained=False)
    timm_state = timm_model.state_dict()

    for repo_id in args.repos:
        print(f"\nConverting {repo_id}")
        convert_repo(repo_id, args, timm_state)


if __name__ == "__main__":
    main()
