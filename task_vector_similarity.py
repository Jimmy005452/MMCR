import argparse
import csv
import json
from pathlib import Path

from mmcr.checkpoints import ENCODER_FILE
from mmcr.task_vector_similarity import task_vector_similarity


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--zeroshot", default=None)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/task_vector_similarity")
    parser.add_argument("--title", default="Task Vector Cosine Similarity")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, labels: list[str], matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", *labels])
        for label, row in zip(labels, matrix.tolist()):
            writer.writerow([label, *[f"{value:.6f}" for value in row]])


def write_json(path: Path, labels: list[str], matrix, norms: list[float], keys: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "datasets": labels,
        "cosine_similarity": matrix.tolist(),
        "task_vector_norms": dict(zip(labels, norms)),
        "num_keys": len(keys),
        "keys": keys,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_heatmap(path: Path, labels: list[str], matrix, title: str):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    values = matrix.numpy()
    size = max(6, 0.7 * len(labels) + 2)

    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(values, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)

    for row in range(len(labels)):
        for col in range(len(labels)):
            value = values[row, col]
            text_color = "white" if abs(value) > 0.55 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    zeroshot_path = Path(args.zeroshot) if args.zeroshot is not None else checkpoint_root / "zeroshot.pt"
    encoder_paths = [checkpoint_root / dataset / ENCODER_FILE for dataset in args.datasets]

    for path in [zeroshot_path, *encoder_paths]:
        if not path.exists():
            raise FileNotFoundError(path)

    matrix, norms, keys = task_vector_similarity(zeroshot_path, encoder_paths)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "cosine_similarity.csv", args.datasets, matrix)
    write_json(output_dir / "cosine_similarity.json", args.datasets, matrix, norms, keys)
    if not args.no_plot:
        write_heatmap(output_dir / "cosine_similarity.png", args.datasets, matrix, args.title)

    print("Task vector cosine similarity:")
    header = "dataset".ljust(14) + " ".join(label[:10].rjust(10) for label in args.datasets)
    print(header)
    for label, row in zip(args.datasets, matrix.tolist()):
        values = " ".join(f"{value:10.3f}" for value in row)
        print(f"{label.ljust(14)}{values}")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
