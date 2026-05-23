import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from mmcr.data import build_loaders
from mmcr.engine import evaluate, train_one_epoch
from mmcr.checkpoints import (
    ENCODER_FILE,
    HEAD_FILE,
    load_classification_head,
    load_image_encoder,
    save_encoder,
    save_head,
)
from mmcr.models import DEFAULT_ARCH, build_model, build_model_transforms
from mmcr.utils import build_device, seed_everything, write_json, write_metrics


def build_grad_scaler(device, amp_enabled: bool):
    enabled = amp_enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_scheduler(optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    data_args = parser.add_argument_group("data")
    data_args.add_argument("--dataset", required=True)
    data_args.add_argument("--data-root", default="data")
    data_args.add_argument("--output-dir", default="checkpoints")

    model_args = parser.add_argument_group("model")
    model_args.add_argument("--arch", default=DEFAULT_ARCH)
    model_args.add_argument("--encoder-checkpoint", default=None)
    model_args.add_argument("--head-checkpoint", default=None)
    model_args.add_argument("--resume-dir", default=None)
    model_args.add_argument("--freeze-encoder", action="store_true")
    model_args.add_argument("--freeze-head", action="store_true")
    model_args.add_argument("--save-zeroshot", action="store_true")

    train_args = parser.add_argument_group("training")
    train_args.add_argument("--epochs", type=int, default=10)
    train_args.add_argument("--batch-size", type=int, default=16)
    train_args.add_argument("--lr", type=float, default=1e-5)
    train_args.add_argument("--weight-decay", type=float, default=0.05)
    train_args.add_argument("--grad-accum-steps", type=int, default=1)
    train_args.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    train_args.add_argument("--min-lr", type=float, default=1e-7)

    runtime_args = parser.add_argument_group("runtime")
    runtime_args.add_argument("--gpu", type=int, default=0, help="CUDA GPU index. Use -1 for CPU.")
    runtime_args.add_argument("--num-workers", type=int, default=4)
    runtime_args.add_argument("--seed", type=int, default=42)
    runtime_args.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = build_device(args.gpu)
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print("Using device: cpu")

    train_transform, eval_transform, data_config = build_model_transforms(args.arch, pretrained=True)
    output_dir = Path(args.output_dir)
    run_dir = output_dir / args.dataset

    train_loader, val_loader, num_classes = build_loaders(
        args.dataset,
        Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    model = build_model(num_classes, arch=args.arch).to(device)
    zeroshot_path = output_dir / "zeroshot.pt"
    if args.save_zeroshot and not zeroshot_path.exists():
        save_encoder(zeroshot_path, model)

    if args.resume_dir is not None:
        resume_dir = Path(args.resume_dir)
        resume_encoder = resume_dir / ENCODER_FILE
        resume_head = resume_dir / HEAD_FILE

        if args.encoder_checkpoint is None and resume_encoder.exists():
            args.encoder_checkpoint = str(resume_encoder)
        if args.head_checkpoint is None and resume_head.exists():
            args.head_checkpoint = str(resume_head)

    if args.encoder_checkpoint is not None:
        model.image_encoder = load_image_encoder(
            args.encoder_checkpoint,
            fallback_encoder=model.image_encoder,
            map_location="cpu",
        ).to(device)

    if args.head_checkpoint is not None:
        model.classification_head = load_classification_head(
            args.head_checkpoint,
            fallback_head=model.classification_head,
            map_location="cpu",
        ).to(device)

    if args.freeze_encoder:
        for param in model.image_encoder.parameters():
            param.requires_grad_(False)
    if args.freeze_head:
        model.freeze_head()

    criterion = nn.CrossEntropyLoss()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters. Do not freeze both encoder and head.")

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args)
    scaler = build_grad_scaler(device, args.amp)

    best_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, args)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, amp=args.amp)

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        write_metrics(run_dir / "metrics.json", history)

        print(
            f"[{args.dataset}] epoch={epoch} "
            f"lr={current_lr:.6g} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            if args.freeze_encoder:
                print("Encoder is frozen; skipping encoder save.")
            else:
                save_encoder(run_dir / ENCODER_FILE, model)
            if not args.freeze_head:
                save_head(run_dir / HEAD_FILE, model)
            print(f"Saved checkpoint to {run_dir} in best acc {best_acc:.4f}.")
            write_json(
                run_dir / "metadata.json",
                {
                    "dataset": args.dataset,
                    "arch": args.arch,
                    "data_config": data_config,
                    "num_classes": num_classes,
                    "best_acc": best_acc,
                    "best_epoch": epoch,
                    "resume_dir": args.resume_dir,
                    "encoder_checkpoint": args.encoder_checkpoint,
                    "head_checkpoint": args.head_checkpoint,
                    "freeze_head": args.freeze_head,
                    "freeze_encoder": args.freeze_encoder,
                    "args": vars(args),
                },
            )
        if scheduler is not None:
            scheduler.step()

    print(f"Done. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
