import os
import argparse
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn as nn
import torch.optim as optim

from mmcr.data import build_loaders
from mmcr.engine import evaluate, train_one_epoch
from mmcr.models import DEFAULT_ARCH, build_model, build_model_transforms, load_classification_head, load_image_encoder, save_encoder, save_head
from mmcr.utils import seed_everything, write_json, write_metrics


def build_grad_scaler(device, amp_enabled: bool):
    enabled = amp_enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--freeze-head", action="store_true")
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--head-checkpoint", default=None)
    parser.add_argument("--resume-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, eval_transform, data_config = build_model_transforms(args.arch, pretrained=True)
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

    if args.resume_dir is not None:
        resume_dir = Path(args.resume_dir)
        resume_encoder = resume_dir / "finetuned_encoder.pt"
        resume_head = resume_dir / "head.pt"

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
            in_features=model.image_encoder.num_features,
            num_classes=num_classes,
            map_location="cpu",
        ).to(device)

    if args.freeze_encoder:
        for param in model.image_encoder.parameters():
            param.requires_grad_(False)
    if args.freeze_head:
        model.freeze_head()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    scaler = build_grad_scaler(device, args.amp)

    run_dir = Path(args.output_dir) / args.dataset
    best_acc = 0.0
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
            save_encoder(run_dir / "finetuned_encoder.pt", model)
            if not args.freeze_head:
                save_head(run_dir / "head.pt", model)
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
        scheduler.step()

    # save_encoder(run_dir / "last_encoder.pt", model)
    # if not args.freeze_head:
    #     save_head(run_dir / "last_head.pt", model)
    print(f"Done. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
