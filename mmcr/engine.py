import torch
from tqdm import tqdm

from mmcr.utils import accuracy


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, args):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="train", leave=False)
    for step, (images, targets) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets) / args.grad_accum_steps

        scaler.scale(loss).backward()

        if step % args.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_loss = loss.item() * args.grad_accum_steps
        batch_acc = accuracy(logits.detach(), targets)
        total_loss += batch_loss
        total_acc += batch_acc
        progress.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.4f}")

    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp: bool = False):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    progress = tqdm(loader, desc="eval", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets) if criterion is not None else None

        batch_acc = accuracy(logits, targets)
        total_acc += batch_acc
        if loss is not None:
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")
        else:
            progress.set_postfix(acc=f"{batch_acc:.4f}")

    avg_loss = total_loss / len(loader) if criterion is not None else None
    return avg_loss, total_acc / len(loader)
