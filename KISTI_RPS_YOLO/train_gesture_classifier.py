import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

CLASSES = ["rock", "paper", "scissors"]

def get_dataloaders(data_dir: str, img_size: int, batch_size: int, workers: int = 4):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_set = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader

def build_model(num_classes: int):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # 분류기 헤드 교체
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        running_correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, running_correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            running_correct += (preds == labels).sum().item()
            total += imgs.size(0)
        return running_loss / total, running_correct / total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/gestures")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img_size", type=int, default=160)
    ap.add_argument("--out", type=str, default="weights/gesture_classifier.pth")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Path(args.out).parent, exist_ok=True)

    train_loader, val_loader = get_dataloaders(args.data_dir, args.img_size, args.batch_size)
    model = build_model(num_classes=len(CLASSES)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch:03d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': CLASSES,
                'img_size': args.img_size,
            }, args.out)
            print(f" \u2514 Saved best to {args.out} (acc={best_acc:.4f})")

if __name__ == "__main__":
    main()
