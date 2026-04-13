import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# =====================
# CONFIG
# =====================

DATASET_PATH = "/home/ghost/Music/efficientnet"
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

save_dir = "efficientnet_results"
os.makedirs(save_dir, exist_ok=True)

# =====================
# TRANSFORMS
# =====================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(
    root=f"{DATASET_PATH}/train",
    transform=transform
)

val_data = datasets.ImageFolder(
    root=f"{DATASET_PATH}/test",
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# =====================
# MODEL (EfficientNet)
# =====================

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=2   # humo + background
)

model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =====================
# TRAIN LOOP
# =====================

train_losses = []
val_losses = []
val_accs = []

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # ---------- VALIDATION ----------
    model.eval()
    correct = 0
    total = 0
    val_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    val_losses.append(val_loss / len(val_loader))
    val_accs.append(acc)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_losses[-1]:.4f}")
    print(f"Accuracy: {acc:.4f}")

# =====================
# SAVE MODEL
# =====================

torch.save(model.state_dict(), f"{save_dir}/efficientnet_smoke.pth")

print("Modelo guardado ✅")

# =====================
# CONFUSION MATRIX
# =====================

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=train_data.classes,
            yticklabels=train_data.classes)

plt.title("Confusion Matrix")
plt.savefig(f"{save_dir}/confusion_matrix.png")
plt.close()

# =====================
# LOSS GRAPH
# =====================

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig(f"{save_dir}/loss.png")
plt.close()

# =====================
# ACCURACY GRAPH
# =====================

plt.plot(val_accs)
plt.title("Validation Accuracy")
plt.savefig(f"{save_dir}/accuracy.png")
plt.close()

print("Gráficas guardadas ✅")