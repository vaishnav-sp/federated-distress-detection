import torch
from pathlib import Path

from models.emotion_model import EmotionModel
from federated.fl.task import (
    load_test_dataset,
    create_loader,
    DEVICE
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "logs/federated/2026-09-15_161349/global_emotion_model.pth"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING FINAL FEDERATED GLOBAL MODEL")
print("=" * 60)

model = EmotionModel(
    num_classes=2
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    state_dict
)

model.to(DEVICE)
model.eval()

print(f"Model loaded from: {MODEL_PATH}")
print(f"Device: {DEVICE}")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_dataset = load_test_dataset()

test_indices = list(
    range(len(test_dataset))
)

test_loader = create_loader(
    test_dataset,
    test_indices,
    shuffle=False
)

print(
    f"Test samples: {len(test_dataset)}"
)


# ============================================================
# EVALUATION
# ============================================================

criterion = torch.nn.CrossEntropyLoss()

total_loss = 0.0
total_correct = 0
total_samples = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item() *
            batch_size
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size


# ============================================================
# RESULTS
# ============================================================

test_loss = (
    total_loss /
    max(total_samples, 1)
)

test_accuracy = (
    total_correct /
    max(total_samples, 1)
)


print("\n" + "=" * 60)
print("GLOBAL MODEL VERIFICATION")
print("=" * 60)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Samples  : {total_samples}"
)

print("=" * 60)