import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.emotion_model import EmotionModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_PATH = "data/raw/Autism emotion recogition dataset"

TRAIN_PATH = f"{DATASET_PATH}/train"
TEST_PATH = f"{DATASET_PATH}/test"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Transforms
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Dataset
# --------------------------------------------------

train_dataset = datasets.ImageFolder(
    TRAIN_PATH,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    TEST_PATH,
    transform=test_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("=" * 60)
print("EMOTION MODEL TRAINING")
print("=" * 60)

print("Device:", DEVICE)
print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))


# --------------------------------------------------
# Model
# --------------------------------------------------

model = EmotionModel(
    num_classes=len(train_dataset.classes)
).to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


    train_accuracy = 100 * correct / total

    average_loss = (
        running_loss / len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "emotion_model.pth"
)

print("\nTraining complete.")
print("Model saved as: emotion_model.pth")