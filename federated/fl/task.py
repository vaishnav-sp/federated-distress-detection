import torch

from torch.utils.data import DataLoader, Subset

from federated.emotion_dataset import FederatedEmotionDataset
from models.emotion_model import EmotionModel


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = (
    "data/raw/Autism emotion recogition dataset"
)

BATCH_SIZE = 32

LEARNING_RATE = 0.0001

LOCAL_EPOCHS = 2


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL
# ============================================================

def create_model():

    model = EmotionModel(
        num_classes=2
    )

    model.to(DEVICE)

    return model


# ============================================================
# DATASET
# ============================================================

def load_train_dataset():

    return FederatedEmotionDataset(
        DATASET_PATH,
        split="train"
    )


def load_test_dataset():

    return FederatedEmotionDataset(
        DATASET_PATH,
        split="test"
    )


# ============================================================
# DATA LOADER
# ============================================================

def create_loader(
    dataset,
    indices,
    shuffle=True
):

    subset = Subset(
        dataset,
        indices
    )

    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )


# ============================================================
# TRAIN
# ============================================================

def train(
    model,
    loader,
    epochs=LOCAL_EPOCHS
):

    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    for epoch in range(epochs):

        epoch_loss = 0.0

        epoch_correct = 0

        epoch_samples = 0

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            batch_size = labels.size(0)

            epoch_loss += (
                loss.item()
                * batch_size
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            epoch_correct += (
                predictions == labels
            ).sum().item()

            epoch_samples += batch_size

        total_loss = (
            epoch_loss
        )

        total_correct = (
            epoch_correct
        )

        total_samples = (
            epoch_samples
        )

    loss = (
        total_loss /
        max(total_samples, 1)
    )

    accuracy = (
        total_correct /
        max(total_samples, 1)
    )

    return loss, accuracy


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += (
                batch_size
            )

    loss = (
        total_loss /
        max(total_samples, 1)
    )

    accuracy = (
        total_correct /
        max(total_samples, 1)
    )

    return loss, accuracy
