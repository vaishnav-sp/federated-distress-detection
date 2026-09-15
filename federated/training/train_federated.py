import copy
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from federated.emotion_dataset import FederatedEmotionDataset
from models.emotion_model import EmotionModel


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = (
    "data/raw/Autism emotion recogition dataset"
)

NUM_CLIENTS = 3

NUM_ROUNDS = 5

LOCAL_EPOCHS = 2

BATCH_SIZE = 32

LEARNING_RATE = 0.0001

SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOG DIRECTORY
# ============================================================

timestamp = datetime.now().strftime(
    "%Y-%m-%d_%H%M%S"
)

LOG_DIR = (
    Path("federated")
    / "logs"
    / timestamp
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "FederatedTraining"
)

logger.setLevel(
    logging.INFO
)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

file_handler = logging.FileHandler(
    LOG_DIR / "training.log",
    encoding="utf-8"
)

file_handler.setFormatter(
    formatter
)

console_handler = logging.StreamHandler()

console_handler.setFormatter(
    formatter
)

logger.addHandler(
    file_handler
)

logger.addHandler(
    console_handler
)


# ============================================================
# SAVE JSON
# ============================================================

def save_json(filename, data):

    path = LOG_DIR / filename

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )


# ============================================================
# CLIENT TRAINING
# ============================================================

def train_client(
    global_model,
    dataset,
    indices,
    client_id
):

    model = copy.deepcopy(
        global_model
    )

    model.to(DEVICE)

    model.train()

    subset = Subset(
        dataset,
        indices
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    logger.info(
        f"Client {client_id}: "
        f"starting local training "
        f"with {len(indices)} samples"
    )

    epoch_metrics = []

    for epoch in range(
        LOCAL_EPOCHS
    ):

        total_loss = 0.0

        correct = 0

        total = 0

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

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        epoch_loss = (
            total_loss
            / max(total, 1)
        )

        epoch_accuracy = (
            correct
            / max(total, 1)
        )

        epoch_metrics.append({

            "epoch": epoch + 1,

            "loss": epoch_loss,

            "accuracy": epoch_accuracy

        })

        logger.info(
            f"Client {client_id} | "
            f"Epoch {epoch + 1}/{LOCAL_EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {epoch_accuracy:.4f}"
        )

    return (
        model.state_dict(),
        len(indices),
        epoch_metrics
    )


# ============================================================
# FEDAVG
# ============================================================

def fedavg(
    client_states,
    client_sizes
):

    total_samples = sum(
        client_sizes
    )

    global_state = {}

    for key in client_states[0]:

        global_state[key] = (
            client_states[0][key].clone()
            * (
                client_sizes[0]
                / total_samples
            )
        )

        for i in range(
            1,
            len(client_states)
        ):

            global_state[key] += (
                client_states[i][key].clone()
                * (
                    client_sizes[i]
                    / total_samples
                )
            )

    return global_state


# ============================================================
# GLOBAL EVALUATION
# ============================================================

def evaluate(
    model,
    dataset
):

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0

    correct = 0

    total = 0

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

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return {

        "loss": total_loss / max(total, 1),

        "accuracy": correct / max(total, 1)

    }


# ============================================================
# DATASET SPLIT
# ============================================================

def create_client_splits(
    dataset
):

    indices = list(
        range(len(dataset))
    )

    random.shuffle(
        indices
    )

    client_splits = np.array_split(
        indices,
        NUM_CLIENTS
    )

    client_splits = [
        list(split)
        for split in client_splits
    ]

    return client_splits


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "=========================================="
    )

    logger.info(
        "FEDERATED DISTRESS EMOTION TRAINING"
    )

    logger.info(
        "=========================================="
    )

    logger.info(
        f"Device: {DEVICE}"
    )

    logger.info(
        f"Clients: {NUM_CLIENTS}"
    )

    logger.info(
        f"Rounds: {NUM_ROUNDS}"
    )

    logger.info(
        f"Local epochs: {LOCAL_EPOCHS}"
    )

    logger.info(
        f"Batch size: {BATCH_SIZE}"
    )

    logger.info(
        f"Learning rate: {LEARNING_RATE}"
    )


    # ========================================================
    # DATASET
    # ========================================================

    dataset = FederatedEmotionDataset(
        DATASET_PATH,
        split="train"
    )

    logger.info(
        f"Total training samples: "
        f"{len(dataset)}"
    )

    logger.info(
        f"Binary distribution: "
        f"{dataset.get_class_distribution()}"
    )

    logger.info(
        f"Original distribution: "
        f"{dataset.get_original_distribution()}"
    )


    save_json(
        "dataset_distribution.json",
        {
            "total_samples": len(dataset),

            "binary_distribution":
                dataset.get_class_distribution(),

            "original_distribution":
                dataset.get_original_distribution()
        }
    )


    # ========================================================
    # CLIENT SPLITS
    # ========================================================

    client_splits = create_client_splits(
        dataset
    )

    client_distribution = {}

    for i, indices in enumerate(
        client_splits
    ):

        distribution = {
            "NORMAL": 0,
            "DISTRESS": 0
        }

        for index in indices:

            _, label = dataset[index]

            if label == 0:

                distribution["NORMAL"] += 1

            else:

                distribution["DISTRESS"] += 1

        client_distribution[
            f"client_{i + 1}"
        ] = {

            "samples": len(indices),

            "distribution": distribution
        }

        logger.info(
            f"Client {i + 1}: "
            f"{len(indices)} samples | "
            f"NORMAL={distribution['NORMAL']} | "
            f"DISTRESS={distribution['DISTRESS']}"
        )


    save_json(
        "client_distribution.json",
        client_distribution
    )


    # ========================================================
    # GLOBAL MODEL
    # ========================================================

    global_model = EmotionModel(
        num_classes=2
    )

    global_model.to(
        DEVICE
    )


    round_metrics = []


    # ========================================================
    # FEDERATED ROUNDS
    # ========================================================

    for round_number in range(
        1,
        NUM_ROUNDS + 1
    ):

        logger.info("")
        logger.info(
            "=========================================="
        )

        logger.info(
            f"FEDERATED ROUND {round_number}/{NUM_ROUNDS}"
        )

        logger.info(
            "=========================================="
        )


        client_states = []

        client_sizes = []

        round_client_metrics = []


        # ----------------------------------------------------
        # CLIENT TRAINING
        # ----------------------------------------------------

        for client_id, indices in enumerate(
            client_splits,
            start=1
        ):

            state_dict, size, metrics = train_client(

                global_model,

                dataset,

                indices,

                client_id
            )

            client_states.append(
                state_dict
            )

            client_sizes.append(
                size
            )

            round_client_metrics.append({

                "client_id": client_id,

                "samples": size,

                "metrics": metrics

            })


        # ----------------------------------------------------
        # FEDAVG
        # ----------------------------------------------------

        logger.info(
            "Server: aggregating client models using FedAvg"
        )

        global_state = fedavg(
            client_states,
            client_sizes
        )

        global_model.load_state_dict(
            global_state
        )

        logger.info(
            "Server: global model updated"
        )


        # ----------------------------------------------------
        # GLOBAL EVALUATION
        # ----------------------------------------------------

        metrics = evaluate(
            global_model,
            dataset
        )

        logger.info(
            f"Global Model | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f}"
        )


        round_metrics.append({

            "round": round_number,

            "global_loss":
                metrics["loss"],

            "global_accuracy":
                metrics["accuracy"],

            "clients":
                round_client_metrics

        })


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        Path("models")
        / "federated_emotion_model.pth"
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        global_model.state_dict(),
        model_path
    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    save_json(
        "round_metrics.json",
        round_metrics
    )


    # ========================================================
    # CONFIG
    # ========================================================

    save_json(
        "config.json",
        {

            "dataset":
                DATASET_PATH,

            "num_clients":
                NUM_CLIENTS,

            "num_rounds":
                NUM_ROUNDS,

            "local_epochs":
                LOCAL_EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "seed":
                SEED,

            "device":
                str(DEVICE),

            "model":
                "MobileNetV3",

            "classes": {

                "0": "NORMAL",

                "1": "DISTRESS"

            }

        }
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    final_metrics = round_metrics[-1]

    summary = {

        "status": "completed",

        "model":
            str(model_path),

        "final_round":
            final_metrics["round"],

        "final_global_loss":
            final_metrics["global_loss"],

        "final_global_accuracy":
            final_metrics["global_accuracy"]

    }


    save_json(
        "summary.json",
        summary
    )


    logger.info("")
    logger.info(
        "=========================================="
    )

    logger.info(
        "FEDERATED TRAINING COMPLETED"
    )

    logger.info(
        f"Global model saved to: {model_path}"
    )

    logger.info(
        f"Logs saved to: {LOG_DIR}"
    )

    logger.info(
        "=========================================="
    )


if __name__ == "__main__":

    main()