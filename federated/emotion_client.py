import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import flwr as fl

from federated.emotion_dataset import FederatedEmotionDataset
from models.emotion_model import EmotionModel


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = (
    "data/raw/Autism emotion recogition dataset"
)

NUM_CLIENTS = 3

BATCH_SIZE = 16

LOCAL_EPOCHS = 2

LEARNING_RATE = 0.0001


# ============================================================
# BINARY EMOTION MODEL
# ============================================================
#
# We use the existing EmotionModel architecture.
#
# Original:
#   6 emotion classes
#
# FL training:
#   2 classes
#
#   0 = NORMAL
#   1 = DISTRESS
#
# The architecture itself is reused, but the final
# classification layer is changed to 2 outputs.
# ============================================================


def create_model():

    model = EmotionModel(
        num_classes=2
    )

    return model


# ============================================================
# CLIENT DATA SPLIT
# ============================================================


def create_client_indices(
    dataset,
    client_id,
    num_clients
):

    # --------------------------------------------------------
    # Separate samples by binary label
    # --------------------------------------------------------

    normal_indices = []

    distress_indices = []

    for index, (_, label) in enumerate(dataset):

        label = int(label.item())

        if label == 0:

            normal_indices.append(index)

        else:

            distress_indices.append(index)

    # --------------------------------------------------------
    # Reproducible shuffle
    # --------------------------------------------------------

    rng = np.random.default_rng(42)

    rng.shuffle(normal_indices)
    rng.shuffle(distress_indices)

    # --------------------------------------------------------
    # Split each class independently
    #
    # This ensures every client gets both:
    #
    # NORMAL
    # DISTRESS
    # --------------------------------------------------------

    normal_split = np.array_split(
        normal_indices,
        num_clients
    )

    distress_split = np.array_split(
        distress_indices,
        num_clients
    )

    client_indices = list(
        normal_split[client_id]
    ) + list(
        distress_split[client_id]
    )

    # Shuffle again so classes are mixed
    rng.shuffle(client_indices)

    return client_indices


# ============================================================
# FLOWER CLIENT
# ============================================================


class EmotionClient(
    fl.client.NumPyClient
):

    def __init__(
        self,
        client_id,
        train_dataset,
        indices
    ):

        self.client_id = client_id

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model = create_model()

        self.model.to(
            self.device
        )

        # ----------------------------------------------------
        # Local dataset
        # ----------------------------------------------------

        self.dataset = Subset(
            train_dataset,
            indices
        )

        self.train_loader = DataLoader(
            self.dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        self.loss_fn = nn.CrossEntropyLoss()

    # ========================================================
    # GET PARAMETERS
    # ========================================================

    def get_parameters(
        self,
        config
    ):

        return [
            value.detach()
            .cpu()
            .numpy()
            for value in self.model.state_dict().values()
        ]

    # ========================================================
    # SET PARAMETERS
    # ========================================================

    def set_parameters(
        self,
        parameters
    ):

        state_dict = self.model.state_dict()

        new_state_dict = {}

        for (
            (key, old_value),
            new_value
        ) in zip(
            state_dict.items(),
            parameters
        ):

            new_state_dict[key] = torch.tensor(
                new_value,
                dtype=old_value.dtype
            )

        self.model.load_state_dict(
            new_state_dict
        )

    # ========================================================
    # LOCAL TRAINING
    # ========================================================

    def fit(
        self,
        parameters,
        config
    ):

        # ----------------------------------------------------
        # Receive global model
        # ----------------------------------------------------

        self.set_parameters(
            parameters
        )

        self.model.train()

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE
        )

        # ----------------------------------------------------
        # Local training
        # ----------------------------------------------------

        total_loss = 0.0

        total_samples = 0

        for epoch in range(
            LOCAL_EPOCHS
        ):

            epoch_loss = 0.0

            epoch_samples = 0

            for images, labels in self.train_loader:

                images = images.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                optimizer.zero_grad()

                outputs = self.model(
                    images
                )

                loss = self.loss_fn(
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

                epoch_samples += batch_size

            average_epoch_loss = (
                epoch_loss
                / max(epoch_samples, 1)
            )

            print(
                f"Client {self.client_id} | "
                f"Epoch {epoch + 1}/{LOCAL_EPOCHS} | "
                f"Loss: {average_epoch_loss:.4f}"
            )

            total_loss = epoch_loss
            total_samples = epoch_samples

        average_loss = (
            total_loss
            / max(total_samples, 1)
        )

        print(
            f"Client {self.client_id} | "
            f"Local samples: {len(self.dataset)} | "
            f"Final loss: {average_loss:.4f}"
        )

        # ----------------------------------------------------
        # Send updated model to server
        # ----------------------------------------------------

        return (
            self.get_parameters(config),
            len(self.dataset),
            {
                "loss": float(
                    average_loss
                )
            }
        )

    # ========================================================
    # EVALUATION
    # ========================================================

    def evaluate(
        self,
        parameters,
        config
    ):

        self.set_parameters(
            parameters
        )

        self.model.eval()

        total_loss = 0.0

        correct = 0

        total = 0

        with torch.no_grad():

            for images, labels in self.train_loader:

                images = images.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                outputs = self.model(
                    images
                )

                loss = self.loss_fn(
                    outputs,
                    labels
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                batch_size = labels.size(0)

                total_loss += (
                    loss.item()
                    * batch_size
                )

                correct += (
                    predictions == labels
                ).sum().item()

                total += batch_size

        average_loss = (
            total_loss
            / max(total, 1)
        )

        accuracy = (
            correct
            / max(total, 1)
        )

        print(
            f"Client {self.client_id} | "
            f"Evaluation loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )

        return (
            float(average_loss),
            total,
            {
                "accuracy": float(
                    accuracy
                )
            }
        )


# ============================================================
# FLOWER CLIENT FUNCTION
# ============================================================


def client_fn(cid):

    client_id = int(cid)

    print()
    print(
        "================================"
    )
    print(
        f"STARTING CLIENT {client_id + 1}"
    )
    print(
        "================================"
    )

    # --------------------------------------------------------
    # Load training dataset
    # --------------------------------------------------------

    dataset = FederatedEmotionDataset(
        DATASET_PATH,
        split="train"
    )

    # --------------------------------------------------------
    # Create this client's private subset
    # --------------------------------------------------------

    indices = create_client_indices(
        dataset,
        client_id,
        NUM_CLIENTS
    )

    # --------------------------------------------------------
    # Count labels
    # --------------------------------------------------------

    normal_count = 0

    distress_count = 0

    for index in indices:

        _, label = dataset[index]

        if int(label.item()) == 0:

            normal_count += 1

        else:

            distress_count += 1

    print(
        f"Client {client_id + 1} dataset:"
    )

    print(
        f"  NORMAL   : {normal_count}"
    )

    print(
        f"  DISTRESS : {distress_count}"
    )

    print(
        f"  TOTAL    : {len(indices)}"
    )

    # --------------------------------------------------------
    # Create client
    # --------------------------------------------------------

    client = EmotionClient(
        client_id=client_id + 1,
        train_dataset=dataset,
        indices=indices
    )

    return client.to_client()