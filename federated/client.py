import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import flwr as fl

from models.distress_model import DistressModel


FEATURE_FILE = "data/processed/distress_features.csv"


FEATURE_COLUMNS = [
    "autism_anger",
    "autism_fear",
    "autism_joy",
    "autism_Natural",
    "autism_sadness",
    "autism_surprise",

    "general_angry",
    "general_disgust",
    "general_fear",
    "general_happy",
    "general_neutral",
    "general_sad",
    "general_surprise",

    "movement",
    "head_movement",
    "eye_openness",
    "mouth_openness",

    "movement_mean",
    "movement_std",
    "movement_peak",
    "movement_activity",

    "head_mean",
    "head_std",
    "head_peak"
]


class DistressClient(fl.client.NumPyClient):

    def __init__(
        self,
        train_x,
        train_y,
        test_x,
        test_y,
        client_id
    ):

        self.client_id = client_id

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = DistressModel(
            input_size=24,
            num_classes=2
        ).to(self.device)

        # -----------------------------------------
        # Training data
        # -----------------------------------------

        self.train_x = torch.tensor(
            train_x,
            dtype=torch.float32
        ).to(self.device)

        self.train_y = torch.tensor(
            train_y,
            dtype=torch.long
        ).to(self.device)

        # -----------------------------------------
        # Test data
        # -----------------------------------------

        self.test_x = torch.tensor(
            test_x,
            dtype=torch.float32
        ).to(self.device)

        self.test_y = torch.tensor(
            test_y,
            dtype=torch.long
        ).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()

    # =========================================
    # Get model parameters
    # =========================================

    def get_parameters(self, config):

        return [
            value.detach().cpu().numpy()
            for value in self.model.state_dict().values()
        ]

    # =========================================
    # Set model parameters
    # =========================================

    def set_parameters(self, parameters):

        state_dict = self.model.state_dict()

        new_state_dict = {}

        for (key, old_value), new_value in zip(
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

    # =========================================
    # Local training
    # =========================================

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        epochs = 3

        for _ in range(epochs):

            optimizer.zero_grad()

            output = self.model(
                self.train_x
            )

            loss = self.loss_fn(
                output,
                self.train_y
            )

            loss.backward()

            optimizer.step()

        print(
            f"Client {self.client_id} | "
            f"Loss: {loss.item():.4f}"
        )

        return (
            self.get_parameters(config),
            len(self.train_x),
            {
                "loss": float(loss.item())
            }
        )

    # =========================================
    # Evaluation on held-out test data
    # =========================================

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        self.model.eval()

        with torch.no_grad():

            output = self.model(
                self.test_x
            )

            loss = self.loss_fn(
                output,
                self.test_y
            )

            predictions = torch.argmax(
                output,
                dim=1
            )

            accuracy = (
                predictions == self.test_y
            ).float().mean().item()

        print(
            f"Client {self.client_id} | "
            f"Test Loss: {loss.item():.4f} | "
            f"Test Accuracy: {accuracy:.4f}"
        )

        return (
            float(loss.item()),
            len(self.test_x),
            {
                "accuracy": float(accuracy)
            }
        )


# =========================================
# Flower client creation
# =========================================

def client_fn(cid):

    df = pd.read_csv(
        FEATURE_FILE
    )

    # -----------------------------------------
    # Separate training and test data
    # -----------------------------------------

    train_df = df[
        df["split"] == "train"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    # -----------------------------------------
    # Training data
    # -----------------------------------------

    X_train = train_df[
        FEATURE_COLUMNS
    ].values.astype(
        np.float32
    )

    y_train = train_df[
        "distress_label"
    ].values.astype(
        np.int64
    )

    # -----------------------------------------
    # Test data
    # -----------------------------------------

    X_test = test_df[
        FEATURE_COLUMNS
    ].values.astype(
        np.float32
    )

    y_test = test_df[
        "distress_label"
    ].values.astype(
        np.int64
    )

    # -----------------------------------------
    # Divide training data among 3 clients
    # -----------------------------------------

    num_clients = 3

    client_id = int(cid) - 1

    client_indices = np.array_split(
        np.arange(len(X_train)),
        num_clients
    )

    indices = client_indices[client_id]

    train_x = X_train[indices]
    train_y = y_train[indices]

    print(
        f"Client {client_id + 1}: "
        f"{len(train_x)} training samples | "
        f"{len(X_test)} test samples"
    )

    # -----------------------------------------
    # Create client
    # -----------------------------------------

    return DistressClient(
        train_x=train_x,
        train_y=train_y,
        test_x=X_test,
        test_y=y_test,
        client_id=client_id + 1
    ).to_client()