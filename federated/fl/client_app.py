import random
import logging

import flwr as fl
import torch

from federated.fl.task import (
    create_model,
    load_train_dataset,
    train,
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLIENTS = 3

SERVER_ADDRESS = "127.0.0.1:8080"

SEED = 42

random.seed(SEED)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("FL_CLIENT")


# ============================================================
# CREATE CLIENT PARTITIONS
# ============================================================

def create_client_partitions(dataset):

    indices = list(range(len(dataset)))

    random.Random(SEED).shuffle(indices)

    split_size = len(indices) // NUM_CLIENTS

    partitions = []

    for client_id in range(NUM_CLIENTS):

        start = client_id * split_size

        if client_id == NUM_CLIENTS - 1:
            end = len(indices)
        else:
            end = start + split_size

        partitions.append(
            indices[start:end]
        )

    return partitions


# ============================================================
# CLIENT
# ============================================================

class EmotionClient(fl.client.NumPyClient):

    def __init__(
        self,
        client_id,
        train_dataset,
        indices
    ):

        self.client_id = client_id

        self.train_dataset = train_dataset

        self.indices = indices

        self.model = create_model()

        logger.info(
            f"CLIENT {self.client_id} initialized | "
            f"Samples: {len(self.indices)}"
        )

    # --------------------------------------------------------
    # GET PARAMETERS
    # --------------------------------------------------------

    def get_parameters(self, config):

        return [
            parameter.detach()
            .cpu()
            .numpy()
            for parameter
            in self.model.state_dict().values()
        ]

    # --------------------------------------------------------
    # SET PARAMETERS
    # --------------------------------------------------------

    def set_parameters(self, parameters):

        state_dict = self.model.state_dict()

        new_state_dict = {}

        for (
            key,
            value
        ), parameter in zip(
            state_dict.items(),
            parameters
        ):

            new_state_dict[key] = torch.tensor(
                parameter
            )

        self.model.load_state_dict(
            new_state_dict,
            strict=True
        )

    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        round_number = int(
            config.get(
                "server_round",
                0
            )
        )

        epochs = int(
            config.get(
                "local_epochs",
                2
            )
        )

        logger.info(
            f"CLIENT {self.client_id} | "
            f"Starting local training | "
            f"Round {round_number}"
        )

        loader = (
            __import__(
                "federated.fl.task",
                fromlist=["create_loader"]
            )
            .create_loader(
                self.train_dataset,
                self.indices
            )
        )

        loss, accuracy = train(
            self.model,
            loader,
            epochs
        )

        logger.info(
            f"CLIENT {self.client_id} | "
            f"Round {round_number} | "
            f"Samples: {len(self.indices)} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )

        return (
            self.get_parameters({}),
            len(self.indices),
            {
                "train_loss": float(loss),
                "train_accuracy": float(accuracy),
                "client_id": self.client_id,
            }
        )


# ============================================================
# DATASET
# ============================================================

logger.info(
    "Loading federated training dataset..."
)

train_dataset = load_train_dataset()

logger.info(
    f"Total training samples: {len(train_dataset)}"
)


# ============================================================
# PARTITIONS
# ============================================================

client_partitions = create_client_partitions(
    train_dataset
)

logger.info(
    "Client partitions created."
)

for client_id, indices in enumerate(
    client_partitions,
    start=1
):

    logger.info(
        f"CLIENT {client_id} | "
        f"Samples: {len(indices)}"
    )


# ============================================================
# CLIENT FACTORY
# ============================================================

def client_fn(cid):

    client_id = int(cid)

    indices = client_partitions[
        client_id
    ]

    client = EmotionClient(
        client_id=client_id + 1,
        train_dataset=train_dataset,
        indices=indices
    )

    return client.to_client()


# ============================================================
# START CLIENT
# ============================================================

if __name__ == "__main__":

    logger.info(
        "=========================================="
    )

    logger.info(
        "STARTING FEDERATED CLIENT"
    )

    logger.info(
        "=========================================="
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # cid is supplied by Flower.
    #
    # For this demonstration, we start one client process
    # at a time and assign the client ID through an argument.
    # --------------------------------------------------------

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python -m federated.fl.client_app <client_id>"
        )

        print(
            "Example:"
        )

        print(
            "python -m federated.fl.client_app 0"
        )

        print(
            "python -m federated.fl.client_app 1"
        )

        print(
            "python -m federated.fl.client_app 2"
        )

        raise SystemExit(1)

    cid = int(sys.argv[1])

    if cid < 0 or cid >= NUM_CLIENTS:

        raise ValueError(
            f"Client ID must be 0-{NUM_CLIENTS - 1}"
        )

    logger.info(
        f"Connecting CLIENT {cid + 1} "
        f"to server {SERVER_ADDRESS}"
    )

    client = client_fn(
        str(cid)
    )

    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client
    )