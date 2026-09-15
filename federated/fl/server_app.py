import json
import logging
from datetime import datetime
from pathlib import Path

import flwr as fl
import torch

from models.emotion_model import EmotionModel
from federated.fl.task import (
    DEVICE,
    load_test_dataset,
    create_loader,
)


# ============================================================
# CONFIG
# ============================================================

NUM_CLIENTS = 3
NUM_ROUNDS = 5
LOCAL_EPOCHS = 2


# ============================================================
# LOG DIRECTORY
# ============================================================

timestamp = datetime.now().strftime(
    "%Y-%m-%d_%H%M%S"
)

LOG_DIR = (
    Path("logs")
    / "federated"
    / timestamp
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
    handlers=[
        logging.FileHandler(
            LOG_DIR / "training.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(
    "FL_SERVER"
)


# ============================================================
# METRICS
# ============================================================

round_metrics = []


# ============================================================
# AGGREGATE CLIENT METRICS
# ============================================================

def aggregate_fit_metrics(metrics):

    total_examples = sum(
        num_examples
        for num_examples, _
        in metrics
    )

    if total_examples == 0:
        return {}

    weighted_loss = sum(
        num_examples * values["train_loss"]
        for num_examples, values
        in metrics
    ) / total_examples

    weighted_accuracy = sum(
        num_examples * values["train_accuracy"]
        for num_examples, values
        in metrics
    ) / total_examples

    return {
        "train_loss": float(weighted_loss),
        "train_accuracy": float(weighted_accuracy)
    }


# ============================================================
# SERVER STRATEGY
# ============================================================

class LoggingFedAvg(fl.server.strategy.FedAvg):

    final_parameters = None

    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):

        logger.info(
            "=========================================="
        )

        logger.info(
            f"SERVER | FEDERATED ROUND "
            f"{server_round}"
        )

        logger.info(
            f"SERVER | Clients participating: "
            f"{len(results)}"
        )

        for client_proxy, fit_res in results:

            logger.info(
                f"SERVER | Client "
                f"{client_proxy.cid} | "
                f"Samples: "
                f"{fit_res.num_examples}"
            )

        aggregated_parameters, metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures
            )
        )

        # ----------------------------------------------------
        # Store latest global parameters
        # ----------------------------------------------------

        if aggregated_parameters is not None:

            self.final_parameters = (
                aggregated_parameters
            )

        # ----------------------------------------------------
        # Log aggregated metrics
        # ----------------------------------------------------

        if metrics:

            logger.info(
                f"SERVER | Aggregated Loss: "
                f"{metrics.get('train_loss', 0):.4f}"
            )

            logger.info(
                f"SERVER | Aggregated Accuracy: "
                f"{metrics.get('train_accuracy', 0):.4f}"
            )

        # ----------------------------------------------------
        # Store round information
        # ----------------------------------------------------

        round_metrics.append({
            "round": server_round,
            "clients": len(results),
            "metrics": metrics
        })

        return (
            aggregated_parameters,
            metrics
        )


# ============================================================
# INITIAL GLOBAL MODEL PARAMETERS
# ============================================================

def get_initial_parameters():

    model = EmotionModel(
        num_classes=2
    )

    parameters = [
        value.detach()
        .cpu()
        .numpy()
        for value in model.state_dict().values()
    ]

    return fl.common.ndarrays_to_parameters(
        parameters
    )


# ============================================================
# EVALUATE FINAL GLOBAL MODEL
# ============================================================

def evaluate_global_model(model):

    logger.info(
        "Loading test dataset..."
    )

    test_dataset = load_test_dataset()

    test_indices = list(
        range(
            len(test_dataset)
        )
    )

    test_loader = create_loader(
        test_dataset,
        test_indices,
        shuffle=False
    )

    logger.info(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    model.to(DEVICE)
    model.eval()

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

            total_samples += batch_size

    loss = (
        total_loss
        / max(total_samples, 1)
    )

    accuracy = (
        total_correct
        / max(total_samples, 1)
    )

    return loss, accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "=========================================="
    )

    logger.info(
        "FEDERATED DISTRESS EMOTION SERVER"
    )

    logger.info(
        "=========================================="
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
        f"Device: {DEVICE}"
    )

    logger.info(
        f"Log directory: {LOG_DIR}"
    )

    # ========================================================
    # FEDAVG STRATEGY
    # ========================================================

    strategy = LoggingFedAvg(

        fraction_fit=1.0,

        fraction_evaluate=0.0,

        min_fit_clients=NUM_CLIENTS,

        min_available_clients=NUM_CLIENTS,

        initial_parameters=(
            get_initial_parameters()
        ),

        fit_metrics_aggregation_fn=(
            aggregate_fit_metrics
        ),

        on_fit_config_fn=(
            lambda round_number: {
                "server_round": round_number,
                "local_epochs": LOCAL_EPOCHS
            }
        )
    )

    # ========================================================
    # START FEDERATED SERVER
    # ========================================================

    fl.server.start_server(

        server_address="127.0.0.1:8080",

        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),

        strategy=strategy
    )

    # ========================================================
    # FEDERATED TRAINING COMPLETED
    # ========================================================

    logger.info(
        "=========================================="
    )

    logger.info(
        "FEDERATED TRAINING COMPLETED"
    )

    logger.info(
        "=========================================="
    )

    # ========================================================
    # RECONSTRUCT FINAL GLOBAL MODEL
    # ========================================================

    model = None

    if strategy.final_parameters is not None:

        model = EmotionModel(
            num_classes=2
        )

        ndarrays = (
            fl.common.parameters_to_ndarrays(
                strategy.final_parameters
            )
        )

        state_dict = model.state_dict()

        new_state_dict = {}

        for (
            key,
            _
        ), value in zip(
            state_dict.items(),
            ndarrays
        ):

            new_state_dict[key] = torch.tensor(
                value
            )

        model.load_state_dict(
            new_state_dict,
            strict=True
        )

        model.to(DEVICE)

        # ====================================================
        # SAVE FINAL GLOBAL MODEL
        # ====================================================

        model_path = (
            LOG_DIR
            / "global_emotion_model.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        logger.info(
            f"Final global model saved to: "
            f"{model_path}"
        )

        # ====================================================
        # FINAL TEST EVALUATION
        # ====================================================

        test_loss, test_accuracy = (
            evaluate_global_model(
                model
            )
        )

        logger.info(
            "=========================================="
        )

        logger.info(
            "FINAL GLOBAL MODEL EVALUATION"
        )

        logger.info(
            "=========================================="
        )

        logger.info(
            f"Test Loss: "
            f"{test_loss:.4f}"
        )

        logger.info(
            f"Test Accuracy: "
            f"{test_accuracy:.4f}"
        )

        logger.info(
            f"Test Accuracy (%): "
            f"{test_accuracy * 100:.2f}%"
        )

        logger.info(
            "=========================================="
        )

    # ========================================================
    # SAVE ROUND METRICS
    # ========================================================

    metrics_path = (
        LOG_DIR
        / "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            round_metrics,
            file,
            indent=4
        )

    logger.info(
        f"Round metrics saved to: "
        f"{metrics_path}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    logger.info(
        "=========================================="
    )

    logger.info(
        "FEDERATED EXPERIMENT SUMMARY"
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

    if model is not None:

        logger.info(
            f"Final Test Accuracy: "
            f"{test_accuracy * 100:.2f}%"
        )

    logger.info(
        f"Logs: {LOG_DIR}"
    )

    logger.info(
        "=========================================="
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()