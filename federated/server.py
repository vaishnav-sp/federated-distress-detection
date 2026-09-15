import flwr as fl
import torch

from models.distress_model import DistressModel


MODEL_OUTPUT = "models/global_distress_model.pth"


class SaveModelFedAvg(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures
        )

        if aggregated_parameters is not None:

            print(
                f"\nFedAvg aggregation completed for round "
                f"{server_round}"
            )

            if server_round == 3:

                model = DistressModel(
                    input_size=24,
                    num_classes=2
                )

                parameters = fl.common.parameters_to_ndarrays(
                    aggregated_parameters
                )

                state_dict = model.state_dict()

                new_state_dict = {}

                for (key, old_value), new_value in zip(
                    state_dict.items(),
                    parameters
                ):
                    new_state_dict[key] = torch.tensor(
                        new_value,
                        dtype=old_value.dtype
                    )

                model.load_state_dict(
                    new_state_dict
                )

                torch.save(
                    model.state_dict(),
                    MODEL_OUTPUT
                )

                print()
                print("==============================")
                print("GLOBAL MODEL SAVED")
                print("==============================")
                print(MODEL_OUTPUT)

        return aggregated_parameters, aggregated_metrics


strategy = SaveModelFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3
)


print("==============================")
print("STARTING FEDERATED SERVER")
print("==============================")

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(
        num_rounds=3
    ),
    strategy=strategy
)