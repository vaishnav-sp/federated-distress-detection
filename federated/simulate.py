import sys
import flwr as fl

from federated.client import client_fn


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python -m federated.simulate <client_id>"
        )

        sys.exit(1)

    client_id = sys.argv[1]

    print(
        f"Starting Federated Client {client_id}"
    )

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client_fn(client_id)
    )