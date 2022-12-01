import random
import wandb
import numpy as np
# Launch 5 simulated experiments
total_runs = 1
for run in range(total_runs):
    # 🐝 1️⃣ Start a new run to track this script
    wandb.init(
        # Set the project where this run will be logged
        project="wandbexample1",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{run}",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        })

    # This simple block simulates a training loop logging metrics
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        # 🐝 2️⃣ Log metrics from your script to W&B
        wandb.log({"acc": acc, "loss": loss})
        # Log points and boxes in W&B
    wandb.log(
        {
            "point_scene": wandb.Object3D(
                {
                    "type": "lidar/beta",
                    "points": np.array(
                        [
                            [0.4, 1, 1.3],
                            [1, 1, 1],
                            [1.2, 1, 1.2]
                        ]
                    ),
                    "boxes": np.array(
                        [
                            {
                                "corners": [
                                    [0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1],
                                    [1, 0, 0],
                                    [1, 1, 0],
                                    [0, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]
                                ],
                                "label": "Box",
                                "color": [123, 321, 111],
                            },
                            {
                                "corners": [
                                    [0, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 2],
                                    [2, 0, 0],
                                    [2, 2, 0],
                                    [0, 2, 2],
                                    [2, 0, 2],
                                    [2, 2, 2]
                                ],
                                "label": "Box-2",
                                "color": [111, 321, 0],
                            }
                        ]
                    )

                }
            )
        }
    )
        # Mark the run as finished
    wandb.finish()
