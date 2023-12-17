import os
import math
import json
import matplotlib.pyplot as plt
from typing import List, Optional
from transformers.trainer import TRAINER_STATE_NAME

from llmtuner.extras.logging import get_logger


logger = get_logger(__name__)


def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5) # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss(save_dictionary: os.PathLike, keys: Optional[List[str]] = ["loss"]) -> None:

    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if not metrics:
            logger.warning(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), label="smoothed")
        plt.title(f"training {key} of {save_dictionary}")
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        plt.savefig(
            os.path.join(save_dictionary, f"training_{key}.png"),
            format="png",
            dpi=100,
        )
        print("Figure saved:", os.path.join(save_dictionary, f"training_{key}.png"))
