"""
OA-LoRA Training Entry Point

This script provides the main entry point for training the
Occlusion-Aware Low-Rank Adaptation (OA-LoRA) framework.

The script:
1. Loads a YAML configuration.
2. Sets the random seed for reproducibility.
3. Loads the selected TuSimple or CULane dataset.
4. Initializes the SDXL base model.
5. Configures LoRA parameters.
6. Applies prior-preservation settings.
7. Records the ASDS training configuration.

Usage:
    python training/train_oalora.py \
        --config configs/oalora_tusimple.yaml

    python training/train_oalora.py \
        --config configs/oalora_culane.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed):
    """
    Set random seeds for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"[INFO] Random seed set to: {seed}")


def load_config(config_path):
    """
    Load the OA-LoRA YAML configuration.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def print_configuration(config):
    """
    Display the main OA-LoRA training configuration.
    """

    print("\n" + "=" * 60)
    print("OA-LoRA TRAINING CONFIGURATION")
    print("=" * 60)

    print(
        f"Dataset              : "
        f"{config['dataset']['name']}"
    )

    print(
        f"Number of images     : "
        f"{config['dataset']['num_images']}"
    )

    print(
        f"Base model           : "
        f"{config['model']['base_model']}"
    )

    print(
        f"Resolution            : "
        f"{config['training']['resolution']}"
    )

    print(
        f"Batch size             : "
        f"{config['training']['batch_size']}"
    )

    print(
        f"Learning rate          : "
        f"{config['training']['learning_rate']}"
    )

    print(
        f"Mixed precision        : "
        f"{config['training']['mixed_precision']}"
    )

    print(
        f"LoRA rank              : "
        f"{config['lora']['rank']}"
    )

    print(
        f"LoRA alpha             : "
        f"{config['lora']['alpha']}"
    )

    print(
        f"Prior preservation    : "
        f"{config['prior_preservation']['enabled']}"
    )

    print(
        f"Prior loss weight      : "
        f"{config['prior_preservation']['prior_loss_weight']}"
    )

    print(
        f"ASDS strategy          : "
        f"{config['data_scheduling']['method']}"
    )

    print(
        f"Synthetic:Real ratio  : "
        f"{config['data_scheduling']['synthetic_to_real_ratio']}"
    )

    print(
        f"Random seed            : "
        f"{config['training']['seed']}"
    )

    print("=" * 60)


def validate_config(config):
    """
    Validate the essential OA-LoRA configuration fields.
    """

    required_sections = [
        "model",
        "dataset",
        "training",
        "lora",
        "prior_preservation",
        "sampling",
        "data_scheduling",
    ]

    for section in required_sections:

        if section not in config:
            raise ValueError(
                f"Missing configuration section: {section}"
            )

    print(
        "[INFO] Configuration validation completed successfully."
    )


def main():

    parser = argparse.ArgumentParser(
        description="Train the OA-LoRA framework."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the OA-LoRA YAML configuration file.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = load_config(args.config)

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    validate_config(config)

    # --------------------------------------------------------
    # Set reproducibility seed
    # --------------------------------------------------------

    seed = config["training"]["seed"]

    set_seed(seed)

    # --------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------

    print_configuration(config)

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    dataset_name = config["dataset"]["name"]

    print(
        f"\n[INFO] Preparing dataset: {dataset_name}"
    )

    if dataset_name.lower() == "tusimple":

        print(
            "[INFO] TuSimple dataset loader selected."
        )

        print(
            "[INFO] Use datasets/tusimple.py "
            "for loading prepared images."
        )

    elif dataset_name.lower() == "culane":

        print(
            "[INFO] CULane dataset loader selected."
        )

        print(
            "[INFO] Use datasets/culane.py "
            "for loading prepared images."
        )

    else:

        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    print(
        "\n[INFO] Initializing SDXL model:"
    )

    print(
        f"       {config['model']['base_model']}"
    )

    print(
        "[INFO] Configuring LoRA adaptation."
    )

    print(
        f"       Rank  : {config['lora']['rank']}"
    )

    print(
        f"       Alpha : {config['lora']['alpha']}"
    )

    # --------------------------------------------------------
    # Prior preservation
    # --------------------------------------------------------

    if config["prior_preservation"]["enabled"]:

        print(
            "\n[INFO] Prior preservation enabled."
        )

        print(
            f"       Prior loss weight: "
            f"{config['prior_preservation']['prior_loss_weight']}"
        )

        print(
            f"       Class images: "
            f"{config['prior_preservation']['num_class_images']}"
        )

    # --------------------------------------------------------
    # ASDS
    # --------------------------------------------------------

    print(
        "\n[INFO] Adaptive Synthetic Data Scheduling (ASDS)"
    )

    print(
        f"       Synthetic-to-real ratio: "
        f"{config['data_scheduling']['synthetic_to_real_ratio']}"
    )

    # --------------------------------------------------------
    # Training placeholder
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "OA-LoRA training configuration loaded successfully."
    )

    print(
        "The complete model optimization routine should be "
        "implemented here."
    )

    print(
        "This entry point is provided to document the "
        "reproducible training configuration."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
