"""
OA-LoRA: Occlusion-Aware Low-Rank Adaptation

This module implements the core OA-LoRA components used for
diffusion-based lane image generation.

Concepts:
    1. Low-Rank Adaptation (LoRA)
    2. Occlusion-aware conditioning
    3. Layer-wise multi-concept adaptation
    4. Separate adaptation for lane geometry and
       environmental appearance

Default layer assignment used in the OA-LoRA experiments:

    Lane geometry:
        Layers 4 and 6

    Environmental condition / style:
        Layer 7

These assignments can be modified through the configuration.
"""

import torch
import torch.nn as nn


# ============================================================
# 1. LoRA Linear Layer
# ============================================================

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation layer.

    The original linear transformation is kept frozen while
    trainable low-rank matrices A and B learn the task-specific
    adaptation.

    W' = W + (alpha / rank) * B @ A
    """

    def __init__(
        self,
        in_features,
        out_features,
        rank=32,
        alpha=32,
        dropout=0.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # Frozen base transformation
        self.base = nn.Linear(
            in_features,
            out_features,
        )

        # Freeze original parameters
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        # LoRA adaptation matrices
        self.lora_A = nn.Parameter(
            torch.zeros(
                rank,
                in_features,
            )
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                out_features,
                rank,
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.scaling = (
            alpha / rank
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize LoRA parameters.
        """

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=5 ** 0.5,
        )

        nn.init.zeros_(
            self.lora_B
        )

    def forward(self, x):
        """
        Forward pass.

        Base output:
            W(x)

        LoRA update:
            scaling * B(A(x))

        Final output:
            W(x) + LoRA update
        """

        base_output = self.base(x)

        lora_output = (
            self.dropout(x)
            @ self.lora_A.T
            @ self.lora_B.T
        )

        return (
            base_output
            + self.scaling * lora_output
        )


# ============================================================
# 2. Occlusion-Aware Conditioning
# ============================================================

class OcclusionAwareConditioning(nn.Module):
    """
    Encodes the occlusion level and combines it with
    the text conditioning representation.

    The occlusion value is expected to be normalized
    between 0 and 1.

    Example:
        0.20 -> low occlusion
        0.50 -> high occlusion
    """

    def __init__(
        self,
        text_embedding_dim,
        hidden_dim=None,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = text_embedding_dim

        self.occlusion_projection = nn.Sequential(
            nn.Linear(
                1,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                text_embedding_dim,
            ),
        )

    def forward(
        self,
        text_embeddings,
        occlusion_level,
    ):
        """
        Args:
            text_embeddings:
                Text conditioning tensor.

            occlusion_level:
                Occlusion severity in [0, 1].

        Returns:
            Occlusion-aware conditioning.
        """

        if not torch.is_tensor(
            occlusion_level
        ):
            occlusion_level = torch.tensor(
                occlusion_level,
                dtype=text_embeddings.dtype,
                device=text_embeddings.device,
            )

        if occlusion_level.dim() == 0:
            occlusion_level = (
                occlusion_level
                .view(1, 1)
            )

        elif occlusion_level.dim() == 1:
            occlusion_level = (
                occlusion_level
                .view(-1, 1)
            )

        occlusion_embedding = (
            self.occlusion_projection(
                occlusion_level
            )
        )

        # Expand to match token dimension
        if text_embeddings.dim() == 3:

            occlusion_embedding = (
                occlusion_embedding
                .unsqueeze(1)
            )

        return (
            text_embeddings
            + occlusion_embedding
        )


# ============================================================
# 3. Concept Gating
# ============================================================

class ConceptGate(nn.Module):
    """
    Learnable gate for controlling the contribution of
    a specific concept at a selected diffusion layer.

    Concepts:
        - lane_geometry
        - environment
    """

    def __init__(
        self,
        embedding_dim,
    ):
        super().__init__()

        self.gate = nn.Parameter(
            torch.ones(
                embedding_dim
            )
        )

    def forward(
        self,
        x,
    ):

        return (
            x
            * torch.sigmoid(
                self.gate
            )
        )


# ============================================================
# 4. OA-LoRA Layer Assignment
# ============================================================

class OALayerAssignment:
    """
    Defines the layer-wise concept assignment used by OA-LoRA.

    Default configuration:

        Layers 4 and 6:
            Lane geometry

        Layer 7:
            Environmental appearance/style

    This assignment can be changed for ablation studies.
    """

    def __init__(
        self,
        geometry_layers=None,
        environment_layers=None,
    ):

        if geometry_layers is None:
            geometry_layers = [
                4,
                6,
            ]

        if environment_layers is None:
            environment_layers = [
                7,
            ]

        self.geometry_layers = set(
            geometry_layers
        )

        self.environment_layers = set(
            environment_layers
        )

    def get_concept(
        self,
        layer_index,
    ):
        """
        Return the concept assigned to
        a diffusion layer.
        """

        if (
            layer_index
            in self.geometry_layers
        ):
            return "lane_geometry"

        if (
            layer_index
            in self.environment_layers
        ):
            return "environment"

        return "default"


# ============================================================
# 5. OA-LoRA Adapter
# ============================================================

class OALoRAAdapter(nn.Module):
    """
    Main OA-LoRA adaptation module.

    It combines:

        - Low-rank adaptation
        - Occlusion-aware conditioning
        - Layer-wise concept gating
    """

    def __init__(
        self,
        embedding_dim,
        rank=32,
        alpha=32,
        geometry_layers=None,
        environment_layers=None,
    ):
        super().__init__()

        # LoRA adapters
        self.geometry_adapter = LoRALinear(
            embedding_dim,
            embedding_dim,
            rank=rank,
            alpha=alpha,
        )

        self.environment_adapter = LoRALinear(
            embedding_dim,
            embedding_dim,
            rank=rank,
            alpha=alpha,
        )

        # Concept gates
        self.geometry_gate = ConceptGate(
            embedding_dim
        )

        self.environment_gate = ConceptGate(
            embedding_dim
        )

        # Occlusion conditioning
        self.occlusion_conditioning = (
            OcclusionAwareConditioning(
                embedding_dim
            )
        )

        # Layer assignment
        self.layer_assignment = (
            OALayerAssignment(
                geometry_layers=geometry_layers,
                environment_layers=environment_layers,
            )
        )

    def forward(
        self,
        hidden_states,
        layer_index,
        occlusion_level=None,
    ):
        """
        Apply the OA-LoRA adaptation according
        to the assigned diffusion layer.

        Args:
            hidden_states:
                Input hidden representation.

            layer_index:
                Current diffusion network layer.

            occlusion_level:
                Normalized occlusion severity.

        Returns:
            Adapted hidden representation.
        """

        concept = (
            self.layer_assignment
            .get_concept(
                layer_index
            )
        )

        # ----------------------------------------------------
        # Lane geometry adaptation
        # ----------------------------------------------------

        if concept == "lane_geometry":

            adapted = (
                self.geometry_adapter(
                    hidden_states
                )
            )

            adapted = (
                self.geometry_gate(
                    adapted
                )
            )

            return (
                hidden_states
                + adapted
            )

        # ----------------------------------------------------
        # Environmental appearance adaptation
        # ----------------------------------------------------

        elif concept == "environment":

            adapted = (
                self.environment_adapter(
                    hidden_states
                )
            )

            adapted = (
                self.environment_gate(
                    adapted
                )
            )

            return (
                hidden_states
                + adapted
            )

        # ----------------------------------------------------
        # Default layer
        # ----------------------------------------------------

        return hidden_states


# ============================================================
# 6. Parameter Utility
# ============================================================

def mark_only_lora_trainable(
    model,
):
    """
    Freeze all model parameters except
    parameters belonging to OA-LoRA modules.
    """

    for name, parameter in model.named_parameters():

        if (
            "lora_A" in name
            or "lora_B" in name
            or "gate" in name
            or "occlusion_projection" in name
        ):

            parameter.requires_grad = True

        else:

            parameter.requires_grad = False


def count_trainable_parameters(
    model,
):
    """
    Count trainable parameters.
    """

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    return trainable, total


# ============================================================
# 7. Example
# ============================================================

if __name__ == "__main__":

    # Example feature dimension
    embedding_dim = 768

    # Create OA-LoRA adapter
    adapter = OALoRAAdapter(
        embedding_dim=embedding_dim,
        rank=32,
        alpha=32,
        geometry_layers=[
            4,
            6,
        ],
        environment_layers=[
            7,
        ],
    )

    # Example input
    hidden_states = torch.randn(
        1,
        77,
        embedding_dim,
    )

    # Example occlusion level
    occlusion_level = 0.40

    # Example layer 4:
    # Lane geometry adaptation
    output_geometry = adapter(
        hidden_states,
        layer_index=4,
        occlusion_level=occlusion_level,
    )

    # Example layer 7:
    # Environmental adaptation
    output_environment = adapter(
        hidden_states,
        layer_index=7,
        occlusion_level=occlusion_level,
    )

    trainable, total = (
        count_trainable_parameters(
            adapter
        )
    )

    print(
        "OA-LoRA initialized successfully."
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"Total parameters: "
        f"{total:,}"
    )

    print(
        "Geometry output shape:",
        output_geometry.shape,
    )

    print(
        "Environment output shape:",
        output_environment.shape,
    )
