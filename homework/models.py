from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class MLPPlanner(nn.Module):
    def __init__(self, n_track: int = 10, n_waypoints: int = 3):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super(MLPPlanner, self).__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Define the MLP layers
        # Flatten the track data to a 1D tensor, input size will be n_track * 4 (x, y for both left and right)
        input_size = n_track * 4
        hidden_size = 128  # You can adjust the hidden size as needed

        # Define the layers of the MLP
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_waypoints * 2)  # Output n_waypoints * 2 (for x and y)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Concatenate track_left and track_right to have shape (b, n_track, 4)
        x = torch.cat([track_left, track_right], dim=-1)  # (B, n_track, 4)

        # Flatten the tensor to shape (B, n_track * 4)
        x = x.view(x.size(0), -1)  # (B, n_track * 4)

        # Pass through the MLP layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape the output to (B, n_waypoints, 2)
        x = x.view(x.size(0), self.n_waypoints, 2)

        return x


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Query embedding: one embedding per waypoint
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Positional encoding for the track input
        self.track_pos_embed = nn.Parameter(torch.randn(n_track * 2, d_model))

        # Input projection to d_model
        self.input_proj = nn.Linear(2, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final output projection to 2D waypoint space
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,   # (b, n_track, 2)
        track_right: torch.Tensor,  # (b, n_track, 2)
        **kwargs,
    ) -> torch.Tensor:
        b = track_left.size(0)

        # Concatenate both boundaries: (b, 2 * n_track, 2)
        track_input = torch.cat([track_left, track_right], dim=1)

        # Project to d_model: (b, 2 * n_track, d_model)
        memory = self.input_proj(track_input)

        # Add positional encoding
        memory = memory + self.track_pos_embed.unsqueeze(0)

        # Get learned queries: (b, n_waypoints, d_model)
        query_pos = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)

        # Decoder cross-attends to memory (track)
        decoder_output = self.decoder(query_pos, memory)

        # Predict 2D waypoints from decoder output
        waypoints = self.output_proj(decoder_output)  # (b, n_waypoints, 2)

        return waypoints

class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
