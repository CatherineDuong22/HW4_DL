"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.nn.functional as F
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data  # Adjust if necessary for your dataset
from homework.metrics import PlannerMetric  # Optional: you can use this or create a new metric


def train_planner(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner_model",
    num_epoch: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load model
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    # Load training and validation data
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # Loss function
    mse_loss = torch.nn.MSELoss()  # For waypoint prediction

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")  # Track the best validation loss
    global_step = 0

    # Initialize the PlannerMetric for tracking errors
    planner_metric = PlannerMetric()

    for epoch in range(num_epoch):
        model.train()
        train_losses = []

        # Reset the metric for each epoch
        planner_metric.reset()

        for item in train_data:
            # Access the data from the dictionary
            track_left = item['track_left'].to(device)
            track_right = item['track_right'].to(device)
            waypoints = item['waypoints'].to(device)
            labels_mask = item['waypoints_mask'].to(device)  # Assuming mask is part of the data

            # Forward pass through the model
            predicted_waypoints = model(track_left, track_right)

            # Compute loss
            loss = torch.nn.L1Loss()(predicted_waypoints, waypoints)

            # Apply the mask to the loss
            loss = loss * labels_mask[..., None]  # Broadcasting the mask to match the loss shape

            # Average the loss over the valid points (where mask is 1)
            loss = loss.sum() / labels_mask.sum()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging the training loss
            train_losses.append(loss.item())
            logger.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

            # Add metrics for the planner (longitudinal, lateral, and l1 error)
            planner_metric.add(predicted_waypoints, waypoints, labels_mask)

        # Compute average loss for the epoch
        avg_train_loss = np.mean(train_losses)

        # Compute planner metrics
        metrics = planner_metric.compute()

        # Validation
        val_losses = []
        model.eval()

        with torch.no_grad():
            for item in val_data:
                # Extract data and move them to the device
                track_left = item["track_left"].to(device)
                track_right = item["track_right"].to(device)
                waypoints = item["waypoints"].to(device)
                labels_mask = item["waypoints_mask"].to(device)

                # Forward pass through the model
                predicted_waypoints = model(track_left, track_right)

                # Compute loss
                val_loss = torch.nn.L1Loss()(predicted_waypoints, waypoints)

                # Apply the mask to the loss
                val_loss = val_loss * labels_mask[..., None]
                val_loss = val_loss.sum() / labels_mask.sum()
                val_losses.append(val_loss.item())

                # Add metrics for the planner
                planner_metric.add(predicted_waypoints, waypoints, labels_mask)

        # Compute average validation loss
        avg_val_loss = np.mean(val_losses)

        # Log metrics
        logger.add_scalar("train/loss_avg", avg_train_loss, global_step)
        logger.add_scalar("val/loss_avg", avg_val_loss, global_step)

        # Log planner metrics
        logger.add_scalar("train/l1_error", metrics['l1_error'], global_step)
        logger.add_scalar("train/longitudinal_error", metrics['longitudinal_error'], global_step)
        logger.add_scalar("train/lateral_error", metrics['lateral_error'], global_step)

        # Print losses and metrics
        print(f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
              f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
              f"L1 Error={metrics['l1_error']:.4f}, "
              f"Longitudinal Error={metrics['longitudinal_error']:.4f}, "
              f"Lateral Error={metrics['lateral_error']:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model)
            torch.save(model.state_dict(), log_dir / f"{model_name}_best.th")
            print(f"Best model saved to {log_dir / f'{model_name}_best.th'}")

    # Final model save
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Final model saved to {log_dir / f'{model_name}.th'}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)

    train_mlp_planner(**vars(parser.parse_args()))

