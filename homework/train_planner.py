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
    num_epoch: int = 40,
    lr: float = 1e-4,
    batch_size: int = 128,
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
    mse_loss = torch.nn.MSELoss()
    l1_loss_fn = torch.nn.L1Loss(reduction='none')  # use reduction='none' to apply mask

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    global_step = 0

    planner_metric = PlannerMetric()

    for epoch in range(num_epoch):
        model.train()
        train_losses = []
        planner_metric.reset()

        for item in train_data:
            waypoints = item['waypoints'].to(device)
            labels_mask = item['waypoints_mask'].to(device)

            if model_name.startswith("cnn"):
                assert "image" in item, "Expected 'image' in dataset for CNN model"
                image = item["image"].to(device)
                predicted_waypoints = model(image)
            else:
                assert "track_left" in item and "track_right" in item, "Expected 'track_left' and 'track_right' for MLP model"
                track_left = item['track_left'].to(device)
                track_right = item['track_right'].to(device)
                predicted_waypoints = model(track_left, track_right)

            # Compute masked loss
            loss = l1_loss_fn(predicted_waypoints, waypoints)
            loss = loss * labels_mask[..., None]  # Apply mask
            loss = loss.sum() / labels_mask.sum()  # Average over valid points

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            logger.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

            planner_metric.add(predicted_waypoints, waypoints, labels_mask)

        avg_train_loss = np.mean(train_losses)
        metrics = planner_metric.compute()

        # Validation
        val_losses = []
        model.eval()
        planner_metric.reset()

        with torch.no_grad():
            for item in val_data:
                waypoints = item['waypoints'].to(device)
                labels_mask = item['waypoints_mask'].to(device)

                if model_name.startswith("cnn"):
                    assert "image" in item, "Expected 'image' in dataset for CNN model"
                    image = item["image"].to(device)
                    predicted_waypoints = model(image)
                else:
                    assert "track_left" in item and "track_right" in item, "Expected 'track_left' and 'track_right' for MLP model"
                    track_left = item['track_left'].to(device)
                    track_right = item['track_right'].to(device)
                    predicted_waypoints = model(track_left, track_right)

                val_loss = l1_loss_fn(predicted_waypoints, waypoints)
                val_loss = val_loss * labels_mask[..., None]
                val_loss = val_loss.sum() / labels_mask.sum()
                val_losses.append(val_loss.item())

                planner_metric.add(predicted_waypoints, waypoints, labels_mask)

        avg_val_loss = np.mean(val_losses)

        logger.add_scalar("train/loss_avg", avg_train_loss, global_step)
        logger.add_scalar("val/loss_avg", avg_val_loss, global_step)

        logger.add_scalar("train/l1_error", metrics['l1_error'], global_step)
        logger.add_scalar("train/longitudinal_error", metrics['longitudinal_error'], global_step)
        logger.add_scalar("train/lateral_error", metrics['lateral_error'], global_step)

        print(f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
              f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
              f"L1 Error={metrics['l1_error']:.4f}, "
              f"Longitudinal Error={metrics['longitudinal_error']:.4f}, "
              f"Lateral Error={metrics['lateral_error']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model)
            torch.save(model.state_dict(), log_dir / f"{model_name}_best.th")
            print(f"Best model saved to {log_dir / f'{model_name}_best.th'}")

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

