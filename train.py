import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.utils.tensorboard as tb
from torch.optim.lr_scheduler import StepLR  # Optional: for learning rate scheduling
import torch.nn.functional as F
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric  # Import IoU and Depth Error metric

def iou_loss(pred, target, eps=1e-6):
    """
    Computes the IoU loss for segmentation.
    
    :param pred: Predicted segmentation map (logits before softmax).
    :param target: Ground truth segmentation map (integer labels).
    :param eps: Small epsilon to avoid division by zero.
    :return: IoU loss value.
    """
    pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities
    pred = pred[:, 1, :, :]  # Assuming class 1 is the foreground (change as needed)

    target = (target == 1).float()  # Convert ground truth to binary mask

    intersection = (pred * target).sum(dim=(1, 2))  # Intersection over batch
    union = (pred + target).sum(dim=(1, 2)) - intersection  # Union

    iou = (intersection + eps) / (union + eps)
    return 1 - iou.mean()  # IoU loss (1 - IoU)

def train_detection(
    exp_dir: str = "logs",
    model_name: str = "detection_model",
    num_epoch: int = 20,
    lr: float = 1e-2,
    batch_size: int = 128,
    seed: int = 2024,
    lr_scheduler_step: int = 5,  # Optional: StepLR scheduler step
    segmentation_loss_weight: float = 1.0,  # Weight for segmentation loss
    depth_loss_weight: float = 1.0,  # Weight for depth loss
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

    custom_weights = torch.tensor([1.0, 1.5, 4.0]).cuda()  # Move weights to the GPU

    # Loss functions
    cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=custom_weights)  # Segmentation loss
    mse_loss = torch.nn.MSELoss()  # Depth prediction loss

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    

    # Optional: Learning rate scheduler
    # scheduler = StepLR(optimizer, step_size=lr_scheduler_step, gamma=0.1)

    detection_metric = DetectionMetric(num_classes=3)  

    best_val_loss = float("inf")  # Track the best validation loss
    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        detection_metric.reset()
        train_losses = []

        for item in train_data:
            # Access the data from the dictionary using the appropriate keys
            image = item['image'].to(device)
            seg_targets = item['track'].to(device)
            depth = item['depth'].to(device)
            
            # Assuming that the model returns both segmentation logits and depth predictions
            outputs = model(image)
            segmentation_logits = outputs[0]
            depth_predictions = outputs[1]
            # Segmentation loss (using cross-entropy)
            segmentation_logits = F.interpolate(segmentation_logits, size=seg_targets.shape[-2:], mode="bilinear", align_corners=False)

            seg_loss = cross_entropy_loss(segmentation_logits, seg_targets)

            # Depth loss (e.g., using mean squared error)

            depth_predictions = F.interpolate(depth_predictions.unsqueeze(1), size=depth.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

            depth_loss = mse_loss(depth_predictions, depth)

                # Compute IoU loss
            iou_loss_value = iou_loss(segmentation_logits, seg_targets)
            # Total loss combining both tasks
            total_loss = seg_loss + depth_loss + iou_loss_value

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update training metrics
            seg_preds = segmentation_logits.argmax(dim=1)
            detection_metric.add(seg_preds, seg_targets, depth_predictions, depth)

            # Logging the training loss
            train_losses.append(total_loss.item())
            logger.add_scalar("train/loss", total_loss.item(), global_step)
            global_step += 1

        # Compute IoU and depth error for training
        train_metrics = detection_metric.compute()
        train_iou = train_metrics["iou"]
        train_depth_error = train_metrics["abs_depth_error"]

        # Validation
        val_losses = []
        model.eval()
        detection_metric.reset()

        with torch.no_grad():
            for item in val_data:
                # Extract data and move them to the device
                img = item["image"].to(device)
                targets = {k: v.to(device) for k, v in item.items() if k != "image"}

                # Forward pass through the model
                logits, raw_depth = model(img)

                # Ensure that raw_depth matches the target depth size
                raw_depth = F.interpolate(raw_depth.unsqueeze(1), size=targets["depth"].shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
                logits = F.interpolate(logits, size=seg_targets.shape[-2:], mode="bilinear", align_corners=False)
                # Compute losses
                seg_loss = cross_entropy_loss(logits, targets["track"])
                depth_loss = mse_loss(raw_depth, targets["depth"])

                # Combine the losses
                total_val_loss = segmentation_loss_weight * seg_loss + depth_loss_weight * depth_loss
                val_losses.append(total_val_loss.item())


# # Optionally, log or print the validation loss
# avg_val_loss = np.mean(val_losses)
# logger.add_scalar("val/loss", avg_val_loss, global_step)
# print(f"Validation Loss: {avg_val_loss:.4f}")


                # Update validation metrics
                seg_preds = logits.argmax(dim=1)
                detection_metric.add(seg_preds, targets["track"], raw_depth, targets["depth"])

        # Compute validation metrics
        val_metrics = detection_metric.compute()
        val_iou = val_metrics["iou"]
        val_abs_depth_error = val_metrics["abs_depth_error"]
        val_tp_depth_error = val_metrics['tp_depth_error']
        # Compute average loss
        avg_train_loss = torch.tensor(train_losses).mean()
        avg_val_loss = torch.tensor(val_losses).mean()

        # Log metrics
        logger.add_scalar("train/iou", train_iou, epoch)
        logger.add_scalar("train/depth_error", train_depth_error, epoch)
        logger.add_scalar("val/iou", val_iou, epoch)
        logger.add_scalar("val/abs_depth_error", val_abs_depth_error, epoch)
        logger.add_scalar("val/tp_depth_error", val_tp_depth_error, epoch)
        logger.add_scalar("train/loss_avg", avg_train_loss, global_step)
        logger.add_scalar("val/loss_avg", avg_val_loss, global_step)

        # Print losses and IoU metrics
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}    "
            f"Train IoU={train_iou:.4f}, Val IoU={val_iou:.4f}, "
            f"Val Abs Depth Err={val_abs_depth_error:.4f}, "
            f"Val TP Depth Err={val_tp_depth_error:.4f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model)
            torch.save(model.state_dict(), log_dir / f"{model_name}_best.th")
            print(f"Best model saved to {log_dir / f'{model_name}_best.th'}")

        # # Scheduler step
        # scheduler.step()

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
    parser.add_argument("--lr_scheduler_step", type=int, default=5)
    parser.add_argument("--segmentation_loss_weight", type=float, default=1.0)
    parser.add_argument("--depth_loss_weight", type=float, default=1.0)

    train_detection(**vars(parser.parse_args()))

