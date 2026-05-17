import datetime
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn)
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy, JaccardIndex
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from wrc.utils.training import get_batch, get_dataloaders

LOG_INTERVAL = 100


def get_feature(target: str) -> int:
    """Resolve feature dimensions from backbone name patterns"""
    model_name = target.lower()

    if "vits" in model_name or "vit_small" in model_name:
        return 384
    if "vitb" in model_name or "vit_base" in model_name:
        return 768
    if "vitl" in model_name or "vit_large" in model_name:
        return 1024
    if model_name == "efficientnet_b4":
        return 128
    if model_name == "maskclip":
        return 512
    if model_name == "radio_v2.5-h":
        return 1280
    if model_name == "radio_v2.5-l":
        return 1024
    if model_name == "radio_v2.5-b":
        return 768

    raise ValueError(f"Unsupported backbone: {model_name}")


def eval_metrics(gt, pred, min_depth=1e-3, max_depth=10):
    mask_1 = gt > min_depth
    mask_2 = gt < max_depth
    mask = np.logical_and(mask_1, mask_2)

    gt = gt[mask]
    pred = pred[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25**2).mean()
    d3 = (thresh < 1.25**3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err**2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(
        d1=d1,
        d2=d2,
        d3=d3,
        abs_rel=abs_rel,
        rmse=rmse,
        log_10=log_10,
        rmse_log=rmse_log,
        silog=silog,
        sq_rel=sq_rel,
    )


class UpsamplerEvaluator:

    def __init__(self, model, backbone, device, cfg, writer, console):
        self.model, self.backbone, self.device, self.cfg, self.writer, self.console = (
            model,
            backbone,
            device,
            cfg,
            writer,
            console,
        )

        self.mean = backbone.config["mean"]
        self.std = backbone.config["std"]

        self.model_type = cfg.model.name  # 'wrc'
        self.console.print(f"[bold cyan]Model type: {self.model_type}[/bold cyan]")

        feature_dim = get_feature(cfg.backbone.name)

        # Initialize task-specific components
        if "seg" == cfg.eval.task:
            self.accuracy_metric = Accuracy(num_classes=cfg.metrics.seg.num_classes, task="multiclass").to(device)
            self.iou_metric = JaccardIndex(num_classes=cfg.metrics.seg.num_classes, task="multiclass").to(device)
            if self.model_type == "wrc":
                self.classifier = nn.Conv2d(cfg.model.in_channels, cfg.metrics.seg.num_classes, 1).to(device)
            else:
                self.classifier = nn.Conv2d(feature_dim, cfg.metrics.seg.num_classes, 1).to(device)
        elif "depth" == cfg.eval.task:
            from transformers import (AutoImageProcessor,
                                      AutoModelForDepthEstimation)

            from wrc.loss import GradientLoss, SigLoss

            if self.model_type == "wrc":
                self.classifier = nn.Conv2d(
                    (2 * cfg.model.in_channels),
                    256,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                ).to(device)
            else:
                self.classifier = nn.Conv2d(
                    (2 * feature_dim),
                    256,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                ).to(device)
            self.image_processor = AutoImageProcessor.from_pretrained("/hy57/zitong/zhiyuan/Depth-Anything-V2-Small-hf")
            self.depth_model = (
                AutoModelForDepthEstimation.from_pretrained("/hy57/zitong/zhiyuan/Depth-Anything-V2-Small-hf").to(device).eval()
            )
            self.sigloss = SigLoss(
                valid_mask=True,
                loss_weight=1.0,
                warm_up=True,
                max_depth=cfg.metrics.depth.max_depth,
            )
            self.gradientloss = GradientLoss(valid_mask=True, loss_weight=0.5, max_depth=cfg.metrics.depth.max_depth)

    def set_up_classifier(self, checkpoint_path):
        """Load classifier weights from a checkpoint."""
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path)
            self.classifier.load_state_dict(checkpoint["model_state_dict"])
            self.console.print(f"Loaded classifier from checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    def set_optimizer(self, cfg):
        params = []
        params_classifier = self.classifier.parameters()
        params_model = self.model.parameters()

        params = list(params_classifier)
        optimizer = instantiate(cfg.optimizer, params=params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Log number of parameters
        num_params = sum(p.numel() for p in params if p.requires_grad)
        self.log_print(f"[bold cyan]Number of optimized parameters: {num_params:,}[/bold cyan]")

    def log_print(self, *args, **kwargs):
        """Log to both file and terminal with immediate updates"""
        # Write to terminal
        Console(force_terminal=True).print(*args, **kwargs)
        # Write to file and flush
        self.console.print(*args, **kwargs)
        if hasattr(self.console, "file") and self.console.file:
            self.console.file.flush()

    def log_tensorboard(self, step, loss=None, metrics=None):
        """Log losses and metrics to TensorBoard."""
        if loss is not None:
            self.writer.add_scalar("Loss/Step", loss, step)
        if metrics is not None:
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, step)

    def process_batch(self, image_batch, target, is_training=True):
        H, W = target.shape[-2:]
        with torch.no_grad():
            pred = self.backbone(image_batch)
            patch_tokens, cls_token = pred[0], pred[1]

            if self.model_type == "wrc":
                pred = self.model(image_batch, patch_tokens)
            else:
                pred = self.model(image_batch, patch_tokens, (H, W))

        if self.cfg.eval.task == "depth":
            cls_token = F.normalize(cls_token, dim=2)[:, 0, :]  # Extract CLS token
            cls_token = cls_token[:, :, None, None]
            pred = torch.cat([pred, cls_token.expand_as(pred)], dim=1)

        pred = self.classifier(pred)

        # WRC outputs may need resizing to match the target map
        if pred.shape[-2:] != (H, W):
            pred = F.interpolate(pred, size=(H, W), mode="bilinear")

        if self.cfg.eval.task == "seg":
            if target.shape[-2:] != pred.shape[-2:]:
                target = (
                    F.interpolate(
                        target.unsqueeze(1),
                        size=pred.shape[-2:],
                        mode="nearest-exact",
                    )
                    .squeeze(1)
                    .to(target.dtype)
                )

            # Create mask for valid pixels (not 255)
            valid_mask = target != 255

            # Reshape predictions and targets
            pred = rearrange(pred, "b c h w -> (b h w) c")
            target = rearrange(target, "b h w -> (b h w)")
            valid_mask = rearrange(valid_mask, "b h w -> (b h w)")

            # Apply mask to both pred and target
            pred = pred[valid_mask]
            target = target[valid_mask]

            return pred, target

        elif self.cfg.eval.task == "depth":
            depth_image_batch = (255 * image_batch.permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)
            inputs = self.image_processor(images=depth_image_batch, return_tensors="pt").to("cuda")
            with torch.no_grad():
                pseudo_depth = self.depth_model(**inputs)["predicted_depth"]

            target = F.interpolate(
                pseudo_depth.unsqueeze(1),
                size=pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            bins = torch.linspace(
                self.cfg.metrics.depth.min_depth,
                self.cfg.metrics.depth.max_depth,
                256,
                device="cuda",
            )
            # Dinov2
            pred = F.relu(pred)
            eps = 0.1
            pred = pred + eps
            pred = pred / pred.sum(dim=1, keepdim=True)
            pred = torch.einsum("ikmn,k->imn", [pred, bins]).unsqueeze(dim=1)
            return pred, target

        else:
            return NotImplementedError

    def train(
        self,
        train_dataloader,
        progress,
        epoch,
        start_time,
    ):
        self.log_print(f"[yellow]Training model epoch {epoch+1}...[/yellow]")
        self.backbone.eval()
        self.model.eval()
        self.classifier.train()

        epoch_task = progress.add_task(
            f"Epoch {epoch+1}/{self.cfg.num_epochs}",
            total=len(train_dataloader),
            loss=0.0,
            step=0,
        )
        total_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            # Process batch using get_batch
            batch = get_batch(batch, self.device)
            image_batch = batch["image"]
            target = batch["label"].to(self.device)

            if random.random() < 0.5:
                image_batch = torch.flip(image_batch, dims=[3])  # Flip along width (W)
                target = torch.flip(target, dims=[2])  # Flip along width (W), assuming (H, W) or (1, H, W)

            self.optimizer.zero_grad()

            pred, target = self.process_batch(image_batch, target, is_training=True)

            if self.cfg.eval.task == "seg":
                loss = F.cross_entropy(pred, target)
            elif self.cfg.eval.task == "depth":
                loss = self.sigloss(pred, target) + self.gradientloss(pred, target)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            avg_loss = total_loss / (batch_idx + 1)

            # Update progress only every log_interval iterations
            if (batch_idx + 1) % LOG_INTERVAL == 0 or batch_idx == len(train_dataloader) - 1:
                elapsed_time = datetime.datetime.now() - start_time
                elapsed_str = str(elapsed_time).split(".")[0]
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Update progress bar
                progress.update(
                    epoch_task,
                    advance=LOG_INTERVAL,
                    loss=avg_loss,
                    step=batch_idx + 1,
                )

                # Log with learning rate
                self.log_print(
                    f"[cyan]Iteration {batch_idx + 1}[/cyan] - "
                    f"Loss: {avg_loss:.6f} - "
                    f"LR: {current_lr:.5e} - "
                    f"Elapsed Time: {elapsed_str}"
                )

                # Force console update
                if self.console and hasattr(self.console, "file"):
                    self.console.file.flush()

                # Ensure progress is displayed immediately
                progress.refresh()

                # Log loss to TensorBoard
                self.log_tensorboard(len(train_dataloader) + batch_idx, loss=avg_loss)

            if self.cfg.sanity and batch_idx == 0:
                break

            if self.cfg.sanity:
                break

        self.scheduler.step()
        # Add learning rate to epoch summary
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.log_print(
            f"[bold cyan]Epoch {epoch+1} Summary:[/bold cyan] " f"Loss = {avg_loss:.6f} - " f"LR = {current_lr:.2e}"
        )

        return

    def save_checkpoint(self, checkpoint_path):
        console = self.console
        # Save final model state after training
        checkpoint = {
            "epoch": self.cfg.num_epochs - 1,
            "model_state_dict": self.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "task": self.cfg.eval.task,
            "backbone": self.cfg.backbone.name,
        }
        torch.save(checkpoint, checkpoint_path)
        self.log_print(f"[bold green]Training completed. Model saved at: {checkpoint_path}[/bold green]")
        console.file.flush()
        return

    @torch.inference_mode()
    def evaluate(self, dataloader, epoch):
        self.log_print("[yellow]Evaluating model...[/yellow]")
        torch.cuda.empty_cache()

        self.backbone.eval()
        self.model.eval()
        self.classifier.eval()

        # Reset metrics at the start of evaluation
        if self.cfg.eval.task == "seg":
            self.accuracy_metric.reset()
            self.iou_metric.reset()

        elif self.cfg.eval.task == "depth":
            results = {
                "d1": 0,
                "d2": 0,
                "d3": 0,
                "abs_rel": 0,
                "sq_rel": 0,
                "rmse": 0,
                "rmse_log": 0,
                "log_10": 0,
                "silog": 0,
            }

            nsamples = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = get_batch(batch, self.device)
            image_batch = batch["image"]
            target = batch["label"].to(self.device)

            # Process batch and get masked predictions and targets
            pred, target = self.process_batch(image_batch, target, is_training=False)

            if self.cfg.eval.task == "seg":
                self.accuracy_metric(pred, target)
                self.iou_metric(pred, target)

            elif self.cfg.eval.task == "depth":
                cur_results = eval_metrics(target.cpu().detach().numpy(), pred.cpu().detach().numpy())

                for k in results.keys():
                    results[k] += cur_results[k]
                nsamples += 1

            if self.cfg.sanity and batch_idx == 0:
                break

        metrics = {}
        if self.cfg.eval.task == "seg":
            metrics.update(
                {
                    "accuracy": self.accuracy_metric.compute().item(),
                    "iou": self.iou_metric.compute().item(),
                }
            )
        elif self.cfg.eval.task == "depth":
            for k in results.keys():
                metrics[k] = results[k] / nsamples

        # Log metrics to TensorBoard
        self.log_tensorboard(step=epoch, metrics=metrics)

        self.log_print(f"[bold green]Results: {metrics}[/bold green]")
        return

    @torch.inference_mode()
    def simple_inference(self, image_batch):
        self.backbone.eval()
        self.model.eval()
        self.classifier.eval()

        H, W = image_batch.shape[-2:]
        with torch.no_grad():
            hr_feats, _ = self.backbone(image_batch)
            if self.model_type == "wrc":
                features = self.model(image_batch, hr_feats)
            else:
                features = self.model(image_batch, hr_feats, (H, W))

            if self.cfg.eval.task == 'depth':
                pred1 = self.backbone(image_batch)
                patch_tokens, cls_token = pred1[0], pred1[1]
                cls_token = F.normalize(cls_token, dim=2)[:, 0, :]  # Extract CLS token
                cls_token = cls_token[:, :, None, None]
                features = torch.cat([features, cls_token.expand_as(features)], dim=1)

        pred = features  # Get the last prediction
        pred = self.classifier(pred)  # Pass through the classifier
        pred = pred.argmax(dim=1)  # Get the predicted class for each pixel

        return pred, features, hr_feats


@hydra.main(version_base='1.1', config_path="../config", config_name="eval")
def main(cfg):
    task = cfg.eval.task

    # Setup Classifier
    # Either we train one classifier per backbone.
    checkpoint_dir = f"./checkpoints/{task}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/{cfg.model.name}.pth"

    # Create persistent consoles instead of creating new ones each time
    terminal_console = Console()  # Terminal output
    if Path(checkpoint_path).exists():
        file_name = f"eval_{cfg.model.name}_{task}.log"
    else:
        file_name = f"train_{cfg.model.name}_{task}.log"
    current_run_dir = os.getcwd()
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    source_path = os.path.join(current_run_dir, file_name)
    symlink_path = os.path.join(current_run_dir, log_dir, file_name)
    if not os.path.exists(symlink_path):
        os.symlink(source_path, symlink_path)
    file_console = Console(file=open(source_path, "w"))

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(current_run_dir, "./tb", file_name.replace(".log", "")))

    def log_print(*args, **kwargs):
        """Log to both terminal and file with immediate flushing"""
        terminal_console.print(*args, **kwargs)
        file_console.print(*args, **kwargs)
        file_console.file.flush()

    # Start logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"\n[bold blue]{'='*50}[/bold blue]")
    log_print(f"[bold blue]Starting at {timestamp}[/bold blue]")
    log_print(f"[bold green]Configuration:[/bold green]")
    log_print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"[bold yellow]Using device: {device}[/bold yellow]")

    log_print(f"\n[bold cyan]Processing {task} task:[/bold cyan]")
    log_print(f"\n[bold cyan]Image size: {cfg.img_size}[/bold cyan]")

    # Setup Backbone
    backbone = instantiate(cfg.backbone).to(device)
    backbone.requires_grad_(False)
    backbone.eval()

    # Setup Model
    model = instantiate(cfg.model).to(device)
    if cfg.eval.model_ckpt:
        checkpoint = torch.load(cfg.eval.model_ckpt)
        model.load_state_dict(checkpoint['wrc'], strict=False)
        log_print(f"[green]Loaded model from checkpoint: {cfg.eval.model_ckpt}[/green]")
    else:
        model.train()

    # Setup Dataloaders
    if cfg.eval.task == "depth":
        mean = [0, 0, 0]
        std = [1, 1, 1]
    else:
        mean, std = None, None
    train_loader, val_loader = get_dataloaders(cfg, backbone, is_evaluation=True, mean=mean, std=std)
    log_print(f"[bold cyan]Train Dataset size: {len(train_loader.dataset)}[/bold cyan]")
    log_print(f"[bold cyan]Val Dataset size: {len(val_loader.dataset)}[/bold cyan]")

    # Setup Evaluator
    evaluator = UpsamplerEvaluator(model, backbone, device, cfg, writer, file_console)

    # Already trained
    if Path(checkpoint_path).exists():
        log_print(f"[green]Loading classifier from {checkpoint_path}[/green]")
        evaluator.set_up_classifier(checkpoint_path)
        evaluator.evaluate(val_loader, epoch=0)
    else:
        log_print(f"[yellow]Training classifier... {checkpoint_path} not found[/yellow]\n")
        evaluator.set_optimizer(cfg)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[yellow]Loss: {task.fields[loss]:.6f}"),
            TextColumn("[green]Step: {task.fields[step]:.6e}"),
            console=file_console,
        )

        start_time = datetime.datetime.now()

        with progress:
            # Standard training for upsampler
            log_print(f"[yellow]Training for {cfg.num_epochs} epochs[/yellow]\n")

            for epoch in range(cfg.num_epochs):
                evaluator.train(train_loader, progress, epoch, start_time)
                evaluator.evaluate(val_loader, epoch)
                if cfg.sanity:
                    break

            evaluator.save_checkpoint(checkpoint_path)

    file_console.file.close()
    writer.close()  # Close TensorBoard writer


if __name__ == "__main__":
    main()
