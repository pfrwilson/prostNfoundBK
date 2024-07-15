from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
import argparse
from datetime import datetime
import json
import os

import pandas as pd
from src.dataset import make_corewise_bk_dataloaders
from src.experiment_setup import setup, ExperimentUtility
import torch
from src.prostnfound import ProstNFound
from src.sam_wrappers import build_medsam
from src.loss import CancerDetectionValidRegionLoss, MaskedPredictionModule
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.amp
from src.utils import calculate_metrics
import numpy as np
from tqdm import tqdm
from src.slurm import add_submitit_args, submit_job
import matplotlib.pyplot as plt


def main():
    def _format_dir(s): 
        # replace %d with current date
        return s.replace("%d", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Training arguments
    train_parser = argparse.ArgumentParser(add_help=False)
    train_parser.add_argument(
        "--exp_dir", type=_format_dir, required=True, help="Experiment directory."
    )
    train_parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    train_parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    train_parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="Number of warmup epochs."
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training."
    )
    train_parser.add_argument(
        "--accumulate_grad_steps", type=int, default=8, help="Accumulate gradient steps."
    )
    train_parser.add_argument("--debug", action="store_true", help="Debug mode.")
    train_parser.add_argument(
        "--use_wandb", action="store_true", help="Whether to log to wandb."
    )
    train_parser.add_argument("--use_amp", action="store_true", help="Whether to use amp.")
    train_parser.add_argument("--device", type=str, default="cuda", help="Device.")

    parser = ArgumentParser(
        add_help=True,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(title="command", dest="command")
    train_parser_ = subparsers.add_parser("train", parents=[train_parser])
    train_parser_submitit = subparsers.add_parser("train_slurm", parents=[train_parser])
    add_submitit_args(train_parser_submitit)

    args = parser.parse_args()
    if args.command == 'train_slurm': 
        submit_job(train, args, setup=["export WANDB_RESUME=allow", "export WANDB_RUN_ID=$SLURM_JOB_ID"])
    elif args.command == 'train':
        train(args)


def train(args):

    if 'SLURM_JOB_ID' in os.environ:
        ckpt_dir_symlink_target = os.path.join("/checkpoint", os.environ["USER"], os.environ["SLURM_JOB_ID"])
    else: 
        ckpt_dir_symlink_target = None
        
    utility = setup(
        args.exp_dir,
        ckpt_dir_symlink_target=ckpt_dir_symlink_target,
        use_distributed=False,
        debug=False,
        conf=vars(args),
        ckpt_file="state.pt",
        use_wandb=args.use_wandb,
        wandb_kwargs={"project": "prostnfound_bk"},
        overwrite=False,
    )

    # ======== DATA =========
    train, val, test = make_corewise_bk_dataloaders(
        batch_sz=args.batch_size, style="last_frame", fold=args.fold, num_folds=args.num_folds, 
    )

    import pandas as pd

    table_train = train.dataset.table
    table_val = val.dataset.table
    table_test = test.dataset.table

    table_train["split"] = "train"
    table_val["split"] = "val"
    table_test["split"] = "test"

    table = pd.concat([table_train, table_val, table_test])
    table.to_csv(os.path.join(args.exp_dir, "metadata.csv"), index=False)

    train_ids = table_train.patient_id.unique()
    val_ids = table_val.patient_id.unique()
    test_ids = table_test.patient_id.unique()

    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)

    # ======== MODEL =========

    sam_backbone = build_medsam()
    model = ProstNFound(sam_backbone, auto_resize_image_to_native=False).to(args.device)
    if utility.state is not None: 
        model.load_state_dict(utility.state["model"])

    # torch.compile(model)
    loss_fn = CancerDetectionValidRegionLoss(
        loss_pos_weight=2.0,
    )

    opt = Adam(model.parameters(), lr=args.lr)
    if utility.state is not None:
        opt.load_state_dict(utility.state["opt"])

    n_iters_per_epoch = len(train)
    scheduler = LambdaLR(opt, LRCalculator(0, args.warmup_epochs, args.epochs, n_iters_per_epoch))
    if utility.state is not None:
        scheduler.load_state_dict(utility.state["scheduler"])
    scaler = torch.cuda.amp.GradScaler()
    if utility.state is not None:
        scaler.load_state_dict(utility.state["scaler"])

    def _log_train_info(d):
        # env.log_info(d)
        utility.log_metrics({f"train/{k}": v for k, v in d.items()})

    def _log_val_info(d):
        # env.log_info(d)
        utility.log_metrics({f"val/{k}": v for k, v in d.items()})

    start_epoch = 0 if utility.state is None else utility.state["epoch"]
    for epoch in range(start_epoch, args.epochs):
        utility.log_info(f"====== Epoch {epoch} ==========")
        state = {}
        state["epoch"] = epoch
        state["model"] = model.state_dict()
        state["opt"] = opt.state_dict()
        state["scheduler"] = scheduler.state_dict()
        state["scaler"] = scaler.state_dict()
        utility.save_checkpoint(state)
        
        metrics, _ = run_epoch(
            train,
            model,
            loss_fn,
            opt,
            scheduler,
            scaler=scaler,
            metric_logger=_log_train_info,
            debug=args.debug,
            use_amp=args.use_amp,
            device=args.device,
            accumulate_grad_steps=args.accumulate_grad_steps,
        )
        utility.log_metrics({f"train/{k}": v for k, v in metrics.items()})
        metrics, _ = run_epoch(
            val,
            model,
            loss_fn,
            args.device,
            metric_logger=_log_val_info,
            debug=args.debug,
            use_amp=args.use_amp,
            train=False,
        )
        utility.log_metrics({f"val/{k}": v for k, v in metrics.items()})

    
def prepare_batch(batch, device):
    img, needle, prostate, label, inv, core_id, patient_id = batch
    img = (img - img.min()) / (img.max() - img.min())
    # from torchvision.transforms import Resize
    # img = Resize((img_size, img_size))(img)
    img = img.unsqueeze(1)
    img = torch.cat([img, img, img], dim=1).float()

    needle = needle.unsqueeze(0)
    prostate = prostate.unsqueeze(0)

    return (
        img.to(device),
        needle.to(device),
        prostate.to(device),
        label.to(device),
        core_id,
    )


def run_epoch(
    loader,
    model,
    loss_fn,
    opt=None,
    scheduler=None,
    scaler=None,
    device="cuda",
    accumulate_grad_steps=1,
    metric_logger=None,
    debug=False,
    use_amp=False,
    train=True,
):
    with torch.set_grad_enabled(train):

        tracker = Tracker()
        model.train(train)

        for i, batch in enumerate(tqdm(loader)):
            if debug and i > 20:
                break

            # forward passes
            img, needle_mask, prostate_mask, label, core_ids = prepare_batch(
                batch, device
            )
            with torch.amp.autocast_mode.autocast(device, enabled=use_amp):
                logits = model(img)
                loss = loss_fn(logits, prostate_mask, needle_mask, label, None)
                loss /= accumulate_grad_steps

                masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
                predictions, batch_idx = MaskedPredictionModule()(logits, masks)
                mean_predictions_in_needle = []
                B = img.shape[0]
                for j in range(B):
                    mean_predictions_in_needle.append(
                        predictions[batch_idx == j].sigmoid().mean()
                    )
                mean_predictions_in_needle = torch.stack(mean_predictions_in_needle)

            step_metrics = {"loss": loss.item()}
            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (i + 1) % accumulate_grad_steps == 0:
                    if use_amp:
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad()
                    else:
                        opt.step()
                        opt.zero_grad()

                scheduler.step()
                step_metrics["lr"] = scheduler.get_last_lr()[0]

            if metric_logger is not None:
                metric_logger(step_metrics)

            # update tracker
            tracker.update(core_ids, mean_predictions_in_needle)

        table = loader.dataset.table.copy()
        table.set_index("core_id", inplace=True)
        d = table.to_dict(orient="index")
        metrics = tracker.compute_metrics(d, log_images=True)

        return metrics, tracker


class Tracker:
    def __init__(self):
        self.core_ids = []
        self.mean_predictions_in_needle = []

    def update(self, core_ids, mean_predictions_in_needle):
        self.core_ids.extend(core_ids)
        self.mean_predictions_in_needle.append(
            mean_predictions_in_needle.detach().cpu()
        )

    def get(self):
        return self.core_ids, torch.cat(self.mean_predictions_in_needle)

    def compute_metrics(self, core_id_to_metadata_dict, log_images=False):
        core_ids, mean_predictions_in_needle = self.get()

        labels = []
        preds = []
        involvement = []
        for core_id, prediction in zip(core_ids, mean_predictions_in_needle):
            label = core_id_to_metadata_dict[core_id]["label"]
            labels.append(label)
            involvement.append(core_id_to_metadata_dict[core_id]["inv"])
            preds.append(prediction.item())

        labels = np.array(labels)
        preds = np.array(preds)
        involvement = np.array(involvement)

        metrics_all = calculate_metrics(preds, labels, log_images=log_images)

        high_inv_mask = (involvement > 0.4) | (labels == 0)
        metrics_high_inv = calculate_metrics(
            preds[high_inv_mask], labels[high_inv_mask], log_images=log_images
        )

        metrics = {}
        metrics.update(metrics_all)
        metrics.update({f"{k}_high_inv": v for k, v in metrics_high_inv.items()})

        return metrics


class LRCalculator:
    def __init__(
        self, frozen_epochs, warmup_epochs, total_epochs, niter_per_ep
    ):
        self.frozen_epochs = frozen_epochs
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.niter_per_ep = niter_per_ep

    def __call__(self, iter):
        if iter < self.frozen_epochs * self.niter_per_ep:
            return 0
        elif (
            iter < (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep
        ):
            return (iter - self.frozen_epochs * self.niter_per_ep) / (
                self.warmup_epochs * self.niter_per_ep
            )
        else:
            cur_iter = (
                iter
                - (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep
            )
            total_iter = (
                self.total_epochs - self.warmup_epochs - self.frozen_epochs
            ) * self.niter_per_ep
            return 0.5 * (1 + np.cos(np.pi * cur_iter / total_iter))


if __name__ == "__main__":
    main()
