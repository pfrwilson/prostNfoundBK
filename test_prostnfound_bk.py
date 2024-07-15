from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
import json
import os
import pandas as pd
import wandb
from src.dataset import make_corewise_bk_dataloaders
import torch
from src.prostnfound import ProstNFound
from src.sam_wrappers import build_medsam
from src.loss import CancerDetectionValidRegionLoss, MaskedPredictionModule
import torch.amp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from train_prostnfound_bk import prepare_batch, run_epoch
from torch import nn
from torch.optim import LBFGS
from skimage.transform import resize
from skimage.morphology import dilation
from skimage.filters import gaussian


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to the config file."
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, required=True, help="Path to the output directory."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint."
    )
    parser.add_argument(
        "--tc_pos_weight", type=float, default=2.0, 
        help="""Positive weight for temperature calibration. Setting this higher will encourage the model to be more sensitive to positive examples."""
    )
    parser.add_argument(
        '--save_heatmaps', action='store_true', help='Save heatmaps.'
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_args = json.load(open(args.config, "r"))
    train_args = Namespace(**train_args)

    # ======== DATA =========
    train, val, test = make_corewise_bk_dataloaders(
        batch_sz=1,
        style="last_frame",
        fold=train_args.fold,
        num_folds=train_args.num_folds,
    )

    # ======== MODEL =========
    sam_backbone = build_medsam()
    model = ProstNFound(sam_backbone, auto_resize_image_to_native=False).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint)["model"])

    # torch.compile(model)
    loss_fn = CancerDetectionValidRegionLoss(
        loss_pos_weight=2.0,
    )

    # extract pixel predictions for validation set - these will be used for temperature scaling
    print("Extracting pixel predictions for temperature calibration.")
    pixel_preds, pixel_labels, core_ids = extract_all_pixel_predictions(
        model, val, args.device
    )

    # ======= TEMPERATURE CALIBRATION ========

    core_ids = np.array(core_ids)

    # fit temperature and bias to center and scale the predictions
    print("Fitting temperature calibration on validation outputs.")
    temp = nn.Parameter(torch.ones(1))
    bias = nn.Parameter(torch.zeros(1))

    optim = LBFGS([temp, bias], lr=1e-3, max_iter=100, line_search_fn="strong_wolfe")

    # weight the loss to account for class imbalance
    pos_weight = (1 - pixel_labels).sum() / pixel_labels.sum()
    # encourage sensitivity over specificity
    pos_weight *= args.tc_pos_weight

    def closure():
        optim.zero_grad()
        logits = pixel_preds / temp + bias
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits[:, 0], pixel_labels)
        loss.backward()
        return loss

    for _ in range(10):
        print(optim.step(closure))

    print(f"Converged to temp={temp.item()} bias={bias.item()}.")

    # make temperature calibrated model
    tc_layer = nn.Conv2d(1, 1, 1)
    tc_layer.weight.data[0, 0, 0, 0] = temp.data
    tc_layer.bias.data[0] = bias.data

    class TCModel(nn.Module):
        def __init__(self, model, tc_layer):
            super().__init__()
            self.model = model
            self.tc_layer = tc_layer

        def forward(self, x, *args, **kwargs):
            x = self.model(x, *args, **kwargs)
            x = self.tc_layer(x)
            return x

    tc_model = TCModel(model, tc_layer).cuda()
    # save the temperature calibrated model
    torch.save(tc_model.state_dict(), os.path.join(args.output_dir, "tc_model.pth"))

    # ======= EVALUATION ========
    metrics = {}

    val_metrics, val_tracker = run_epoch(
        val, tc_model, loss_fn, None, None, None, device=args.device, train=False
    )
    metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

    # full predictions dataframe
    core_ids, predictions = val_tracker.get()
    core_info_dict = get_core_info_dict(val.dataset.table)
    df = make_predictions_dataframe(predictions, core_ids, core_info_dict)
    df.to_csv(os.path.join(args.output_dir, "val_predictions.csv"), index=False)

    # saving the rest of the metrics
    for k, v in metrics.items():
        if isinstance(v, wandb.Image): 
            image = v.image
            plt.imshow(np.array(image))
            plt.savefig(os.path.join(args.output_dir, f"{k}.png"))
        else: 
            with open(os.path.join(args.output_dir, f"metrics.txt"), "a") as f:
                f.write(f"{k}: {v}\n")

    # generate heatmaps
    if args.save_heatmaps:
        os.makedirs(os.path.join(args.output_dir, "heatmaps"), exist_ok=True)
        for batch in tqdm(val, desc="Saving heatmaps"):
            img, needle, prostate, label, core_id = prepare_batch(batch, args.device)
            with torch.no_grad():
                logits = tc_model(img)
            probs = logits.sigmoid().detach().cpu().numpy()[0][0]
            img = img.detach().cpu().numpy()[0][0]
            prostate = prostate.detach().cpu().numpy()[0][0]
            needle = needle.detach().cpu().numpy()[0][0]
            metadata_dict = get_core_info_dict(val.dataset.table)
            d = metadata_dict[core_id[0]]
            d["core_id"] = core_id[0]

            render_heatmap_v2(probs, img, prostate, needle, d)

            plt.savefig(
                os.path.join(args.output_dir, "heatmaps", f"{core_id[0]}.png"), dpi=300
            )
            plt.close()

    

def extract_all_pixel_predictions(model: ProstNFound, loader, device):
    pixel_labels = []
    pixel_preds = []
    core_ids = []

    model.eval()
    model.to(device)

    for i, batch in enumerate(tqdm(loader)):
        with torch.no_grad():

            img, needle, prostate, label, core_id = prepare_batch(batch, device)

            # run the model
            heatmap_logits = model(img)

            # compute predictions
            masks = (prostate > 0.5) & (needle > 0.5)

            predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)

            labels = torch.zeros(len(predictions), device=predictions.device)
            for i in range(len(predictions)):
                labels[i] = label[batch_idx[i]]
            pixel_preds.append(predictions.cpu())
            pixel_labels.append(labels.cpu())

            core_ids.extend(core_id[batch_idx[i]] for i in range(len(predictions)))

    pixel_preds = torch.cat(pixel_preds)
    pixel_labels = torch.cat(pixel_labels)

    return pixel_preds, pixel_labels, core_ids


def make_predictions_dataframe(predictions, core_ids, core_info_dict):
    d = []
    for i, (core_id, prediction) in enumerate(zip(core_ids, predictions)):
        d.append(
            {
                "core_id": core_id,
                "prediction": prediction.item(),
                **core_info_dict[core_id],
            }
        )
    return pd.DataFrame(d)


def get_core_info_dict(table):
    table = table.set_index("core_id")
    return table.to_dict(orient="index")


def render_heatmap_v2(heatmap, bmode, prostate_mask, needle_mask, metadata):
    cmap = "gray"

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    extent = (0, 1, 0, 1)

    heatmap_logits = np.flip(heatmap.copy(), axis=0)
    bmode = np.flip(bmode, axis=0)
    prostate_mask = np.flip(prostate_mask, axis=0)
    needle_mask = np.flip(needle_mask, axis=0)

    prostate_mask_for_alpha = resize(
        prostate_mask, (heatmap_logits.shape[0], heatmap_logits.shape[1]), order=0
    )
    # expand the prostate mask
    prostate_mask_for_alpha = dilation(prostate_mask_for_alpha)
    prostate_mask_for_alpha = gaussian(prostate_mask_for_alpha, sigma=3)

    heatmap_logits = gaussian(heatmap_logits, sigma=1)

    ax[1].imshow(bmode, extent=extent, cmap=cmap)

    ax[1].imshow(
        heatmap_logits,
        vmin=0,
        vmax=1,
        extent=extent,
        cmap="jet",
        alpha=prostate_mask_for_alpha * 0.5,
    )
    ax[1].contour(np.flip(prostate_mask, 0), extent=extent, colors="white")
    ax[1].contour(np.flip(needle_mask, 0), extent=extent, colors="white")

    ax[0].imshow(bmode, extent=extent, cmap=cmap)
    ax[0].contour(np.flip(prostate_mask, 0), extent=extent, colors="white")
    ax[0].contour(np.flip(needle_mask, 0), extent=extent, colors="white")

    fig.suptitle(
        f'Core {metadata["core_id"]}; Pathology {metadata["pathology"]}; Involvement {metadata["inv"]}; Center {metadata["center"]}'
    )

    for a in ax:
        a.axis("off")

    fig.tight_layout()


if __name__ == "__main__": 
    main()