
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from mmfi_dataset.decode_config import MMFIConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import (
    loss_epe,
    loss_epe_amplified,
    loss_epe_gt_static,
    loss_epe_gt_valid,
    loss_epe_gt_valid_amplified,
    loss_motion_mask,
)
from WIFlow.WIFlow_allwhite import WIFlowAllWhite
from WIFlow.WIFlow_combined import WiFlowCombined
from WIFlow.WIFlow_mask import WiFlowMask, WiFlowMaskedRCNN
from WIFlow.WIFlow_rnn import RNNArgs, WIFlow, WiFlowRNN

SUPPORTED_MODELS_MAP = {
    "WiFlowRNN": WiFlowRNN,
    "WiFlowCombined": WiFlowCombined,
    "WiFlowMaskedRCNN": WiFlowMaskedRCNN,
    "WIFlowAllWhite": WIFlowAllWhite,
}

loss_funcs = [loss_epe,loss_epe_gt_valid,loss_epe_gt_static,loss_epe_amplified,loss_epe_gt_valid_amplified]


def evaluate(model:WIFlow, loader:DataLoader, device:str,  loss_funs:list[Callable]=loss_funcs):
    model.eval()
    metrics = defaultdict(lambda:defaultdict(list))

    with torch.no_grad():
        for X1,X2 in tqdm(loader,desc="evaluation"):
            csi_key = "csi"
            csi1, csi2, gt_flow = X1[csi_key], X2[csi_key], X1["flow"]
            csi1,csi2, gt_flow = csi1.to(device), csi2.to(device), gt_flow.to(device)
            cls = X1['class'][0]
            output = model(csi1, csi2)
            for loss_fun in loss_funs:
                loss = loss_fun(output, gt_flow)
                metrics[cls][loss_fun.__name__].append(loss.item())

    return metrics

def load_model(model_cls:type[WIFlow], args_cls:type[RNNArgs], path:Path, device="cpu"):

    json_path = path.parent/"model_params.yaml"
    with open(json_path,"r") as file:
        #args = json_to_args(json_path)
        args = args_cls.model_validate_json(file.read())
    eval_model = model_cls(args)

    state_dict = torch.load(path, map_location=device)
    eval_model.load_state_dict(state_dict, strict=False)
    eval_model = eval_model.to(device)
    eval_model.eval()
    return eval_model


if __name__ == "__main__":
    from train import prepare_dataloader, prepare_dataloader_random
    checkpoint_folder = Path("runs/x_Quotient_combined_60k")
    print("Starting evaluation of all checkpoints in folder: ", checkpoint_folder)

    for folder in checkpoint_folder.iterdir():
        print("evaluating folder: ", folder.name)
        if folder.is_dir() and (folder/"model.pth").exists():
            # if any(folder.glob("*metrics.json")):
            #     print("Metrics already exist for this checkpoint, skipping evaluation.")
            #     continue
            checkpoint_path = folder / "model.pth"

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("starting evaluation of ",checkpoint_path.parent.name)
            eval_model =load_model(
                model_cls=SUPPORTED_MODELS_MAP[checkpoint_path.parents[0].name.split("_")[0]],
                args_cls=RNNArgs,
                path=checkpoint_path,
                device=device
            )
            assert checkpoint_path.parents[0].name.split("_")[0] == eval_model.__class__.__name__, "Checkpoint model class does not match the loaded model class"
            config_model =  MMFIConfig.load(checkpoint_path.parent/"dataset_config.yaml")
            config_model.modalities = config_model.modalities + ["class"]
            #config_model.train.batch_size = 1

            if config_model.random:
                print("LOADING RANDOM DATSET")
                train_loader, val_loader,test_loader = prepare_dataloader_random(config_model,checkpoint_path.parent/"splits.json",num_workers=15,shuffle_val=False)
            else:
                train_loader, val_loader,test_loader = prepare_dataloader(config_model,num_workers=15,shuffle_val=False) # has to be zero because it does not render properly with multiple workers
            assert len(test_loader.dataset) > 0
            print("dataset loaded with ", len(test_loader.dataset), " samples")

            if isinstance(eval_model, WiFlowMask):
                loss_funcs += [loss_motion_mask]
            folder.mkdir(exist_ok=True)
            out_path = folder / f"{checkpoint_path.parent.name.replace(' ', '_')}_metrics.json"

            eval_model.to(device)
            results = evaluate(eval_model,test_loader,device, loss_funcs)
            results
            json.dump(results, open(out_path,"w"))
