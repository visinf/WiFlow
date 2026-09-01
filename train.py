import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from WIFlow.WIFlow_classifier import CSIClassifier
from WIFlow.WIFlow_regressor import CSIRegressor

from evaluation import evaluate
from viz import my_flow_to_image
from WIFlow.dataset import collate_fn_padd

sys.path.append('core')
import argparse
import logging

import numpy as np
import torch
import torch.optim as optim
from mmfi_dataset.decode_config import MMFIConfig

# from sea_raft.ddp_utils import *
from mmfi_dataset.mmfi import MMFi_Database
from pydantic import BaseModel
from sea_raft.utils.utils import load_ckpt
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize
from tqdm import tqdm
from WIFlow.WIFlow_transformer import TransformerArgs, WIFlowTransformer

from loss import (
    loss_epe,
    sequence_loss,
    sequence_loss_filtered,
)
from WIFlow import MMFI_DatasetPairwise, WIFlow

os.system("export KMP_INIT_AT_FORK=FALSE")


class TrainingArgs(BaseModel):
    name: str
    gamma:float = 0.85
    wdecay:float = 1e-5
    lr:float = 1e-5
    epsilon:float = 1e-8
    num_steps:int
    clip:Optional[float] = None#0
    restore_ckpt:Optional[str] = None
    validation_freq:int = 200
    checkpoint_freq:int = 1000
    dataset_config_path:str='dataset_configs/mmfi_config_mini.yml'

    @property
    def restore_ckpt_path(self) -> Optional[Path]:
        return Path(self.restore_ckpt) if self.restore_ckpt else None
def json_to_args(json_path):
    # return a argparse.Namespace object
    with open(json_path, 'r') as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args

def fetch_optimizer(args:TrainingArgs, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler



def prepare_dataloader_random(dataset_config: MMFIConfig, split_file:Path, num_workers=10, shuffle_val:bool=False):
    database = MMFi_Database(dataset_config.dataset_root)

    # --- Build a dataset that contains both train + validation data ---
    train_dataset_raw = MMFI_DatasetPairwise(database, dataset_config.modalities, dataset_config.train)
    val_dataset_raw   = MMFI_DatasetPairwise(database, dataset_config.modalities, dataset_config.validation)
    test_dataset  = MMFI_DatasetPairwise(database, dataset_config.modalities, dataset_config.test)

    full_dataset = ConcatDataset([train_dataset_raw, val_dataset_raw])

    #  Compute split sizes (80% train / 20% val)
    total_size = len(full_dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size




    if split_file is not None and os.path.exists(split_file):
        # --- Load pre-saved split from JSON ---
        with open(split_file, "r") as f:
            split_data = json.load(f)
        train_indices = split_data["train_indices"]
        val_indices   = split_data["val_indices"]
    else:
        # --- Generate new split ---
        rng_generator = torch.Generator().manual_seed(dataset_config.seed)
        perm = torch.randperm(total_size, generator=rng_generator).tolist()
        train_indices = perm[:train_size]
        val_indices   = perm[train_size:]

        if split_file is not None:
            with open(split_file, "w") as f:
                json.dump(
                    {
                        "train_indices": train_indices,
                        "val_indices": val_indices,
                        "seed": dataset_config.seed,
                        "total_size": total_size,
                    },
                    f,
                    indent=2
                )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_size=dataset_config.train.batch_size,
        collate_fn=collate_fn_padd,
        shuffle=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        num_workers=num_workers,
        batch_size=dataset_config.validation.batch_size,
        collate_fn=collate_fn_padd,
        shuffle=shuffle_val,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=num_workers,
        batch_size=dataset_config.test.batch_size,
        collate_fn=collate_fn_padd,
        shuffle=shuffle_val,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def prepare_dataloader(dataset_config:MMFIConfig, num_workers=4, shuffle_val:bool=False, training:bool=False)-> tuple[DataLoader,DataLoader,DataLoader]:
    database = MMFi_Database(dataset_config.dataset_root)
    train_dataset = MMFI_DatasetPairwise(database,dataset_config.modalities,dataset_config.train)
    val_dataset  = MMFI_DatasetPairwise(database,dataset_config.modalities,dataset_config.validation)
    if not training:
        test_dataset  = MMFI_DatasetPairwise(database,dataset_config.modalities,dataset_config.test)
    else: # for evaluation
        test_dataset  = MMFI_DatasetPairwise(database,dataset_config.modalities+ ["class"],dataset_config.test)

    rng_generator = torch.manual_seed(dataset_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config.train.batch_size,
        collate_fn=collate_fn_padd,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        generator=rng_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataset_config.validation.batch_size,
        collate_fn=collate_fn_padd,
        num_workers=num_workers,
        shuffle=shuffle_val,
        drop_last=False,
        generator=rng_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=dataset_config.test.batch_size,
        collate_fn=collate_fn_padd,
        num_workers=num_workers,
        shuffle=shuffle_val,
        drop_last=False,
        generator=rng_generator,
    )
    return train_loader, val_loader, test_loader





def merge_visualization(image:torch.Tensor, gt_flow:torch.Tensor, prediction_flow:torch.Tensor, mask:Optional[torch.Tensor]=None)-> torch.Tensor:
    # Define target size using the original image shape

    _, _, H, W = gt_flow.shape
    resize = Resize((H, W))

    # Get and prepare tensors
    img = image[0]  # [C, H, W]
    flow_gt = my_flow_to_image(gt_flow.float())[0]      # [C, H', W']
    flow_output = my_flow_to_image(prediction_flow)[0]     # [C, H'', W'']

    # Resize flows to match the image size
    img = resize(img)
    flow_output = resize(flow_output)
    if mask is not None:
        mask_resized = (resize(mask[0,:1,:,:]).repeat(3,1,1).to(img.device)*255.0).to(torch.uint8)
        merged = torch.cat([img, flow_gt, flow_output, mask_resized], dim=2)[None]
    else:
        merged = torch.cat([img, flow_gt, flow_output], dim=2)[None]
    return merged

def validate(model:WIFlow, loader:DataLoader, device:str, writer:SummaryWriter=None, total_steps:int=None, max_count=200, loss_fun=sequence_loss):
    model.eval()
    metrics = defaultdict(list)
    losses , epes = [], []
    i_val = 0
    images = []

    with torch.no_grad():
        for X1,X2 in tqdm(loader,desc="validation"):
            if max_count and i_val > max_count:
                break
            if isinstance(model,CSIRegressor):
                X1,X2 = CSIRegressor.filter_data_by_dominant(X1,X2)

            if not len(X1["flow"]):
                continue
            csi_key = "csi" if "csi" in X1 else "wifi-csi"  # caused by mmfi vs seemo dataset structure
            img_key = "image" if "image" in X1 else "rgb" # caused by mmfi vs seemo dataset structure
            image1, csi1, csi2, gt_flow = X1[img_key], X1[csi_key], X2[csi_key], X1["flow"]
            image1, csi1, csi2, gt_flow = image1.to(device), csi1.to(device), csi2.to(device), gt_flow.to(device)
            output = model(csi1, csi2)
            if isinstance(model,CSIClassifier):
                gt_indices = model.classes_to_indices(X1["class"]).to(device)
                loss = loss_fun(output, gt_indices)
                losses.append(loss)
            else:
                loss = loss_fun(output, gt_flow)
                losses.append(loss)
                metrics[loss_fun.__name__].append(loss.item())
            metrics["epe"].append(loss_epe(output,gt_flow))

            epes.append(metrics["epe"][-1])
            i_val += 1
            if "mask" in output:
                images.append(merge_visualization(image1[0][None],gt_flow[0][None].float(),output["final"][0][None],output["mask"][-1][0][None]))
            else:
                images.append(merge_visualization(image1[0][None],gt_flow[0][None].float(),output["final"][0][None]))
    mean_loss = torch.mean(torch.Tensor(losses))
    mean_epe = torch.mean(torch.Tensor(epes))
    if writer:

        writer.add_images('valid/image_gt_flow',torch.concat([torch.concat(images[:5],dim=2),torch.concat(images[5:10],dim=2),torch.concat(images[10:15],dim=2),torch.concat(images[-5:],dim=2)], dim=3),total_steps)
        writer.add_scalar('valid/Loss/loss', mean_loss, total_steps)
        writer.add_scalar('valid/Loss/epe', mean_epe, total_steps)
    model.train()
    return metrics


def train(model:WIFlowTransformer, train_args:TrainingArgs, loss_func=sequence_loss_filtered):
    """ Full training loop """

    try:
        suffix=f"_{Path(train_args.dataset_config_path).stem.split('_')[-1]}_{model.args.preprocessor.__name__}"
    except Exception:
        suffix= ""
    PATH = Path(f'runs/{train_args.name}{suffix}')
    if not PATH.parent.exists():
        os.mkdir(PATH.parent)

    if not PATH.exists():
        os.mkdir(PATH)
        shutil.copy(train_args.dataset_config_path, PATH/"dataset_config.yaml")
        model.args.preprocessor = model.preprocessor.__class__.__name__
        if isinstance(model.args, argparse.Namespace):
            (PATH/"model_params.yaml").write_text(json.dumps(vars(model.args), indent=4))
        else:
            (PATH/"model_params.yaml").write_text(model.args.model_dump_json(indent=2))


    writer = SummaryWriter(log_dir=PATH)
    writer.add_text("train/config",json.dumps(vars(train_args), indent=2))
    device_id = torch.device('cuda')
    model.to(device_id)
    if train_args.restore_ckpt is not None:
        load_ckpt(model, train_args.restore_ckpt)
        print(f"restore ckpt from {train_args.restore_ckpt}")
    logging.info(f"created model {model.__class__.__name__}")
    model.train()
    dataset_config = MMFIConfig.load(train_args.dataset_config_path)
    torch.manual_seed(dataset_config.seed)
    np.random.seed(dataset_config.seed)
    if dataset_config.random:
        train_loader, val_loader,test_loader = prepare_dataloader_random(dataset_config,PATH/"splits.json", shuffle_val=False)
    else:
        train_loader, val_loader, test_loader = prepare_dataloader(dataset_config, shuffle_val=False, training=True)
    writer.add_text("train/config/data",dataset_config.model_dump_json(indent=2))
    logging.info("prepared train data")
    optimizer, scheduler = fetch_optimizer(train_args, model)
    total_steps = 0
    epoch = 0
    should_keep_training = True
    # torch.autograd.set_detect_anomaly(True)

    losses, epes = [],[]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_args.num_steps)

    while should_keep_training:
        # shuffle sampler
        # train_loader.sampler.set_epoch(epoch)
        epoch += 1
        logging.info(f"EPOCH: {epoch}")
        process_bar = tqdm(train_loader,desc="training")
        for X1,X2 in process_bar:
            if (X1["flow"]**2).sum(dim=1).sqrt().max() > 40.:
                print(f'skipping batch because of to high motion inside {(X1["flow"]**2).sum(dim=1).sqrt().max()} at {X1["flow_path"]}')
                continue
            if isinstance(model,CSIRegressor):
                X1,X2 = CSIRegressor.filter_data_by_dominant(X1,X2)
            if not len(X1["flow"]):
                continue
            csi_key = "csi" if "csi" in X1 else "wifi-csi"  # caused by mmfi vs seemo dataset structure
            img_key = "image" if "image" in X1 else "rgb" # caused by mmfi vs seemo dataset structure
            process_bar.set_description(f"{model.__class__.__name__}_{train_args.name[-3:]}:{total_steps}/{train_args.num_steps}")
            optimizer.zero_grad()
            image1, csi1, csi2, gt_flow = X1[img_key], X1[csi_key], X2[csi_key], X1["flow"]
            image1, csi1, csi2, gt_flow = image1.to(device_id), csi1.to(device_id), csi2.to(device_id), gt_flow.to(device_id)
            output = model(csi1, csi2)
            if isinstance(model,CSIClassifier):
                gt_indices = model.classes_to_indices(X1["class"]).to(device_id)
                loss = loss_func(output, gt_indices)
            else:
                loss = loss_func(output, gt_flow)
            with torch.no_grad():
                epe = loss_epe(output,gt_flow)
            loss.backward()
            losses.append(loss.to("cpu").detach().numpy())
            epes.append(epe.to("cpu").detach().numpy())
            if train_args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.clip)
            optimizer.step()
            scheduler.step()

            if not total_steps % train_args.checkpoint_freq:
                checkpoint_path = PATH/f"model_{total_steps}.pth"
                torch.save(model.state_dict(), checkpoint_path)
            if not total_steps % train_args.validation_freq:
                #loss_function = lambda output, gt :loss_func(output,gt,train_args.gamma)
                validate(model, val_loader,
                                device_id,loss_fun=loss_func,
                                writer=writer,total_steps=total_steps)

                writer.add_scalar('train/Loss/loss', np.mean(losses), total_steps)
                writer.add_scalar('train/Loss/epe',np.mean(epes), total_steps)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], total_steps)

                if "mask" in output:
                    images = merge_visualization(image1[0][None],gt_flow[0][None].float(),output["final"][0][None],output["mask"][-1][0][None])
                else:
                    images = merge_visualization(image1[0][None],gt_flow[0][None].float(),output["final"][0][None])
                writer.add_images('train/image_gt_flow',images,total_steps)


                losses, epes = [],[]
            if total_steps > train_args.num_steps:
                should_keep_training = False
                break

            total_steps += 1


    torch.save(model.state_dict(), PATH/"model.pth")
    results = evaluate(model,test_loader,device_id)
    json.dump(results, open(PATH/f"{PATH.name}_metrics.json","w"))

if __name__ == '__main__':
    json_path = "WiRaft_model_config.json.json"
    # train_args = json_to_args(json_path)
    logging.getLogger().setLevel(logging.INFO)
    # logging.info(train_args)
    model_args = TransformerArgs()
    model = WIFlowTransformer(model_args)
    train_args = TrainingArgs(name=f"{model.__class__.__name__}_{datetime.now():%Y-%m-%d %H:%M}",
                              num_steps=20000)
    train(model, train_args)
    print("Done!")
