import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append('core')
import logging
import tempfile

import numpy as np
import torch
import yaml

from loss import loss_epe_amplified, sequence_loss_amplified
from train import TrainingArgs, train
from train_mask import loss_motion_mask
from WIFlow.csi_preprocessor import CSIQuotientPreprocessor
from WIFlow.WIFlow_mask import RCNNArgsMask, WiFlowRoIMasked

os.system("export KMP_INIT_AT_FORK=FALSE")


def loss_motion_mask_combined(output:torch.Tensor, gt:torch.Tensor, alpha=0.1)->torch.Tensor:
    """
    Compute the sequence loss with amplification.
    This function amplifies the loss for certain time steps to focus learning.
    """
    loss = (1 - alpha) * loss_motion_mask(output, gt) + alpha * sequence_loss_amplified(output, gt)
    return loss.float()

def loss_motion_mask_combined_equal(output:torch.Tensor, gt:torch.Tensor)->torch.Tensor:
    """
    Compute the sequence loss with amplification.
    This function amplifies the loss for certain time steps to focus learning.
    """
    return loss_motion_mask_combined(output, gt, alpha=0.5)

def loss_motion_mask_combined_alpha07(output:torch.Tensor, gt:torch.Tensor)->torch.Tensor:
    """
    Compute the sequence loss with amplification.
    This function amplifies the loss for certain time steps to focus learning.
    """
    return loss_motion_mask_combined(output, gt, alpha=0.7)

def loss_motion_mask_combined_epe_amplified(output:torch.Tensor, gt:torch.Tensor, alpha=0.1)->torch.Tensor:
    """
    Compute the sequence loss with amplification.
    This function amplifies the loss for certain time steps to focus learning.
    """
    loss = (1 - alpha) * loss_motion_mask(output, gt) + alpha * loss_epe_amplified(output, gt)
    return loss.float()

def loss_motion_mask_combined_epe_amplified_alpha07(output:torch.Tensor, gt:torch.Tensor)->torch.Tensor:
    return loss_motion_mask_combined_epe_amplified(output, gt, alpha=0.7)

    #Quotient Preprocessor
# mask_checkpoints = {
#     "sideview_time":"runs/x_mask_Quotient_60k/sideview/WiFlowMask_loss_motion_mask_2026-02-17 23:1771366598_time_CSIQuotientPreprocessor",
#     "sideview_subject":"runs/x_mask_Quotient_60k/sideview/WiFlowMask_loss_motion_mask_2026-02-17 23:1771366579_subject_CSIQuotientPreprocessor",

#     "birdview_time":"runs/x_mask_Quotient_60k/birdview/WiFlowMask_loss_motion_mask_2026-02-17 23:1771366499_time_CSIQuotientPreprocessor",
#     "birdview_subject":"runs/x_mask_Quotient_60k/birdview/WiFlowMask_loss_motion_mask_2026-02-17 23:1771366518_subject_CSIQuotientPreprocessor",

#     "birdviewplus_time":"runs/x_mask_Quotient_60k/birdviewplus/WiFlowMask_loss_motion_mask_2026-02-17 23:1771366475_time_CSIQuotientPreprocessor",
#     "birdviewplus_subject":"runs/x_mask_Quotient_60k/birdviewplus/WiFlowMask_loss_motion_mask_2026-02-17 23:1771366400_subject_CSIQuotientPreprocessor"
# }
mask_checkpoints = {
    "sideview_time"         :"runs/x_Quotient_mask_60k/sideview/WiFlowMask_loss_motion_mask_2026-02-25 07:1772000857_time_CSIQuotientPreprocessor",
    "sideview_subject"      :"runs/x_Quotient_mask_60k/sideview/WiFlowMask_loss_motion_mask_2026-02-25 07:1772000844_subject_CSIQuotientPreprocessor",

    "birdview_time"         :"runs/x_Quotient_mask_60k/birdview/WiFlowMask_loss_motion_mask_2026-02-25 07:1772000829_time_CSIQuotientPreprocessor",
    "birdview_subject"      :"runs/x_Quotient_mask_60k/birdview/WiFlowMask_loss_motion_mask_2026-02-25 07:1772000816_subject_CSIQuotientPreprocessor",

    "birdviewplus_time"     :"runs/x_Quotient_mask_60k/birdviewplus/WiFlowMask_loss_motion_mask_2026-02-25 07:1772000801_time_CSIQuotientPreprocessor",
    "birdviewplus_subject"  :"runs/x_Quotient_mask_60k/birdviewplus/WiFlowMask_loss_motion_mask_2026-02-25 07:1772000797_subject_CSIQuotientPreprocessor"
}

if __name__ == '__main__':
    mask_cpkt_path = Path(mask_checkpoints["birdviewplus_subject"])
    dataset_config_path = mask_cpkt_path/"dataset_config.yaml"

    model_args = RCNNArgsMask(antenna_count=16,
                         dim=32,#32,
                         image_size= [128,168],
                         #image_size=[216, 480],
                         preprocessor=CSIQuotientPreprocessor,
                         align_output=3)
    logging.getLogger().setLevel(logging.INFO)
    logging.info(model_args)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    logging.info(f"Set SEED to {seed}")

    # Copy dataset config to temporary file with seed

    # birdviewplus time mask_checkpoint_path = "runs/x_mask_Quotient_60k/birdviewplus/WiFlowMask_loss_motion_mask_2026-02-17 23:1771366475_time_CSIQuotientPreprocessor"





    with open(dataset_config_path, 'r') as f:
        config = yaml.safe_load(f)
    # config['seed'] = seed

    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
    yaml.dump(config, temp_config)
    temp_config.close()

    logging.info(f"Created temporary config file with seed at {temp_config.name}")

    model = WiFlowRoIMasked(model_args)

    loss_func = loss_motion_mask_combined_alpha07
    train_args = TrainingArgs(name=f"{model.__class__.__name__}_{loss_func.__name__}_{datetime.now():%Y-%m-%d %H:%s}",
                            num_steps=60000,# wdecay=0.01,
                            restore_ckpt=str(mask_cpkt_path/"model.pth"), # birdviewplus time
                            dataset_config_path=str(dataset_config_path),#temp_config.name,
                            lr=1e-3)

    train(model, train_args, loss_func=loss_func)
    print("Done!")
