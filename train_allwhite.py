import os
import sys
from datetime import datetime

sys.path.append('core')
import logging

import numpy as np
import torch

from loss import sequence_loss_amplified
from train import TrainingArgs, train
from WIFlow.csi_preprocessor import CSIPreprocessor_dtdc
from WIFlow.WIFlow_allwhite import RNNArgs, WIFlowAllWhite

os.system("export KMP_INIT_AT_FORK=FALSE")



if __name__ == '__main__':
    model_args = RNNArgs(antenna_count=16,
                         dim=32,#32,
                         image_size= [128,168],
                         #image_size=[216, 480],
                         preprocessor=CSIPreprocessor_dtdc)
    logging.getLogger().setLevel(logging.INFO)
    logging.info(model_args)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WIFlowAllWhite(model_args)


    train_args = TrainingArgs(name=f"{model.__class__.__name__}_{sequence_loss_amplified.__name__}_{datetime.now():%Y-%m-%d %H:%s}",
                            num_steps=100,# wdecay=0.01,
                            dataset_config_path="dataset_configs/seemo_config_E9_time.yml", lr=1e-4)

    train(model, train_args, loss_func=sequence_loss_amplified)
    print("Done!")
