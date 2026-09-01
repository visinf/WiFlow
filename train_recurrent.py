import os
import sys
from datetime import datetime

sys.path.append('core')
import logging

import numpy as np
import torch

import loss
from train import TrainingArgs, train
from WIFlow.csi_preprocessor import CSIQuotientPreprocessor
from WIFlow.WIFlow_rnn import RNNArgs, WIFlow

os.system("export KMP_INIT_AT_FORK=FALSE")



if __name__ == '__main__':
    model_args = RNNArgs(antenna_count=16,
                         dim=32,#32,
                         image_size= [128,168],
                         #iters=4,
                         #image_size=[216, 480],
                         preprocessor=CSIQuotientPreprocessor) # CSIQuotientAntennaReduction1Preprocessor CSIQuotientPreprocessor # SingleAntennaCSIQuotientPreprocessor
    logging.getLogger().setLevel(logging.INFO)
    logging.info(model_args)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WIFlow(model_args)


    train_args = TrainingArgs(name=f"{model.__class__.__name__}_{loss.sequence_loss_amplified.__name__}_{datetime.now():%Y-%m-%d %H:%s}",
                            num_steps=60000,# wdecay=0.01,
                            dataset_config_path="dataset_configs/seemo_config_E9_time.yml", lr=1e-3)

    train(model, train_args, loss_func=loss.sequence_loss_amplified)
    print("Done!")
