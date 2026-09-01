import os
import sys
from datetime import datetime

sys.path.append('core')
import logging

from loss import loss_motion_mask
from train import TrainingArgs, train
from WIFlow.csi_preprocessor import MeanNormalizationPhaConjPreprocessor
from WIFlow.WIFlow_mask import WiFlowMask
from WIFlow.WIFlow_rnn import RNNArgs

os.system("export KMP_INIT_AT_FORK=FALSE")







if __name__ == '__main__':
    model_args = RNNArgs(antenna_count=16,
                         dim=32,
                         #image_size= [256,336],
                         image_size=[128,168],
                         #image_size=[216, 480],
                         preprocessor=MeanNormalizationPhaConjPreprocessor)#MeanNormalizationPhaConjPreprocessor)
    logging.getLogger().setLevel(logging.INFO)
    logging.info(model_args)
    model = WiFlowMask(model_args)


    train_args = TrainingArgs(name=f"{model.__class__.__name__}_{loss_motion_mask.__name__}_{datetime.now():%Y-%m-%d %H:%s}",
                            num_steps=60000,# wdecay=0.01,
                            dataset_config_path="dataset_configs/seemo_config_E9_subject.yml", lr=1e-3)

    train(model, train_args, loss_func=loss_motion_mask)
    print("Done!")
