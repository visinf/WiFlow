import torch
from torch import Tensor

from WIFlow.WIFlow_rnn import RNNArgs, WiFlowRNN


class WIFlowAllWhite(WiFlowRNN):
    """Mock RNN that outputs all white flow. Used for testing the rest of the pipeline and evaluation."""
    def __init__(self, args:RNNArgs):
        super().__init__(args)
    def forward(self, csi1:Tensor, csi2:Tensor, iters=None):
        N = csi1.shape[0]
        flow = torch.zeros((N,2,*self.args.image_size ), device=csi1.device)
        return {"final":flow,"flow":[flow]} # to fullfill the raft forward
