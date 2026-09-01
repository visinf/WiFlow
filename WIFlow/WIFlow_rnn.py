
import inspect
from typing import List, Optional

import torch
import torch.nn as nn

#from torchvision.transforms import functional as F
import torch.nn.functional as F
import torch.nn.functional as Fun
from huggingface_hub import PyTorchModelHubMixin
from pydantic import BaseModel, ConfigDict, field_validator

#from sea_raft.update import BasicUpdateBlock
from sea_raft.corr import CorrBlock
from sea_raft.extractor import ResNetFPN
from sea_raft.utils.utils import coords_grid
from torch import Tensor

from WIFlow.csi_preprocessor import CSIPreprocessor, MeanNormalizationPhaConjPreprocessor
from WIFlow.raft_update import BasicUpdateBlock


class RNNArgs(BaseModel):
    model_config = ConfigDict(extra='allow')
    name: str = "seemodataset"
    dataset: str = "seemo"

    use_var: bool = True
    var_min: int = 0
    var_max: int = 10
    pretrain: str = "resnet34"
    initial_dim: int = 64
    block_dims: List[int] = [64, 128, 256]
    corr_radius: int = 4
    corr_levels: int = 4
    dim: int = 128
    num_blocks: int = 2
    iters: int = 4
    image_size: List[int] = [128, 280]
    scale: int = -1
    batch_size: int = 32
    epsilon: float = 1e-6
    dropout: float = 0.2
    clip: float = 1.0
    gamma: float = 0.85

    antenna_count: int = 16
    carrier_count: int = 114

    restore_ckpt: Optional[str] = None
    coarse_config: Optional[str] = None

    preprocessor: type[CSIPreprocessor] = MeanNormalizationPhaConjPreprocessor

    @field_validator('preprocessor', mode="before")
    @classmethod
    def check_is_subclass(cls, v):
        def all_subclasses(cls):
            return set(cls.__subclasses__()).union(
                [s for c in cls.__subclasses__() for s in all_subclasses(c)]
            )
        if not inspect.isclass(v) and not isinstance(v,str):
            raise TypeError(f"{v} is not a class or str.")

        valid_subclasses = {CSIPreprocessor} | all_subclasses(CSIPreprocessor)
        if v in valid_subclasses:
            return v
        try:
            cls = [cls for cls in valid_subclasses if cls.__name__ ==v][0]
            return cls
        except (KeyError,IndexError):
            raise ValueError(f"{v} is not a subclass of {CSIPreprocessor.__name__}")
        raise ValueError(f"{v} is not a subclass of {CSIPreprocessor.__name__}")





class WiFlowRNN(
    nn.Module,
    PyTorchModelHubMixin,
):
    """inspired by SEA-RAFT, implementation of CSIRAFT, one instance of WIFlow
    """
    def __init__(self, args:RNNArgs):
        super().__init__()
        self.args = args

        self.args.corr_channel = args.corr_levels * (args.corr_radius * 2 + 1) ** 2
        self.preprocessor = self.args.preprocessor(self.args.antenna_count,
                                                    self.args.carrier_count)
        input_dims = self.preprocessor.out_channels if hasattr(self.preprocessor,"out_channels") and self.preprocessor.out_channels else args.antenna_count
        self.cnet = nn.Sequential(
                ResNetFPN(args, input_dim=input_dims, output_dim=2*self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=False),
                #ResBlock(3,self.args.dim*2),# TODO make this more complext ///// ResNetFPN(args, input_dim=6, output_dim=2 * self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=True)
                #ResBlock(self.args.dim*2,16),
                nn.Upsample(size=(self.args.image_size[0]//8,self.args.image_size[1]//8), mode='bilinear', align_corners=True))
        self.fnet = nn.Sequential(
                ResNetFPN(args, input_dim=input_dims, output_dim=2*self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=False),
                #ResBlock(3,self.args.dim*2),# TODO make this more complext ///// ResNetFPN(args, input_dim=6, output_dim=2 * self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=True)
                #ResBlock(self.args.dim*2,16),
                nn.Upsample(size=(self.args.image_size[0]//8,self.args.image_size[1]//8), mode='bilinear', align_corners=True))


        if args.iters > 0:
            #self.fnet = ResNetFPN(args, input_dim=3, output_dim=2*self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=True)
            self.update_block = BasicUpdateBlock(args, hidden_dim=args.dim)

    def initialize_flow(self, shape, device):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = shape
        coords0 = coords_grid(N, H, W, device=device)
        coords1 = coords_grid(N, H, W, device=device)

        return coords0, coords1
    @classmethod
    def upsample_flow(cls, flow, mask):
        """From RAFT Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, csi1:Tensor, csi2:Tensor, iters=None):
        """ Estimate optical flow between pair frames based on  CSI captures """
        N, _, H, W = csi1.shape
        if iters is None:
            iters = self.args.iters

        csi1 = self.preprocessor(csi1)
        csi2 = self.preprocessor(csi2)
        csi1 = csi1.contiguous()
        csi2 = csi2.contiguous()
        flow_predictions = []

        N, ANTENNA, CARRIER, TIME = csi1.shape
        H, W = self.args.image_size

        fmap1_8, fmap2_8 = self.fnet(csi1), self.fnet(csi2)
        cross_corr_fn = CorrBlock(fmap1_8, fmap2_8, args=self.args)
        cnet = self.cnet(csi1)

        cnet = F.dropout(cnet, p=self.args.dropout, training=self.training)
        net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)

        coords0, coords1 = self.initialize_flow((N,2,H//8,W//8), csi1.device)

        for itr in range(iters):
            coords1 = coords1.detach()
            if torch.any(coords1[-1].isnan()):
                raise Exception("forward failed, getting nan")
            corr = cross_corr_fn(coords1)
            flow = coords1 - coords0
            #with autocast(enabled=self.args.mixed_precision):
            net, up_mask, delta_flow = self.update_block(net, context, corr, flow)

            coords1 = coords1 + delta_flow

            flow_8 = coords1 - coords0
            flow = Fun.interpolate(flow_8, scale_factor=8, mode="bilinear")
            #flow = self.upsample_flow(flow_8,up_mask)
            flow_predictions.append(flow)

        if torch.any(flow_predictions[-1].isnan()):
            raise Exception("forward failed, getting nan")
        return {'final': flow_predictions[-1], 'flow': flow_predictions,  'nf': None}

WIFlow = WiFlowRNN
