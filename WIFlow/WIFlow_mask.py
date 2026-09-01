from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

#from torchvision.transforms import functional as F
import torch.nn.functional as F
import torch.nn.functional as Fun
from pydantic import Field

#from sea_raft.update import BasicUpdateBlock
from sea_raft.corr import CorrBlock
from torch import Tensor
from torchvision.ops.roi_align import RoIAlign

from WIFlow.WIFlow_rnn import RNNArgs, WiFlowRNN


class WiFlowMask(WiFlowRNN
):
    def __init__(self, args:RNNArgs):
        super().__init__(args)
        self.activation = nn.Sigmoid()
    def forward(self, csi1:Tensor, csi2:Tensor):
        """ Estimate optical flow between pair frames based on  CSI captures """
        N, _, H, W = csi1.shape

        csi1 = self.preprocessor(csi1)
        csi2 = self.preprocessor(csi2)
        csi1 = csi1.contiguous()
        csi2 = csi2.contiguous()
        mask_predictions = []

        N, ANTENNA, CARRIER, TIME = csi1.shape
        H, W = self.args.image_size

        fmap1_8, fmap2_8 = self.fnet(csi1), self.fnet(csi2)
        cross_corr_fn = CorrBlock(fmap1_8, fmap2_8, args=self.args)
        cnet = self.cnet(csi1)

        cnet = F.dropout(cnet, p=self.args.dropout, training=self.training)
        net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)

        double_mask = torch.zeros(N, 2, H//8, W//8).to(csi1.device)
        for itr in range(self.args.iters):

            corr = cross_corr_fn(double_mask)

            #with autocast(enabled=self.args.mixed_precision):
            net, _, delta_mask = self.update_block(net, context, corr, double_mask)

            double_mask=double_mask +delta_mask
            activation=  self.activation(double_mask)[:,:1,:,:]
            mask = Fun.interpolate(activation, scale_factor=8, mode="bilinear")
            mask_predictions.append(mask)

        dummy_flow = Fun.interpolate(double_mask, scale_factor=8, mode="bilinear")
        return {'final': dummy_flow, 'flow': [dummy_flow], "mask": mask_predictions}


class RCNNArgsMask(RNNArgs):
    align_output: int = Field(128//8)


class WiFlowRoIMasked(WiFlowMask
):
    def __init__(self, args:RCNNArgsMask):
        super().__init__(args)
        self.roialign = RoIAlign(output_size=args.align_output, spatial_scale=1.0, sampling_ratio=-1)
        # in_channels = fmap1 + fmap2 + cnet -> Flow
        self.masked_flow_cov = torch.nn.Conv2d(in_channels=(self.args.dim*(2*3)), out_channels=2, kernel_size=1)
        self.mask_thld_percentage = 0.3


    def mask_to_bboxes(self, mask:Tensor)->Optional[Tensor]:
        """ Convert mask to bounding boxes using contour detection """
        thld = self.mask_thld_percentage * 255
        B, _, H, W = mask.shape
        bboxes_res = []
        for i, m in enumerate(mask):
            mask_np = (m.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
            _, binary_mask = cv2.threshold(mask_np, thld, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxes_raw = [cv2.boundingRect(cnt) for cnt in contours]
            bboxes = [torch.Tensor([i, x,y,x + w,y + h])[None] for x,y,w,h in bboxes_raw]
            bboxes_res.extend(bboxes)
        if not bboxes_res:
            return None
        bboxes_res = torch.cat(bboxes_res, dim=0)
        return bboxes_res.to(mask.device)

    def forward(self, csi1:Tensor, csi2:Tensor):
        """ Estimate optical flow between pair frames based on  CSI captures """
        N, _, H, W = csi1.shape

        csi1 = self.preprocessor(csi1)
        csi2 = self.preprocessor(csi2)
        csi1 = csi1.contiguous()
        csi2 = csi2.contiguous()


        N, ANTENNA, CARRIER, TIME = csi1.shape
        H, W = self.args.image_size

        fmap1_8, fmap2_8 = self.fnet(csi1), self.fnet(csi2)
        cross_corr_fn = CorrBlock(fmap1_8, fmap2_8, args=self.args)
        cnet = self.cnet(csi1)

        cnet = F.dropout(cnet, p=self.args.dropout, training=self.training)
        net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)
        mask_predictions = []
        double_mask = torch.zeros(N, 2, H//8, W//8).to(csi1.device)
        for itr in range(self.args.iters):

            corr = cross_corr_fn(double_mask)

            #with autocast(enabled=self.args.mixed_precision):
            net, _, delta_mask = self.update_block(net, context, corr, double_mask)

            double_mask=double_mask +delta_mask
            activation=  self.activation(double_mask)[:,:1,:,:]
            #activation = torch.where(activation < self.mask_thld_percentage, torch.tensor(0.0, device=activation.device), activation)
            mask = Fun.interpolate(activation, scale_factor=8, mode="bilinear")
            mask_predictions.append(mask)


        # RoIAlign from MaskedRCNN
        mask = activation[:,:1,:,:]

        roi_input_features = torch.concat([fmap1_8, fmap2_8, cnet], dim=1)
        roi_bboxes = self.mask_to_bboxes(mask)
        flow_init = torch.zeros(N, 2, H//8, W//8).to(csi1.device)
        if roi_bboxes is None:
            flow_init =Fun.interpolate(flow_init, scale_factor=8, mode="bilinear")
            return {'final': flow_init, 'flow': [flow_init], "mask": mask_predictions}
        roi_outputs = self.roialign(roi_input_features, roi_bboxes)
        assert roi_outputs.shape[0] == roi_bboxes.shape[0], "RoIAlign output shape mismatch. Should be the same as number of boxes."
        roi_flows = self.masked_flow_cov(roi_outputs)

        for bbox, flow in zip(roi_bboxes, roi_flows):
            i, x1, y1, x2, y2 = bbox.int()
            flow_init[i, :, y1:y2, x1:x2] = torch.nn.functional.interpolate(flow.unsqueeze(0), size=(y2-y1,x2-x1), mode='bilinear', align_corners=False)
        flow_8 = torch.where(mask > self.mask_thld_percentage, flow_init, torch.tensor(0.0, device=flow_init.device))
        #flow_8 = flow_init * mask
        flow = Fun.interpolate(flow_8, scale_factor=8, mode="bilinear")
        return {'final': flow, 'flow': [flow], "mask": mask_predictions}


class WiFlowMaskedRCNN(WiFlowRoIMasked):
    def __init__(self, args:RCNNArgsMask):
        print("Deprecated: Use WiFlowRoIMasked instead of WiFlowMaskedRCNN. It got renamed.")
        super().__init__(args)
