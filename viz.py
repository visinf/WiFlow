
import matplotlib.pyplot as plt
import torch
from torchvision.utils import flow_to_image

plt.style.use("ggplot")
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

def my_flow_to_image(flow:torch.Tensor,normalization:float=12.)->torch.Tensor:
    flow_cp = torch.clone(flow)
    flow_cp[0,0,0,0]=-normalization
    flow_cp[0,1,0,0]=-normalization
    flow_cp[0,0,-1,-1]=normalization
    flow_cp[0,1,-1,-1]=normalization
    img = flow_to_image(flow_cp)
    img[0,:,0,0]=255
    img[0,:,-1,-1]=255
    return img

def flow_to_mask(flow:torch.Tensor, amplitude=1.)-> torch.Tensor:
    return ((flow**2).sum(dim=1, keepdim=True).sqrt() > amplitude).to(torch.uint8)








