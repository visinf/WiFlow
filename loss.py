
import torch

from viz import flow_to_mask


def loss_epe(output:dict, gt:torch.Tensor, amplifier=2)-> dict:
    epe = torch.sum((output["final"] - gt)**amplifier, dim=1).sqrt()
    return epe.view(-1).mean()

def loss_epe_amplified(output:dict, gt:torch.Tensor)-> dict:
    return loss_epe(output, gt, amplifier=4)


def loss_epe_gt_valid(output:dict, gt:torch.Tensor,amplifier=2)-> dict:
    mask = (gt.abs().sum(dim=1, keepdim=True) >= 0.5)
    epe = torch.sum((output["final"] - gt)**amplifier, dim=1,keepdim=True).sqrt()
    return epe[mask].view(-1).mean() if mask.any() else torch.tensor([0.0],device=gt.device)


def loss_epe_gt_static(output:dict, gt:torch.Tensor,amplifier=2)-> dict:
    """ Only consider pixels where there is no motion in the ground truth, to evaluate how well the model can predict static areas """
    mask = (gt.abs().sum(dim=1, keepdim=True) < 0.5)
    epe = torch.sum((output["final"] - gt)**amplifier, dim=1,keepdim=True).sqrt()
    return epe[mask].view(-1).mean() if mask.any() else torch.tensor([0.0],device=gt.device)



def loss_epe_gt_valid_amplified(output:dict, gt:torch.Tensor)-> dict:
    return loss_epe_gt_valid(output, gt, amplifier=4)



def sequence_loss(output:dict, flow_gt:torch.Tensor, gamma=0.8)-> tuple[float]:
    """ Loss function defined over sequence of flow predictions, ispired by SEA-RAFT without vaild map """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    interpolate_flow(output["final"])
    {interpolate_flow(f) for f in output["flow"]}
    # exlude invalid pixels and extremely large diplacements
    for i, flow in enumerate(output['flow']):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow - flow_gt).abs()
        flow_loss += i_weight * (i_loss.mean())

    flow_loss =flow_loss

    return flow_loss


def sequence_loss_amplified(output:dict, flow_gt:torch.Tensor, gamma=0.8)-> tuple[float]:
    """ Loss function defined over sequence of flow predictions, ispired by SEA-RAFT without vaild map """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    interpolate_flow(output["final"])
    {interpolate_flow(f) for f in output["flow"]}
    # exlude invalid pixels and extremely large diplacements
    for i, flow in enumerate(output['flow']):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow - flow_gt)**4
        flow_loss += i_weight * (i_loss.mean())

    flow_loss =flow_loss

    return flow_loss

def sequence_loss_amplified_masked(output:dict, flow_gt:torch.Tensor, gamma=0.8)-> tuple[float]:
    """ Loss function defined over sequence of flow predictions, ispired by SEA-RAFT without vaild map """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    interpolate_flow(output["final"])
    {interpolate_flow(f) for f in output["flow"]}
    # exlude invalid pixels and extremely large diplacements
    for i, flow in enumerate(output['flow']):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow - flow_gt)**4
        flow_loss += i_weight * (i_loss.mean())

    mask_loss = ((flow_to_mask(output["final"]) - flow_to_mask(flow_gt))*2).sum()
    flow_loss =flow_loss + mask_loss

    return flow_loss


def interpolate_flow(flow:torch.Tensor)->torch.Tensor:
    return flow
    # if flow.isinf().any(): # this made the training very slow, so only use it for debugging
    #     logging.warning(f"noticed some inf values in flow output {flow.isinf().sum()}")
    #     flow[flow == torch.inf] = 0
    # if flow.isnan().any():
    #     logging.warning(f"noticed some nan values in flow output {flow.isnan().sum()}")
    #     flow[flow == torch.nan] = 0

def sequence_loss_filtered(output, flow_gt, gamma=0.8)-> tuple[float,dict]:
    """ Loss function defined over sequence of flow predictions, ispired by SEA-RAFT without vaild map """

    n_predictions = len(output['flow'])
    interpolate_flow(output["final"])
    {interpolate_flow(f) for f in output["flow"]}
    valid = [flow_gt != 0]
    if not valid[0].any():
        return sequence_loss(output, flow_gt, gamma)

    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    for i, flow in enumerate(output['flow']):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow[valid] - flow_gt[valid]).abs()
        flow_loss += i_weight * (i_loss.mean())

    flow_loss =flow_loss



    return flow_loss


def loss_motion_mask(output:torch.Tensor, gt:torch.Tensor)->torch.Tensor:
    """
    Compute the sequence loss with amplification.
    This function amplifies the loss for certain time steps to focus learning.
    """
    mask = flow_to_mask(gt, amplitude=0.5).float()
    loss = ((output["mask"][-1][:,:1,:,:]- mask)**2).mean()
    return loss.float()



def sequence_loss_union_amplified(output, flow_gt, gamma=0.8, amplifier=2)-> tuple[float,dict]:
    """ Loss function defined over sequence of flow predictions, ispired by SEA-RAFT without vaild map """

    n_predictions = len(output['flow'])
    mask_gt = (flow_gt.abs().sum(dim=1, keepdim=True) >= 0.5)
    mask_pred = (output["final"].abs().sum(dim=1, keepdim=True) >= 0.5)
    mask = mask_gt | mask_pred
    mask = mask.repeat(1,2,1,1) # repeat for u and v flow
    if not mask.any():
        return sequence_loss(output, flow_gt, gamma)

    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    for i, flow in enumerate(output['flow']):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow[mask] - flow_gt[mask])**amplifier
        flow_loss += i_weight * (i_loss.mean())


    return flow_loss

def sequence_loss_union_amplified4(output, flow_gt, gamma=0.8)-> tuple[float,dict]:
    return sequence_loss_union_amplified(output, flow_gt, gamma, amplifier=4)


