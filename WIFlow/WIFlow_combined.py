import torch

from WIFlow.csi_preprocessor import CSIQuotientPreprocessor
from WIFlow.WIFlow_mask import WiFlowMask
from WIFlow.WIFlow_rnn import RNNArgs, WiFlowRNN


def load_ckpt(model, path):
    """ Load checkpoint """
    state_dict = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(state_dict, strict=False)


class WiFlowCombined(torch.nn.Module):
    def __init__(self, args:RNNArgs):
        super(WiFlowCombined, self).__init__()
        self.flow_model = WiFlowRNN(args)
        self.mask_model = WiFlowMask(args)
        self.args = args
        self.preprocessor = self.flow_model.preprocessor
    def forward(self, csi1, csi2):
        mask = self.mask_model(csi1, csi2)
        noisy_flow = self.flow_model(csi1, csi2)
        return {"final": noisy_flow["final"]*mask["mask"][-1], "flow": noisy_flow["flow"], "mask": mask["mask"]}



if __name__ == "__main__":
    import re
    import shutil
    from pathlib import Path

    import yaml

    from train_roi import mask_checkpoints

    def _load_dataset_config(ckpt_path: Path):
        cfg_path = ckpt_path / "dataset_config.yaml"
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    def load_models_into_combined(model: WiFlowCombined, flow_ckpt_path: Path, mask_ckpt_path: Path):


        assert _load_dataset_config(flow_ckpt_path) == _load_dataset_config(mask_ckpt_path), (
            "flow_model and mask_model must be trained on the same dataset_config.yaml"
        )

        load_ckpt(model.flow_model, flow_ckpt_path/"model.pth")
        load_ckpt(model.mask_model, mask_ckpt_path/"model.pth")
    model_args = RNNArgs(antenna_count=16,
                        dim=32,#32,
                        image_size= [128,168],
                        #image_size=[216, 480],
                        preprocessor=CSIQuotientPreprocessor,
                        align_output=3)
    def extract_id_from_ckpt_path(ckpt_path: Path):
        m = re.search(r":(\d{10})_", ckpt_path.name)
        if m:
            return m.group(1)
        else:
            raise ValueError(f"Could not extract id from checkpoint path: {ckpt_path}")

    model = WiFlowCombined(model_args)
    mask_ckpt_path = Path(mask_checkpoints["sideview_time"])
    flow_ckpt_path = Path("runs/x_Quotient_rnn_60k/sideview/WiFlowRNN_sequence_loss_amplified_2026-02-25 09:1772012261_time_CSIQuotientPreprocessor")
    environment = _load_dataset_config(mask_ckpt_path)["train"]["environments"][0]
    load_models_into_combined(model, flow_ckpt_path, mask_ckpt_path)
    output_dir = Path(f"runs/WiFlowCombined_{environment}_flow_{extract_id_from_ckpt_path(flow_ckpt_path)}_mask_{extract_id_from_ckpt_path(mask_ckpt_path)}")
    state = model.state_dict()
    output_dir.mkdir(parents=False, exist_ok=False)
    torch.save(model.state_dict(), output_dir / "model.pth")
    shutil.copy(flow_ckpt_path / "dataset_config.yaml", output_dir / "dataset_config.yaml")
    shutil.copy(flow_ckpt_path / "model_params.yaml", output_dir / "model_params.yaml")
    print("Saved combined model to ", output_dir / "model.pth")
