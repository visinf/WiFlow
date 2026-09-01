
from pathlib import Path

import torch
from mmfi_dataset.mmfi import MMFI_DatasetFrame
from mmfi_dataset.mmfi import collate_fn_padd as collate_fn_padd_original
from mmfi_dataset.modality import MODALITY_MAP, ImageModality, WifiCSIModality


class WifiCSISeemoModality(WifiCSIModality):
    file_ending=".pt"
    def read_frame(self, frame_path: str | Path):
        return torch.load(frame_path).squeeze(dim=0)

MODALITY_MAP["image_blurred"] = ImageModality("image_blurred")


MODALITY_MAP["csi"] = WifiCSISeemoModality("csi")

class MMFI_DatasetPairwise(MMFI_DatasetFrame):
    """returns mmfi pairwise based on frame idx"""
    def load_data(self):
        data_info = []
        for relative_path in self.fragment.create_tree_deterministic():
            if not (self.database.data_root/relative_path).exists():
                continue
            frame_num = len(list((self.database.data_root/relative_path/self.modalities[0].name).glob("*")))
            num_digits = len(next((self.database.data_root/relative_path/self.modalities[0].name).glob("*")).stem.replace("frame", ""))
            form = "0{}d".format(num_digits)
            assert frame_num , f"no data found for modality {self.modalities[0].name} at {relative_path}"
            for idx in range(frame_num):
                data_dict = {
                    'modalities': [m.name for m in self.modalities],
                    'idx': idx
                }
                data_valid = True
                for mod in self.modalities:
                    data_dict[mod.name+'1_path'] =  self.database.data_root/relative_path/mod.name/f"frame{idx+1:{form}}{mod.file_ending}"
                    data_dict[mod.name+'2_path'] =  self.database.data_root/relative_path/mod.name/f"frame{idx+2:{form}}{mod.file_ending}"
                    if not mod.exists(data_dict[mod.name+'1_path']) or not mod.exists(data_dict[mod.name+'2_path']) :
                        data_valid = False
                if data_valid:
                    data_info.append(data_dict)

        return data_info
    def __getitem__(self, idx) -> tuple[dict]:
        item = self.data_list[idx]
        sample_shared = {'modalities': item['modalities'],
                    'idx': item['idx'],
                    }
        sample1, sample2 = {}, {}

        for mod in self.modalities:
            data_path1 = item[mod.name + '1_path']
            data_mod1 = mod.read_frame(data_path1)
            sample1[mod.name+'_path'] = item[mod.name+'1_path']
            sample1[mod.name] = data_mod1
            if mod == MODALITY_MAP["flow"]: # do not load flow for second
                continue
            if mod == MODALITY_MAP["csi"]: # do not load flow for second
                pass
            data_path2 = item[mod.name + '2_path']
            data_mod2 = mod.read_frame(data_path2)
            sample2[mod.name+'_path'] = item[mod.name+'2_path']
            sample2[mod.name] = data_mod2

        return sample_shared|sample1 , sample_shared|sample2
    def __len__(self):
        # every fast frame per scene does not have a valid following pair.
        return super().__len__()

def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''
    X1,X2 = zip(*batch)
    return collate_fn_padd_original(X1), collate_fn_padd_original(X2)
