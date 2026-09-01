# WiFlow: Estimating Optical Flow using WiFi Channel State Information

[WiFlow: Estimating Optical Flow using WiFi Channel State Information](https://visinf.github.io/wiflow)<br/>
Thomas Weigel, Simon Kiefhaber, Fabian Portner, Matthias Hollick, Simone Schaub-Meyer<br/>
TU Darmstadt · TU Delft · hessian.AI<br/>

[[Paper](https://visinf.github.io/wiflow)] [[Dataset](https://visinf.github.io/wiflow)] [[Project Page](https://visinf.github.io/wiflow)]

## Requirements

The code has been tested with Python 3.10 and PyTorch 2.0.

```bash
git clone https://github.com/visinf/wiflow.git
cd wiflow
pip install -e .
```

Or install directly from the Git URL:

```bash
pip install "wiflow @ git+https://github.com/visinf/wiflow.git"
```

## Required Data

> **Note:** The WiFlow dataset will be released publicly soon. Please check the [project page](https://visinf.github.io/wiflow) for updates.

Once available, download the dataset and place it under `dataset/`. The dataset provides three aligned variants — `sideview`, `birdview`, and `birdviewplus` — with synchronized CSI and pseudo ground truth optical flow. Dataset configs are provided in `dataset_configs/`; point the config's `dataset_root` to your download location.

### Pseudo Ground Truth Generation

If you want to generate pseudo-GT flow from your own camera frames, use the notebook `tools/generate_psudo_gt.ipynb`. It runs an ensemble of five optical flow models (rpknet, ms_raft_p, sea_raft_m, memflow, dpflow) via [PTLFlow](https://github.com/hmorimitsu/ptlflow).

## Training

**WiFlowSimple:**
```bash
python train_recurrent.py
```

**WiFlowRoI** (two-step — pretrain mask, then full model):
```bash
python train_mask.py
python train_roi.py
```

**WiFlowCombo:**
```bash
python train_recurrent.py   # train flow branch
python train_mask.py        # train mask branch
# then combine weights via WIFlow/WIFlow_combined.py
```

Training logs and checkpoints are written to `runs/`.

## Evaluation

```bash
python evaluation.py
```

We report EPE (all pixels), EPE<sub>M</sub> (moving pixels), EPE<sub>S</sub> (static pixels), and EPE<sub>A</sub> (amplified, power 4). See the paper for full results.

## Benchmarking

```bash
python benchmarking/main.py
```

## Citation

```bibtex
@inproceedings{weigel2025wiflow,
  title     = {WiFlow: Estimating Optical Flow using WiFi Channel State Information},
  author    = {Weigel, Thomas and Kiefhaber, Simon and Portner, Fabian and Hollick, Matthias and Schaub-Meyer, Simone},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

