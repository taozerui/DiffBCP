## Overview

Code for the paper [[arXiv]](https://arxiv.org/abs/2606.03212):

**Bayesian Tensor Decomposition with Diffusion Model Prior (ICML-26)**

by Zerui Tao and Qibin Zhao.

## Dependencies

The code is developed on Python 3.10, CUDA 12.6, PyTorch 2.5.

`pip install -r requirements.txt`

### Pre-trained models

We use the same pre-trained diffusion checkpoints as
[PnP-DM](https://github.com/zihuiwu/PnP-DM-public). Download them and place
them under `./model/`:

- **FFHQ (256×256, color)** → `./model/ffhq_10m.pt`.
  Download from the FFHQ checkpoint link provided by PnP-DM
  ([Google Drive](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing)).
- **ImageNet (256×256, unconditional)** → `./model/256x256_diffusion_uncond.pt`.
  This is the standard `256x256_diffusion_uncond.pt` checkpoint from OpenAI
  [guided-diffusion](https://github.com/openai/guided-diffusion), also used by PnP-DM.

The model paths are set in `configs/model/edm_unet_adm_dps_ffhq.yaml` and
`configs/model/edm_unet_adm_dps_imagenet.yaml`. Adjust them if you store the
checkpoints elsewhere.

### Datasets

**Face / natural images.** We use FFHQ and ImageNet. Build the 128-image
benchmark arrays with:

`python generate_data.py`

This reads the raw FFHQ / ImageNet images and writes `./data/ffhq_128.npy` and
`./data/imagenet_128.npy`.

**High-resolution images.** For the high-resolution (2048×2048) experiments we use
the same images as [PuTT](https://github.com/sebulo/PuTT) (see their paper
[[arXiv]](https://arxiv.org/abs/2406.04332) and
[project page](https://sebulo.github.io/PuTT_website/)). Download the original
images from the sources below, then resample them to 2048×2048 using PuTT's
[`get_data.py`](https://github.com/sebulo/PuTT/blob/main/PuTT/get_data.py)
script, and place the results under `./data/` with the file names referenced in
`configs/data/*.yaml`:

| Config | Source | Expected file |
| --- | --- | --- |
| `configs/data/marseille.yaml`   | [Pexels — aerial view of Marseille](https://www.pexels.com/photo/aerial-drone-view-of-urban-buildings-from-top-18644280/) | `./data/marseille_2048.jpg` |
| `configs/data/tokyo.yaml`       | [Flickr — gigapixel panorama of Shibuya, Tokyo](https://www.flickr.com/photos/trevor_dobson_inefekt69/29314390837) | `./data/tokyo_2048.png` |
| `configs/data/westerlund.yaml`  | [NASA/Hubble — Westerlund 2](https://science.nasa.gov/asset/hubble/westerlund-2/) | `./data/westerlund_2048.png` |


## Running experiments

- `bash run_ffhq.sh` / `bash run_imagenet.sh` — standard 256×256 inpainting/denoising.
- `bash run_marseille.sh` — high-resolution (2048×2048) inpainting.

Outputs and metrics are written to `./outputs/` and `./results/`.

## Acknowledgements

Our codebase is implemented based on the following projects. Thanks for their contributions.

- PnP-DM: https://github.com/zihuiwu/PnP-DM-public
- PuTT: https://github.com/sebulo/PuTT

## Citation

```
@inproceedings{tao2026bayesian,
    title={Bayesian Tensor Decomposition with Diffusion Model Prior},
    author={Zerui Tao and Qibin Zhao},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=q806xA8NPR}
}
```
