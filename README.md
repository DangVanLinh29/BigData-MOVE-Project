<p align="center">
  <h2 align="center">MOVE: Motion-Guided Few-Shot Video Object Segmentation</h2>
  <p align="center">
    <a><strong>Kaining Ying<sup> * </sup></strong>
    ·
    <a><strong>Hengrui Hu<sup> * </sup></strong></a>
    ·
    <a href=https://henghuiding.com/><strong>Henghui Ding</strong></a><sup> ✉️ </sup>
</p>

<p align="center">
    Fudan University, China
</p>

<p align="center">
    <a href=https://iccv.thecvf.com/>ICCV 2025, Honolulu, Hawai'i</a>
</p>

<p align="center">
  <img src="assets/teaser.jpg" width="85%">
  <br>
</p>
<em>
<strong>TL;DR:</strong> Our task is to segment dynamic objects in videos based on a few annotated examples that share the same motion patterns. This task focuses on understanding motion information rather than relying solely on static object categories.
</em>

# 📰 News

- **20250906** | Code and dataset are released.
- **20250627** | [**MOVE**](https://github.com/FudanCVL/MOVE) is accepted by **ICCV 2025**! 🌺🏄‍♂️🌴

# 📊 Dataset Preparation

Our dataset is available on [Hugging Face 🤗](https://huggingface.co/datasets/FudanCVL/MOVE). You can download it and places it at:
```shell
pip install -U "huggingface_hub[cli]"
huggingface-cli download FudanCVL/MOVE --repo-type dataset --local-dir ./data/ --local-dir-use-symlinks False --max-workers 16
```

# 🛠️ Environment Setup
First, clone the repository:
```shell
git clone https://github.com/FudanCVL/MOVE
cd MOVE
```
Then, set up the conda environment:

```shell
conda create -n move python=3.10 -y 
conda activate move
pip install -r requirements.txt
```

# 🚀 Train
Before getting started, please ensure your file structure is as shown below.
```
MOVE/                  # root of project
├── data/             
│   └── MOVE_release/ # dataset directory
├── pretrain_model/   
│   ├── resnet50_v2.pth # ResNet pretrained weights
│   └── swin_tiny_patch244_window877_kinetics400_1k.pth      # Swin Transformer pretrained weights
└── ...               # other project files
```
Please download the pretrain backbone weights from [Hugging Face 🤗](https://huggingface.co/FudanCVL/DMA/tree/main/pretrain_model).

Use the following command to start training with `OS` setting, `ResNet` backbone, `2-way-1-shot`, and group `0`:
```shell
torchrun --nproc_per_node=8 tools/train.py \
    --snapshot_dir snapshots \
    --group 0 \
    --num_ways 2 \
    --num_shots 1 \
    --total_episodes 15000 \
    --setting default \
    --loss_type default \
    --resume \
    --query_frames 5 \
    --support_frames 5 \
    --save_interval 1000 \
    --ce_loss_weight 0.25 \
    --iou_loss_weight 5.0 \
    --backbone resnet50 \
    --motion_appear_orth \
    --obj_cls_loss_weight 0.005 \
    --motion_cls_loss_weight 0.005 \
    --orth_loss_weight 0.05
```
# 🧪 Test
Use the following command to test the model with `OS` setting, `ResNet` backbone, `2-way-1-shot`, and group `0`:

```shell
torchrun --nproc_per_node=8 tools/inference.py \
    --snapshot snapshots/resnet50/default/2-way-1-shot/group0/latest_checkpoint.pth \
    --group 0 \
    --num_ways 2 \
    --num_shots 1 \
    --num_episodes 2500 \
    --support_frames 5 \
    --setting default \
    --backbone resnet50 \
    --overwrite
```
We also release the pretrain weights at [Hugging Face 🤗](https://huggingface.co/FudanCVL/DMA/tree/main/snapshots) (WIP 🚧).

# 📇 Citation
If you find our paper and dataset useful for your research, please generously cite our paper.
```
@inproceedings{ying2025move,
  title={{MOVE}: {M}otion-{G}uided {F}ew-{S}hot {V}ideo {O}bject {S}egmentation},
  author={Ying, Kaining and Hu, Hengrui and Ding, Henghui},
  year={2025},
  booktitle={ICCV}
}
```

# 📄 License
MOVE is licensed under a CC BY-NC-SA 4.0 License. The data of MOVE is released for non-commercial research purpose only.