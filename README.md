# EulerMormer: Robust Eulerian Motion Magnification via Dynamic Filtering within Transformer
[Fei Wang](https://github.com/Jiafei127/), [Dan Guo](https://scholar.google.com.hk/citations?user=DsEONuMAAAAJ&hl=zh-CN&oi=ao), [Kun Li](https://scholar.google.com.hk/citations?user=UQ_bInoAAAAJ&hl=zh-CN&oi=ao), [Meng Wang](https://scholar.google.com.hk/citations?user=rHagaaIAAAAJ&hl=zh-CN&oi=ao). AAAI 2024.

*Hefei University of Technology, Hefei Comprehensive National Science Center, Anhui Zhonghuitong Technology Co., Ltd.*

#### [[arXiv]](https://arxiv.org/abs/2312.04152) | [[PDF]](https://arxiv.org/pdf/2312.04152.pdf)

## ✒️:Abstract
Video Motion Magnification (VMM) aims to break the resolution limit of human visual perception capability and reveal the imperceptible minor motion that contains valuable information in the macroscopic domain. However, challenges arise in this task due to photon noise inevitably introduced by photographic devices and spatial inconsistency in amplification, leading to flickering artifacts in static fields and motion blur and distortion in dynamic fields in the video. Existing methods focus on explicit motion modeling without emphasizing prioritized denoising during the motion magnification process. This paper proposes a novel dynamic filtering strategy to achieve static-dynamic field adaptive denoising. Specifically, based on Eulerian theory, we separate texture and shape to extract motion representation through inter-frame shape differences, expecting to leverage these subdivided features to solve this task finely. Then, we introduce a novel dynamic filter that eliminates noise cues and preserves critical features in the motion magnification and amplification generation phases. Overall, our unified framework, EulerMormer, is a pioneering effort to first equip with Transformer in learning-based VMM. The core of the dynamic filter lies in a global dynamic sparse cross-covariance attention mechanism that explicitly removes noise while preserving vital information, coupled with a multi-scale dual-path gating mechanism that selectively regulates the dependence on different frequency features to reduce spatial attenuation and complement motion boundaries. We demonstrate extensive experiments that EulerMormer achieves more robust video motion magnification from the Eulerian perspective, significantly outperforming state-of-the-art methods.

--- 

## 📅: Update - Dec, 2023
Congratulations on our work being accepted by AAAI 2024!💥💥💥

We will release the source code soon.

## Data preparation
- For **train datasets** from [Oh et al. ECCV 2018](https://github.com/12dmodel/deep_motion_mag), see the official repository [here](https://drive.google.com/drive/folders/19K09QLouiV5N84wZiTPUMdoH9-UYqZrX?usp=sharing).

- For **Real-world datatsets**, we used three settings:
  - [Static Dataset](https://drive.google.com/drive/folders/1Bm3ItPLhRxRYp-dQ1vZLCYNPajKqxZ1a)
  - [Dynamic Dataset](https://drive.google.com/drive/folders/1t5u8Utvmu6gnxs90NLUIfmIX0_5D3WtK)
  - [Fabric Dataset](http://www.visualvibrometry.com/cvpr2015/dataset.html) from [Davis et al. CVPR 2015 && TPAMI](http://www.visualvibrometry.com/publications/visvib_pami.pdf)

- Real-world videos (or any self-prepared videos) need to be configured via the following:
  - Check the settings of val_dir in **config.py** and modify it if necessary.
  - To convert the **Real-world video** into frames:
    `mkdir VIDEO_NAME && ffmpeg -i VIDEO_NAME.mp4 -f image2 VIDEO_NAME/%06d.png`
    
    eg, `mkdir ./val_baby && ffmpeg -i ./baby.avi -f image2 ./val_baby/%06d.png`
> Tips: ffmpeg can also be installed by conda.
  - Modify the frames into **frameA/frameB/frameC**:
    `python make_frameACB.py `(remember adapt the 'if' at the beginning of the program to select videos.)
> Tips: Thanks to a fellow friend [Peng Zheng](https://github.com/ZhengPeng7/motion_magnification_learning-based) for the help!

## Env
`pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html einops`

## 🔖:Citation

If you found this code useful please consider citing our [paper](https://arxiv.org/abs/2312.04152):
```
@inproceedings{wang2024eulermormer,
  title={Eulermormer: Robust eulerian motion magnification via dynamic filtering within transformer},
  author={Wang, Fei and Guo, Dan and Li, Kun and Wang, Meng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={5345--5353},
  year={2024}
}
```



