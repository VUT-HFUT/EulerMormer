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


## 🔖:Citation

If you found this code useful please consider citing our [paper](https://arxiv.org/abs/2312.04152):

```
@article{wang2023eulermormer,
  title={EulerMormer: Robust Eulerian Motion Magnification via Dynamic Filtering within Transformer},
  author={Wang, Fei and Guo, Dan and Li, Kun and Wang, Meng},
  journal={arXiv preprint arXiv:2312.04152},
  year={2023}
}
```



