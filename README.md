## Introduction

This is the official implementation of Density-Net[ICTAI 21].

[Density-Net: A Density-Aware Network for 3D Object Detection. ICTAI 2021: 1105-1112](https://ieeexplore.ieee.org/document/9643385)


![demo image](resources/mmdet3d_outdoor_demo.gif)

### Major features

- **The density cluster procedure can be referred to DBSCANSampleShrinkV5 method in mmdet3d/datasets/pipelines/transforms_3dV2.py**


- **the SE module can be referred to mmdet3d/ops/pointnet_modules/point_sa_module_mask_v1.py**

Any question will be welcome!

  
## License

This project is released under the [Apache 2.0 license](LICENSE).

## Tnanks to the extraordinary projects in OpenMMLab
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab next-generation platform for general 3D object detection.

