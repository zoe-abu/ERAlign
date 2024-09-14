### README

##### 1. 项目简介

通过基于图像处理的基本操作提取掌静脉图像的ROI，重点在于结合手掌边缘检测，以优化手掌轮廓的提取。此外，根据手指轮廓的方向对手掌进行方向校正，提升对手掌旋转的鲁棒性。

##### 2.环境依赖

- opencv-python==4.1.2.30
- numpy==1.24.4
- shapely==2.0.6
- scikit-image==0.21.0
- networkx==3.1

##### 3.参数传递

在run_roi.py中更改数据集路径，运行即可