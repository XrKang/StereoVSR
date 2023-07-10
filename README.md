Stereo Video Super-Resolution via Exploiting View-Temporal Correlations
====

> Stereo Video Super-Resolution via Exploiting View-Temporal Correlations, In *ACM MM* 2021.\
> Ruikang Xu, Zeyu Xiao, Mingde Yao, Yueyi Zhang, Zhiwei Xiong. 

[Paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475189)|[Supplemental Material](https://dl.acm.org/action/downloadSupplement?doi=10.1145%2F3474085.3475189&file=mfp0182aux.zip)|[Video](https://dl.acm.org/action/downloadSupplement?doi=10.1145%2F3474085.3475189&file=mm2021.mp4)
## Dependencies
- This repository is based on [[EDVR/old_version]](https://github.com/xinntao/EDVR/tree/old_version), you can install DeformConv by following [[EDVR/old_version]](https://github.com/xinntao/EDVR/tree/old_version)
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch 1.2.0](https://pytorch.org/): `conda install pytorch=1.2.0 torchvision cudatoolkit=9.2 -c pytorch`
- numpy: `pip install numpy`
- opencv: `pip install opencv-python`
- tensorboardX: `pip install tensorboardX`

## Datesets
* The SceneFlow dataset can be downloaded from this [link](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html).
* The KITTI-2012 dataset can be downloaded from this [link](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo).
* The KITTI-2015 dataset can be downloaded from this [link](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).

## Quick Start
We take the sceneflow dataset as an example:

* Prepare text files for loading data:
```
cd dataPrepare && python creatTxt_sceneflow.py
```

* Train the model:
```
cd code && python train.py
```

* Test the model:
```
cd code && python test.py
```

## Citation
```
@inproceedings{xu2021stereo,
  title={Stereo video super-resolution via exploiting view-temporal correlations},
  author={Xu, Ruikang and Xiao, Zeyu and Yao, Mingde and Zhang, Yueyi and Xiong, Zhiwei},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={460--468},
  year={2021}
}
```

## Contact
Any question regarding this work can be addressed to xurk@mail.ustc.edu.cn.