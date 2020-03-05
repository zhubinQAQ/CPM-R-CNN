# CPM-R-CNN
CPM R-CNN: Calibrating Point-guided Misalignment in Object Detection
<p align="center"><img width="60%" src="data/introduction.png" /></p>

In this repository, we release the CPM R-CNN code in Pytorch.

- CPM R-CNN pipeline:
<p align="center"><img width="90%" src="data/pipeline.png" /></p>

- Modules in CPM R-CNN:
<p align="center"><img width="70%" src="data/cmm.png" /></p>
<p align="center"><img width="70%" src="data/score.png" /></p>


## Installation
- 8 x TITAN RTX GPU
- pytorch1.1
- python3.6.8

- Other details will be public soon.

## Results and Models

**On MS COCO test-dev**

|  Backbone  |  LR  | mAP | AP50 | (APs/APm/APl) | DOWNLOAD |
|------------|:----:|:------:|:----:|:--------------------------:| :-------:|
|  R-50-FPN  |  2x  | 41.7   | 59.2 |      23.1/44.0/54.7        | [[GoogleDrive]](https://drive.google.com/open?id=1EtqFhrFTdBJNbp67effArVrTNx4q_ELr) [[BaiduPan]:a7k0](https://pan.baidu.com/s/1i5Kvbu4PCA6o4ktlx3P57w)|
|  R-101-FPN  |  2x  | 43.3   | 61.2 |      23.9/46.3/56.6        | [[GoogleDrive]](https://drive.google.com/open?id=1EtqFhrFTdBJNbp67effArVrTNx4q_ELr) [[BaiduPan]:mpc8](https://pan.baidu.com/s/1IbfitzvycDrtm0Hh1SDBRw)|
|  X-101-FPN-DCN |  2x  | 46.4   | 65.3 |      26.8/49.4/61.0        | [[GoogleDrive]](https://drive.google.com/open?id=1EtqFhrFTdBJNbp67effArVrTNx4q_ELr) [[BaiduPan]:enbd](https://pan.baidu.com/s/1YvW4Tb0nrgQADaLlay1tGQ)|

**Component-wise performance**

|  CMM |  ISM | RSM | mAP |
|:-----:|:----:|:----:|:----:|
|       |      |      | 39.9 |
|  yes  |      |      | 40.7 |
|       |  yes |      | 40.5 |
|       |      |  yes | 40.6 |
|  yes  |  yes |  yes | 41.3 |       


**ImageNet pretrained weight**

- [R-50](https://drive.google.com/open?id=1_QXYuUbNUrRbsyPeYB9EJdwGp_SBFBrZ)
- [R-101](https://drive.google.com/open?id=1k1N1wuklAYuBD8DX229ZEMsp8opjDJNE)
- [X-101-64x4d](https://drive.google.com/open?id=1abiIjSUJXOZzxAX66aYjCSsWXGJp3z05)


## Training

To train a model with 8 GPUs run:
```
None
```


## Evaluation

### multi-gpu evaluation,
```
None
```

### single-gpu evaluation,
```
None
```
