# RCAN_EA


[TorchSR](https://arxiv.org/abs/1707.02921)

## Datasets
*   [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
*   [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html), [Set14](https://paperswithcode.com/dataset/set14), [B100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), [Urban100](https://paperswithcode.com/dataset/urban100)


## Training
```bash
python -m torchsr.train --arch rcan_EA --scale 2 --epochs 300 --loss l1 --dataset-train div2k_bicubic
```
