# Image Super-Resolution Using Very Deep Residual Channel Attention Networks + Efficient Attention (RCAN_EA)

## Đề tài được xây dựng trên:
*   [TorchSR](https://github.com/Coloquinte/torchSR)
*   [RCAN](https://github.com/yulunzhang/RCAN)
*   [EA](https://github.com/cmsflash/efficient-attention)

## Tập dữ liệu:
*   [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
*   [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html), [Set14](https://paperswithcode.com/dataset/set14), [B100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), [Urban100](https://paperswithcode.com/dataset/urban100)

## Train:
python -m torchsr.train --arch rcan_EA --scale 2 --epochs 100 --loss l1 --dataset-train div2k_bicubic

## Evaluate:
python -m torchsr.train --validation-only --arch edsr_EA --scale 2 --dataset-val set5 --chop-size 400 --shave-border 2 --eval-luminance --load-checkpoint ./results/models/*Your model file*

Bạn có thể sử dụng model được train sẵn tại (100 epochs): [GoogleDrive](https://drive.google.com/file/d/1Av8NDZU8rHd4hcupSDDQ9KwqxpLFHIVX/view?usp=sharing)

Chi tiết cách sử dụng hãy tham khảo: [TorchSR](https://github.com/Coloquinte/torchSR)
