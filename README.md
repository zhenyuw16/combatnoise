# Combating Noise: Semi-supervised Learning by Region Uncertainty Quantification

This is the mmdetection implementation of our NeurIPS 2021 paper:

>Zhenyu Wang, Yali Li, Ye Guo, Shengjin Wang. Combating Noise: Semi-supervised Learning by Region Uncertainty Quantification. [ArXiv](https://arxiv.org/abs/2111.00928).

# Installation

This code is based on mmdetection v2.18.
Please install the code according to the [mmdetection step](https://github.com/open-mmlab/mmdetection/blob/v2.18.0/docs/get_started.md) first.

### data preparation

```bash
multiphase
├──data
|  ├──VOCdevkit
|  |  ├──VOC2007
|  |  ├──VOC2012
|  ├──coco
|  |  ├──annotations
|  |  |  ├──instances_train2014.json
|  |  |  ├──instances_valminusminival2014.json
|  |  |  ├──instances_minival2014.json
|  |  ├──images
|  |  |  ├──train2014
|  |  |  ├──val2014
```

# Running scripts

## pascal voc

Run:
```bash
python tools/dataset_converters/pascal_voc.py data/VOCdevkit -o labels --out-format coco
python scripts/addscore.py labels/voc07_trainval.json 
```
to prepare the dataset.

Then, to train the supervised model, run (the default gpu number for VOC is 4):
```bash
bash tools/dist_train.sh configs/combatnoise/pascal_voc/faster_rcnn_r50_fpn_1x_voc07_sup.py 4
```
With the supervised model, generating pseudo labels for semi-supervised learning:
```bash
bash scripts/pascal_voc/extract_pl.sh 4 labels/rvoc.pkl labels/voc12_trainval_pl.json 
```
Then, perform semi-supervised learning:
```bash
bash tools/dist_train.sh configs/combatnoise/pascal_voc/faster_rcnn_r50_fpn_1x_voc07_pl.py 4
```

## coco
```bash
python scripts/addscore.py data/coco/annotations/instances_valminusminival2014.json
bash tools/dist_train.sh configs/combatnoise/coco/faster_rcnn_r50_fpn_1x_coco_sup.py 8
bash scripts/coco/extract_pl.sh 8 labels/rcoco.pkl labels/cocotrain2014_pl.json 
bash tools/dist_train.sh configs/combatnoise/coco/faster_rcnn_r50_fpn_1x_coco_pl.py 8
```


# Future features

- [ ] Experiments on COCO partial (1%, 2%, 5%, 10% ratio for labeled images)
