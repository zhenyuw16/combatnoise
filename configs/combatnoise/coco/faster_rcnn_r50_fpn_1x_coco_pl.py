_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn_cb.py',
    '../../_base_/datasets/coco_detection_pl.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    roi_head = dict(
        bbox_head = dict(
            iter_perepoch = 13894,
            total_epochs = 6,
            trans_epoch = 4)))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[4, 5])
runner = dict(type='EpochBasedRunner', max_epochs=6)

