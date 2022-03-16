_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn_cb.py', '../../_base_/datasets/voc_pl.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    roi_head = dict(
        bbox_head = dict(
            num_classes = 20, 
            iter_perepoch = 6079,
            total_epochs = 4,
            trans_epoch = 3
            )))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12
