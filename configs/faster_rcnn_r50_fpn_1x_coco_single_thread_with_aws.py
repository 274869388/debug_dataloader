_base_ = [
    './_base_/models/faster_rcnn_r50_fpn.py',
    './_base_/datasets/coco_detection.py', './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]
# file_client_args = dict(backend='disk')
file_client_args = dict(
    backend='aws',
    path_mapping=dict({
        'data/coco/': 's3://wwcbucket/demo/data/coco/',
        'data/coco/': 's3://wwcbucket/demo/data/coco/'
    }))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        file_client_args=file_client_args),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    # multiprocessing_context='spawn',
    train=dict(pipeline=train_pipeline, file_client_args=file_client_args),
    val=dict(pipeline=test_pipeline, file_client_args=file_client_args),
    test=dict(pipeline=test_pipeline, file_client_args=file_client_args))

evaluation = dict(interval=2)
# interval=2, save_best='acc', out_dir='s3://wwcbucket/demo/ckpt/')

# model = dict(
#     backbone=dict(
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint=
#             's3://wwcbucket/demo/pretrained/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
#         )))

# finetune
# load_from = 's3://wwcbucket/demo/pretrained/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# checkpoint saving
checkpoint_config = dict(interval=2, max_keep_ckpts=2)
# interval=2, max_keep_ckpts=2, out_dir='s3://wwcbucket/demo/ckpt/')

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TextLoggerHook', out_dir='s3://wwcbucket/demo/logs/'),
        # dict(type='TensorboardLoggerHook')
    ])  # yapf:enable

runner = dict(type='EpochBasedRunner', max_epochs=1)
