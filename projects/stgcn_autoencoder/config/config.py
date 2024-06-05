_base_ = "./default_runtime.py"

# models.utils.graph要改

custom_imports = dict(imports=["models", "metrics"])

model = dict(
    type="RecognizerGCNAnimal",
    backbone=dict(
        type="STGCNAnimal",
        data_bn_type="VC",
        in_channels=12,
        base_channels=32,
        num_stages=6,
        enable_social=True,
        deflate_stages=[3, 4, 5],
        inflate_stages=[],
        down_stages=[2, 3, 4, 5],
        gcn_adaptive="init",
        gcn_with_res=True,
        tcn_type="mstcn",
        graph_cfg=dict(layout="mouse", mode="spatial"),
    ),
    cls_head=dict(
        type="GCNHeadDecoderAnimal",
        num_classes=3,
        in_channels=4,
        base_channels=32,
        deflate_stages=[4, 5],
        up_stages=[3, 4, 5, 6],
        num_stages=6,
        graph_cfg=dict(layout="mouse", mode="spatial"),
        label_smooth_eps=0.6,
        topk=(1, 2),
        gcn_adaptive="init",
        gcn_with_res=True,
        tcn_type="unit_tcn",
        enable_social=False,
    ),
)

dataset_type = "PoseDataset"
ann_file = "./mmaction_skeleton_dual_stage3_exp_half.pkl"
# ann_file = '../tracker/mmaction_skeleton_dual_stage3_exp_complete_new.pkl'
train_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="RandomSwapIndividuals"),
    dict(type="RandomRotation", range=180),
    dict(type="RandomZoom", range=0.1),
    dict(type="GenSkeFeat", dataset="mouse", feats=["j", "b", "jm", "bm"]),
    dict(type="UniformSampleFrames", clip_len=64 * 1),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=2),
    dict(
        type="PackActionInputs",
        meta_keys=(
            "img_shape",
            "img_key",
            "video_id",
            "timestamp",
            "comb",
            "video_name",
        ),
    ),
]
val_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="mouse", feats=["j", "b", "jm", "bm"]),
    dict(type="UniformSampleFrames", clip_len=64 * 1, num_clips=1, test_mode=True),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=2),
    dict(
        type="PackActionInputs",
        meta_keys=(
            "img_shape",
            "img_key",
            "video_id",
            "timestamp",
            "comb",
            "video_name",
        ),
    ),
]
test_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="mouse", feats=["j", "b", "jm", "bm"]),
    dict(type="UniformSampleFrames", clip_len=64 * 1, num_clips=1, test_mode=True),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=2),
    dict(
        type="PackActionInputs",
        meta_keys=(
            "img_shape",
            "img_key",
            "video_id",
            "timestamp",
            "comb",
            "video_name",
        ),
    ),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=2,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split="xsub_train",
        ),
    ),
)
val_dataloader = dict(
    batch_size=256,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split="xsub_val",
        test_mode=True,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split="xsub_val",
        test_mode=True,
    ),
)


val_evaluator = [
    dict(type="AccMetric", metric_options=dict(top_k_accuracy=dict(topk=(1, 2)))),
    dict(type="ConfusionMatrix", num_classes=3),
    dict(type="LossRestoreAnimal"),
]
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_begin=1, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        eta_min=0,
        T_max=100,
        by_epoch=True,
        convert_to_iter_based=True,
    )
]

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=0.01, betas=(0.9, 0.999), weight_decay=0.05)
)

default_hooks = dict(checkpoint=dict(interval=1), logger=dict(interval=50))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=256)
