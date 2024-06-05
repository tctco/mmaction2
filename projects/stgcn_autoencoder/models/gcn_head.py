from mmaction.registry import MODELS
from mmaction.models.heads import BaseHead
from mmengine.structures import LabelData
import copy as cp
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList
from torch.nn import Sequential

from mmaction.registry import MODELS
from mmaction.models.utils import Graph, unit_gcn
from mmcv.cnn import build_norm_layer
import torch.nn.functional as F

EPS = 1e-4


class unit_tcn_decoder(BaseModule):
    """The basic unit of temporal convolutional network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the temporal convolution kernel.sssssssss
            Defaults to 9.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        dilation (int): Spacing between temporal kernel elements.
            Defaults to 1.
        norm (str): The name of norm layer. Defaults to ``'BN'``.
        dropout (float): Dropout probability. Defaults to 0.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
                dict(type='Constant', layer='BatchNorm2d', val=1),
                dict(type='Kaiming', layer='Conv2d', mode='fan_out')
            ]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1,
        norm: str = "BN",
        dropout: float = 0,
        init_cfg: Union[Dict, List[Dict]] = [
            dict(type="Constant", layer="BatchNorm2d", val=1),
            dict(type="Kaiming", layer="Conv2d", mode="fan_out"),
        ],
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        output_padding = (1, 0) if stride > 1 else (0, 0)
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            output_padding=output_padding,
            dilation=(dilation, 1),
        )
        self.bn = (
            build_norm_layer(self.norm_cfg, out_channels)[1]
            if norm is not None
            else nn.Identity()
        )

        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.drop(self.bn(self.conv(x)))


class mstcn_decoder(BaseModule):
    """The multi-scale temporal convolutional network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int): Number of middle channels. Defaults to None.
        dropout (float): Dropout probability. Defaults to 0.
        ms_cfg (list): The config of multi-scale branches. Defaults to
            ``[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']``.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        init_cfg (dict or list[dict]): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        dropout: float = 0.0,
        ms_cfg: List = [(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"],
        stride: int = 1,
        activation: str = "relu",
        init_cfg: Union[Dict, List[Dict]] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == None:
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = max(out_channels // num_branches, 1)
            rem_mid_channels = max(out_channels - mid_channels * (num_branches - 1), 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == "1x1":
                branches.append(
                    nn.ConvTranspose2d(
                        in_channels,
                        branch_c,
                        kernel_size=1,
                        output_padding=(1, 0),
                        stride=(stride, 1),
                    )
                )
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == "max":
                branches.append(
                    Sequential(
                        nn.ConvTranspose2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        nn.MaxPool2d(
                            kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0)
                        ),
                    )
                )
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = Sequential(
                nn.ConvTranspose2d(in_channels, branch_c, kernel_size=1),
                nn.BatchNorm2d(branch_c),
                self.act,
                unit_tcn_decoder(
                    branch_c,
                    branch_c,
                    kernel_size=cfg[0],
                    stride=stride,
                    dilation=cfg[1],
                    norm=None,
                ),
            )
            branches.append(branch)

        self.branches = ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = Sequential(
            nn.BatchNorm2d(tin_channels),
            self.act,
            nn.ConvTranspose2d(tin_channels, out_channels, kernel_size=1),
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        N, C, T, V = x.shape

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)


class STGCNDecodeBlock(BaseModule):
    """The basic block of STGCN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_nodes, num_nodes)`.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        residual (bool): Whether to use residual connection. Defaults to True.
        init_cfg (dict or list[dict], optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        activation: str = "relu",
        init_cfg: Optional[Union[Dict, List[Dict]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == "gcn_"}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == "tcn_"}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ["gcn_", "tcn_"]}
        assert len(kwargs) == 0, f"Invalid arguments: {kwargs}"

        tcn_type = tcn_kwargs.pop("type", "unit_tcn")
        assert tcn_type in ["unit_tcn", "mstcn"]
        gcn_type = gcn_kwargs.pop("type", "unit_gcn")
        assert gcn_type in ["unit_gcn"]

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == "unit_tcn":
            self.tcn = unit_tcn_decoder(
                out_channels, out_channels, 9, stride=stride, **tcn_kwargs
            )
        elif tcn_type == "mstcn":
            self.tcn = mstcn_decoder(
                out_channels, out_channels, stride=stride, **tcn_kwargs
            )
        if activation == "silu":
            self.relu = nn.SiLU()
        elif activation == "tanh":
            self.relu = nn.Tanh()
        elif activation == None:
            self.relu = nn.Identity()
        else:
            self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn_decoder(
                in_channels, out_channels, kernel_size=1, stride=stride
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x)) + res
        return self.relu(x)


@MODELS.register_module()
class GCNHeadDecoderAnimal(BaseHead):
    """STGCN backbone.

    Spatial Temporal Graph Convolutional
    Networks for Skeleton-Based Action Recognition.
    More details can be found in the `paper
    <https://arxiv.org/abs/1801.07455>`__ .

    Args:
        graph_cfg (dict): Config for building the graph.
        in_channels (int): Number of input channels. Defaults to 3.
        base_channels (int): Number of base channels. Defaults to 64.
        data_bn_type (str): Type of the data bn layer. Defaults to ``'VC'``.
        ch_ratio (int): Inflation ratio of the number of channels.
            Defaults to 2.
        num_person (int): Maximum number of people. Only used when
            data_bn_type == 'MVC'. Defaults to 2.
        num_stages (int): Total number of stages. Defaults to 10.
        inflate_stages (list[int]): Stages to inflate the number of channels.
            Defaults to ``[5, 8]``.
        up_stages (list[int]): Stages to perform downsampling in
            the time dimension. Defaults to ``[5, 8]``.
        stage_cfgs (dict): Extra config dict for each stage.
            Defaults to ``dict()``.
        init_cfg (dict or list[dict], optional): Config to control
            the initialization. Defaults to None.

        Examples:
        >>> import torch
        >>> from mmaction.models import STGCN
        >>>
        >>> mode = 'stgcn_spatial'
        >>> batch_size, num_person, num_frames = 2, 2, 150
        >>>
        >>> # openpose-18 layout
        >>> num_joints = 18
        >>> model = STGCN(graph_cfg=dict(layout='openpose', mode=mode))
        >>> model.init_weights()
        >>> inputs = torch.randn(batch_size, num_person,
        ...                      num_frames, num_joints, 3)
        >>> output = model(inputs)
        >>> print(output.shape)
        >>>
        >>> # nturgb+d layout
        >>> num_joints = 25
        >>> model = STGCN(graph_cfg=dict(layout='nturgb+d', mode=mode))
        >>> model.init_weights()
        >>> inputs = torch.randn(batch_size, num_person,
        ...                      num_frames, num_joints, 3)
        >>> output = model(inputs)
        >>> print(output.shape)
        >>>
        >>> # coco layout
        >>> num_joints = 17
        >>> model = STGCN(graph_cfg=dict(layout='coco', mode=mode))
        >>> model.init_weights()
        >>> inputs = torch.randn(batch_size, num_person,
        ...                      num_frames, num_joints, 3)
        >>> output = model(inputs)
        >>> print(output.shape)
        >>>
        >>> # custom settings
        >>> # instantiate STGCN++
        >>> model = STGCN(graph_cfg=dict(layout='coco', mode='spatial'),
        ...               gcn_adaptive='init', gcn_with_res=True,
        ...               tcn_type='mstcn')
        >>> model.init_weights()
        >>> output = model(inputs)
        >>> print(output.shape)
        torch.Size([2, 2, 256, 38, 18])
        torch.Size([2, 2, 256, 38, 25])
        torch.Size([2, 2, 256, 38, 17])
        torch.Size([2, 2, 256, 38, 17])
    """

    def __init__(
        self,
        num_classes: int,
        graph_cfg: Dict,
        in_channels: int = 2,
        base_channels: int = 3,
        data_bn_type: str = "VC",
        ch_ratio: int = 2,
        stride_ratio: int = 2,
        num_person: int = 2,
        enable_social: bool = False,
        num_stages: int = 10,
        inflate_stages: List[int] = [],
        deflate_stages: List[int] = [],
        up_stages: List[int] = [],
        loss_cls: Dict = dict(type="CrossEntropyLoss"),
        loss_identification=None,
        num_individuals=None,
        dropout: float = 0.0,
        loss_restore: Dict = dict(type="mmdet.MSELoss"),
        average_clips: str = "prob",
        init_cfg: Union[Dict, List[Dict]] = dict(
            type="Normal", layer="Linear", std=0.01
        ),
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            average_clips=average_clips,
            init_cfg=init_cfg,
            multi_class=kwargs.pop("multi_class", False),
            label_smooth_eps=kwargs.pop("label_smooth_eps", 0.0),
            topk=kwargs.pop("topk", (1, 5)),
        )

        self.dropout_ratio = dropout
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.loss_restore = MODELS.build(loss_restore)

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type

        if data_bn_type == "MVC":
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == "VC":
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, (tuple, list)) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop("tcn_dropout", None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.deflate_stages = deflate_stages
        self.up_stages = up_stages
        self.num_person = num_person
        self.enable_social = enable_social

        modules = []
        if self.in_channels != self.base_channels:
            modules = [
                STGCNDecodeBlock(
                    in_channels,
                    base_channels,
                    A.clone(),
                    1,
                    residual=False,
                    **lw_kwargs[0],
                )
            ]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            if i in up_stages:
                stride = max(stride_ratio, 2)
            else:
                stride = 1
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            elif i in deflate_stages:
                inflate_times -= 1
            out_channels = int(
                self.base_channels * self.ch_ratio**inflate_times + EPS
            )
            base_channels = out_channels
            if i == num_stages:
                activation = "tanh"
            else:
                activation = "relu"
            modules.append(
                STGCNDecodeBlock(
                    in_channels,
                    out_channels,
                    A.clone(),
                    stride,
                    activation=activation,
                    **lw_kwargs[i - 1],
                )
            )

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = ModuleList(modules)
        self.fc_decoder = nn.Linear(out_channels, 2)

        if loss_identification:
            self.loss_id = MODELS.build(dict(type="CrossEntropyLoss"))
            self.fc_id = nn.Linear(self.in_channels, self.num_person)
        else:
            self.loss_id = None
        # self.social = unit_social(num_person, in_channels, activation=activation) if num_person > 1 and enable_social else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        cls_branch = x
        N, M, C, T, V = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == "MVC":
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = (
            x.view(N, M, V, C, T)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(N * M, C, T, V)
        )

        # if self.enable_social and self.num_person > 1:
        #     for i in range(self.num_stages):
        #         x = self.gcn[i*2](x)
        #         x = self.gcn[i*2+1](x)
        # else:
        for i in range(self.num_stages):
            x = self.gcn[i](x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc_decoder(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = x.reshape((N, M) + x.shape[1:])

        N, M, C, T, V = cls_branch.shape
        cls_branch = cls_branch.view(N * M, C, T, V)
        # if self.enable_social and self.num_person > 1:
        #     cls_branch = self.social(cls_branch)
        cls_branch = self.pool(cls_branch)
        cls_branch = cls_branch.view(N, M, C)
        cls_branch = cls_branch.mean(dim=1)
        assert cls_branch.shape[1] == self.in_channels

        if self.dropout is not None:
            cls_branch = self.dropout(cls_branch)

        cls_scores = self.fc(cls_branch)
        if self.loss_id is None:
            return (cls_scores, x)
        else:
            id_scores = self.fc_id(cls_branch)
            return (cls_scores, x, id_scores)

    def loss(self, feats, data_samples, **kwargs):
        """Perform forward propagation of head and loss calculation on the
        features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        inputs = kwargs.pop("inputs")
        if self.loss_id is None:
            cls_scores, restored = self(feats, **kwargs)
        else:
            cls_scores, restored, id_scores = self(feats, **kwargs)
        loss = self.loss_by_feat(cls_scores, data_samples)
        # loss['loss_cls'] = torch.tensor(0, dtype=torch.float32, device=cls_scores.device)
        restore_loss = self.loss_by_input(restored, inputs)
        loss.update(restore_loss)
        if self.loss_id is not None:
            id_loss = self.loss_by_id(id_scores, data_samples)
            loss["loss_id"] = id_loss
        return loss

    def loss_by_input(self, restored, inputs):
        restored = restored.permute(0, 1, 3, 4, 2).contiguous()
        inputs = inputs[..., :2]
        assert (
            inputs.shape == restored.shape
        ), f"inputs shape {inputs.shape} != restored shape {restored.shape}"
        losses = dict()
        loss = self.loss_restore(restored, inputs)
        if isinstance(loss, dict):
            losses.update(loss)
        else:
            losses["loss_restore"] = loss
        return losses

    def loss_by_id(self, id_scores, data_samples):
        labels = [x.target for x in data_samples]
        labels = torch.tensor(labels, dtype=torch.long, device=id_scores.device)
        labels = labels.squeeze()
        if self.label_smooth_eps != 0:
            if id_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_person)
            labels = (
                1 - self.label_smooth_eps
            ) * labels + self.label_smooth_eps / self.num_person
        loss_id = self.loss_id(id_scores, labels)
        return loss_id

    def predict(self, feats, data_samples, **kwargs):
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        inputs = kwargs.pop("inputs")
        if self.loss_id is not None:
            cls_scores, restored, _ = self(feats, **kwargs)
        else:
            cls_scores, restored = self(feats, **kwargs)
        restored = restored.permute(0, 1, 3, 4, 2).contiguous()
        inputs = inputs[..., :2]
        data_samples = self.predict_by_feat(cls_scores, data_samples)
        for data_sample, input, restore, feat in zip(
            data_samples, inputs, restored, feats
        ):
            data_sample.restored_sequence = LabelData(item=restore)
            data_sample.gt_sequence = LabelData(item=input)
            loss = self.loss_restore(restore, input)
            data_sample.restore_loss = LabelData(item=loss)
            data_sample.feat = LabelData(item=feat)
        return data_samples
