import torch
from mmcv.runner import auto_fp16
from torch import nn

from ..builder import MIDDLE_ENCODERS


@MIDDLE_ENCODERS.register_module()
class PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape  # [496,432]
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels  # [64]
        self.fp16_enabled = False

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size=None):
        """Foraward function to scatter features."""
        # TODO: rewrite the function in a batch manner
        # no need to deal with different batch cases
        if batch_size is not None:
            return self.forward_batch(voxel_features, coors, batch_size)
        else:
            return self.forward_single(voxel_features, coors)

    def forward_single(self, voxel_features, coors):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel.
                The first column indicates the sample ID.
        """
        # Create the canvas for this sample
        canvas = torch.zeros(
            self.in_channels,
            self.nx * self.ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device)

        indices = coors[:, 1] * self.nx + coors[:, 2]
        indices = indices.long()
        voxels = voxel_features.t()  # 这里还有一个转置操作
        # Now scatter the blob back to the canvas.
        canvas[:, indices] = voxels
        # Undo the column stacking to final 4-dim tensor
        canvas = canvas.view(1, self.in_channels, self.ny, self.nx)
        return [canvas]

    def forward_batch(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)  # [64, 432*496]

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt  # 获取mask，对应此时batch_itt的位置，为True，否则，为False
            this_coors = coors[batch_mask, :]  # 因为coors是所有的voxel index拼接到一起，此时将对应batch_itt的index给提取出来
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]  # 由此可见coors第二个，对应的是nx， 通过这一步操作就获取voxel在整个voxels中的位置
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]  # 获取当前batch_itt duiyingde feature信息，feature信息是由voxel_encoder产生
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels  # 最后的这个canvas就构建了一个pseudo image，对应位置的voxel 不为空

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny,
                                         self.nx)

        return batch_canvas
