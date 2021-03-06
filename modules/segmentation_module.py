from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d


class SegmentationModule(nn.Module):
    """
    Computing a segmentation map and affine transformations.
    """

    def __init__(self, block_expansion, num_segments, num_channels, max_features,
                 num_blocks, temperature, estimate_affine_part=False, scale_factor=1):
        super(SegmentationModule, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)
        self.num_segments = num_segments
        self.shift = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_segments, kernel_size=(7, 7),
                               padding=(3, 3))

        if estimate_affine_part:
            self.affine = nn.Conv2d(in_channels=self.predictor.out_filters,
                                    out_channels=4 * num_segments, kernel_size=(7, 7), padding=(3, 3))
            self.affine.weight.data.zero_()
            self.affine.bias.data.copy_(torch.tensor([1, 0, 0, 1] * num_segments, dtype=torch.float))
        else:
            self.affine = None

        self.segmentation = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=(1 + num_segments), kernel_size=(7, 7), padding=(3, 3))

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and the variance from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        result = (heatmap * grid).sum(dim=(2, 3))

        return result

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        out = {}
        feature_map = self.predictor(x)
        out['segmentation'] = F.softmax(self.segmentation(feature_map), dim=1)

        prediction = self.shift(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out['shift'] = self.gaussian2kp(heatmap)

        if self.affine is not None:
            affine_map = self.affine(feature_map)
            affine_map = affine_map.reshape(final_shape[0], self.num_segments, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)

            affine = heatmap * affine_map
            affine = affine.view(final_shape[0], final_shape[1], 4, -1)
            affine = affine.sum(dim=-1)
            affine = affine.view(affine.shape[0], affine.shape[1], 2, 2)
            out['affine'] = affine

        return out
