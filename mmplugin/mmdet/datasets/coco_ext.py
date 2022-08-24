from typing import Tuple
from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS
from pycocotools import mask
import numpy as np

from ..pipelines.loading_ext import MultiLayerImageDecode


@DATASETS.register_module()
class ExtendedCocoDataset(CocoDataset):

    def __init__(self, ann_file, pipeline, classes=None, data_root=None,
                 img_prefix='', seg_prefix=None, proposal_file=None,
                 test_mode=False, filter_empty_gt=True):
        super().__init__(ann_file, pipeline, classes, data_root, img_prefix,
                         seg_prefix, proposal_file, test_mode, filter_empty_gt)
        self._grid = self._probe_grid()

    def pre_pipeline(self, results):
        results[MultiLayerImageDecode.GRID_AWARE] = True
        return super().pre_pipeline(results)

    def _probe_grid(self) -> Tuple[int, int]:
        test_sample = self[0]
        ori_shape = self.data_infos[0]['height'], self.data_infos[0]['width']
        produce_shape = MultiLayerImageDecode.probe_ori_shape(test_sample)
        assert ori_shape[0] % produce_shape[0] == 0
        assert ori_shape[1] % produce_shape[1] == 0
        grid_h = ori_shape[0] // produce_shape[0]
        grid_w = ori_shape[1] // produce_shape[1]
        return grid_h, grid_w

    def _segm2json(self, results):
        # re-encode rle in the larger grid-sized image
        # so it can match the image size in GT
        bbox_json_results, segm_json_results = super()._segm2json(results)
        if self._grid != (1, 1):
            for data in segm_json_results:
                seg_size = data['segmentation']['size']
                new_mask = np.zeros((seg_size[0] * self._grid[0],
                                     seg_size[1] * self._grid[1]),
                                    dtype=np.uint8, order='F')
                new_mask[0:seg_size[0], 0:seg_size[1]] = mask.decode(data['segmentation'])
                data['segmentation'] = mask.encode(new_mask)
        return bbox_json_results, segm_json_results
