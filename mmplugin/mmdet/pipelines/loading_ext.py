from typing import List, Tuple, Dict, Union, Any
import numpy as np
from torch import Tensor
import torch

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class MultiLayerImageDecode:
    """
    Decode a multi-layer image which is represented as a (grid_h * h) x (grid_w * w) x (3 or 1) normal image
    into its original form, a.k.a h x w x true_channels. The default arguments of constructor is to bypass a
    normal 3-channels image.
    A multi-layer image usually has more than 3 or 4 channels from different sensors

    Args:
    grid_h (int): number of rows of the grid
    grid_w (int): number of coloumns of the gird
    channel_sel (list[tuple[int, int, int]]): a list of selector, each is a tuple of (grid_y, grid_x, channel_index),
                                              representing given channel of given region is to be a channel of final
                                              composed multilayer image
    """
    GRID_AWARE = 'grid_aware'

    def __init__(self, grid_h: int = 1, grid_w: int = 1,
                 channel_sel: List[Tuple[int, int, int]] = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
                 ) -> None:
        self.grid = tuple(int(v) for v in (grid_h, grid_w))
        if any(v <= 0 for v in self.grid):
            raise ValueError(f'grid h/w should always be positive, found {self.grid}')
        self.sel = channel_sel.copy()

        if any(not isinstance(t, tuple) or len(t) != 3 or
               not isinstance(t[0], int) or t[0] < 0 or t[0] >= self.grid[0] or
               not isinstance(t[1], int) or t[1] < 0 or t[1] >= self.grid[1] or
               not isinstance(t[2], int) or t[1] < 0 or t[1] >= 3
               for t in self.sel):
            raise ValueError(f'invalid channel selector: {self.sel}')

    def _sel(self, image: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        if image.ndim == 2:
            image = image.reshape(image.shape + (1,))
        shape = image.shape
        if shape[0] % self.grid[0] != 0 or shape[1] % self.grid[1] != 0:
            raise RuntimeError(f'image with shape {shape} cannot be split into grid {self.grid}')

        out_shape = (shape[0] // self.grid[0], shape[1] // self.grid[1], len(self.sel))
        ret = torch.empty(out_shape, dtype=image.dtype, device=image.device) \
            if isinstance(image, Tensor) else np.empty(out_shape, dtype=image.dtype)

        _image = image.reshape(self.grid[0], out_shape[0], self.grid[1], out_shape[1], image.shape[2])
        for i, sel in enumerate(self.sel):
            ret[:, :, i] = _image[sel[0], :, sel[1], :, sel[2]]

        return ret

    def __call__(self, results: Dict[str, Union[np.ndarray, Any]]
                 ) -> Dict[str, Union[np.ndarray, Any]]:
        if self.GRID_AWARE not in results:
            raise RuntimeError(f'{self.__class__.__name__} should be used with '
                               f'GRID_AWARE dataset, which would put a key \'{self.GRID_AWARE}\' '
                               f'into results, to avoid image shape inconsistency')

        if 'ann_info' in results and 'masks' in results['ann_info']:
            if any(isinstance(mask, dict) and 'counts' in mask
                   for mask in results['ann_info']['masks']):
                raise RuntimeError(f'{self.__class__.__name__} is not compatible with mask '
                                    'annotation in RLE format, consider use polygons')

        for key in results.get('img_fields', ['img']):
            results[key] = self._sel(results[key])

        # update the shape
        for key in results.get('img_fields', ['img']):
            ori_h, ori_w = results['img_shape'][0:2]
            results['img_shape'] = results[key].shape
            results['ori_shape'] = results[key].shape
            break

        if 'img_info' in results:
            img_info = results['img_info'].copy()
            if 'height' in img_info and img_info['height'] == ori_h:
                img_info['height'] = results['img_shape'][0]
            if 'width' in img_info and img_info['width'] == ori_w:
                img_info['width'] = results['img_shape'][1]
            results['img_info'] = img_info

        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'grid_h={self.grid[0]}, '
                    f"grid_w='{self.grid[1]}', "
                    f'channel_sel={self.sel})')
        return repr_str

    @staticmethod
    def probe_ori_shape(results: Dict[str, Any]) -> Tuple[int, int]:
        from mmcv.parallel.data_container import DataContainer as DC
        produce_shape = None
        if 'ori_shape' in results:
            produce_shape = results['ori_shape'][0:2]
        elif 'img_metas' in results:
            if isinstance(results['img_metas'], list):
                meta = results['img_metas'][0]
            else:
                meta = results['img_metas']
            if isinstance(meta, dict):
                produce_shape = meta['ori_shape'][0:2]
            elif isinstance(meta, DC) and isinstance(meta.data, dict):
                produce_shape = meta.data['ori_shape'][0:2]
            elif isinstance(meta, DC) and isinstance(meta.data, list):
                produce_shape = meta.data[0]['ori_shape'][0:2]
        if produce_shape is None:
            raise RuntimeError(f'Cannot determine ori_shape in a sample after pipeline')
        return produce_shape
