from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.api_wrappers import COCO
from mmcv.parallel.data_container import DataContainer as DC
import os
import os.path as osp
from typing import Dict, Union, Optional, Callable, List
import numpy as np
import torch

from ..pipelines.loading_ext import MultiLayerImageDecode
from ...labelme_utils import (
    get_image_size,
    get_shapes, 
    get_image_path,
    load_json,
    Shape
)
from ...coco_utils import (
    DetAnnotation,
    DetCategory,
    COCOAnnotationFile,
    Image,
    polygons2rle,
    rles_area,
)


def _linestrip2poly(linestrip: Shape) -> Shape:
    STRIP_HALF_WIDTH = 1.5
    assert linestrip.shape_type == 'linestrip'
    pts = np.array(linestrip.points, dtype=np.float32)
    direction_before = np.empty_like(pts)
    direction_after = np.empty_like(pts)
    direction_before[1:] = pts[1:] - pts[:-1]
    direction_after[:-1] = direction_before[1:]
    direction_before[0] = direction_after[0]
    direction_after[-1] = direction_before[-1]
    direction = np.float32(0.5) * (direction_before + direction_after)
    normal = np.empty_like(direction)
    normal[:, 0] = direction[:, 1]
    normal[:, 1] = -direction[:, 0]
    inv_length = torch.rsqrt_(torch.from_numpy(normal).square().sum(dim=1, keepdim=True)).numpy()
    normal *= inv_length * STRIP_HALF_WIDTH
    pts0 = pts + normal
    pts1 = pts - normal
    poly = Shape(label=linestrip.label,
                 points=pts0.tolist() + list(reversed(pts1.tolist())),
                 group_id=linestrip.group_id,
                 shape_type='polygon',
                 flags=linestrip.flags)
    return poly


def _rectangle2ploy(rect: Shape) -> Shape:
    pts = rect.points
    new_pts = [pts[0], [pts[0][0], pts[1][1]], pts[1], [pts[1][0], pts[0][1]]]
    return Shape(label=rect.label,
                 points=new_pts,
                 group_id=rect.group_id,
                 shape_type='polygon',
                 flags=rect.flags)


def _circle2poly(circle: Shape) -> Shape:
    center = np.array(circle.points[0], dtype=np.float32)
    radius_vec = np.array(circle.poitns[1], dtype=np.float32) - center
    radius = np.sqrt(np.sum(radius_vec * radius_vec, keepdims=False))
    cnt = np.int32(np.floor(2 * radius * np.pi))
    rad = np.arange(0, cnt, dtype=np.float32) * 2 * np.pi
    pts = np.empty((cnt, 2), dtype=np.float32)
    pts[:, 0] = np.cos(rad)
    pts[:, 1] = np.sin(rad)
    pts += center.reshape(1, 2)
    return Shape(label=circle.label,
                 points=pts.tolist(),
                 group_id=circle.group_id,
                 shape_type='polygon',
                 flags=circle.flags)


def _polyidentity(shape: Shape) -> Shape:
    return shape


_cvt_to_poly: Dict[str, Callable[[Shape], Shape]] = {
    'circle': _circle2poly,
    'linestrip': _linestrip2poly,
    'rectangle': _rectangle2ploy,
    'polygon': _polyidentity,
}


@DATASETS.register_module()
class LabelmeDataset(CustomDataset):

    MAPPING = None

    def __init__(self, ann_file, pipeline, classes, mapping=None, data_root=None,
                 img_prefix=None, test_mode=False, filter_empty_gt=True):
        self.get_mapping(mapping)
        super().__init__(ann_file, pipeline, classes, data_root, None,
                         None, None, test_mode, filter_empty_gt)
        if img_prefix is not None:
            if ann_file.endswith('/') and not img_prefix.endswith('/'):
                img_prefix += '/'
            for item in self.data_infos:
                item['filename'] = item['filename'].replace(ann_file, img_prefix)

        self.coco = self.create_coco_gt() if test_mode else None
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES) if test_mode else None
        self.img_ids = self.coco.get_img_ids() if test_mode else None

    def pre_pipeline(self, results):
        # set this dataset to grid-aware
        results[MultiLayerImageDecode.GRID_AWARE] = True
        return super().pre_pipeline(results)

    def get_mapping(self, mapping) -> Dict[str, str]:
        if isinstance(mapping, dict):
            if all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
                self.MAPPING = mapping.copy()
        elif isinstance(mapping, str) and osp.exists(mapping):
            from yaml import Loader, load as load_yaml
            try:
                with open(mapping, 'r') as f:
                    mapping_data = load_yaml(f, Loader=Loader)
                assert isinstance(mapping_data, dict)
                assert all(isinstance(k, str) and isinstance(v, str)
                           for k, v in mapping_data.items())
                self.MAPPING = mapping_data
            except Exception:
                pass

    def create_coco_gt(self) -> COCO:
        if True:
            # test if there is a grid decoding by MultiLayerImageDecode
            test_sample = self[0]
            ori_shape = self.data_infos[0]['height'], self.data_infos[0]['width']
            produce_shape = MultiLayerImageDecode.probe_ori_shape(test_sample)
            assert ori_shape[0] % produce_shape[0] == 0
            assert ori_shape[1] % produce_shape[1] == 0
            grid_h = ori_shape[0] // produce_shape[0]
            grid_w = ori_shape[1] // produce_shape[1]
            del test_sample, ori_shape, produce_shape
        
        coco = COCOAnnotationFile()
        coco.categories.extend([
            DetCategory(i, name) for i, name in enumerate(self.CLASSES, 1)
        ])
        img_cnt = 1
        ann_cnt = 1
        for info in self.data_infos:
            coco.images.append(Image(img_cnt,
                                     info['width'] // grid_w,
                                     info['height'] // grid_h,
                                     info['filename']))

            labels: np.ndarray = info['ann']['labels']
            bboxes: np.ndarray = info['ann']['bboxes']
            masks: List[List[List[float]]] = info['ann']['masks']
            rles = [polygons2rle(
                mask, info['height'] // grid_h, info['width'] // grid_w
            ) for mask in masks]
            areas = rles_area(rles)
            xywh_bboxes = np.empty_like(bboxes)
            xywh_bboxes[:, 0:2] = bboxes.reshape(-1, 2, 2).min(axis=1)
            xywh_bboxes[:, 2:4] = bboxes.reshape(-1, 2, 2).max(axis=1) - xywh_bboxes[:, 0:2]
            for i in range(info['ann']['labels'].shape[0]):
                coco.annotations.append(DetAnnotation(
                    ann_cnt, img_cnt,
                    int(labels[i] + 1),  # convert to 1-based
                    rles[i], areas[i],
                    tuple(float(v) for v in xywh_bboxes[i]), 0
                ))
                ann_cnt += 1

            img_cnt += 1

        return coco.toCOCO(True)

    def load_annotations(self, ann_file: str):

        assert self.CLASSES is not None, 'classes MUST be provided for LabelmeDataset'
        mapping = self.MAPPING or {k: k for k in self.CLASSES}
        classes_map = {k: i for i, k in enumerate(self.CLASSES)}

        def map_label(label: str) -> Optional[int]:
            return classes_map.get(mapping.get(label, None), None)

        def get_bbox(shape: Shape) -> np.ndarray:
            pts = np.array(shape.points, dtype=np.float32)
            bbox = np.empty((4,), dtype=np.float32)
            bbox[0:2] = np.min(pts, axis=0)
            bbox[2:4] = np.max(pts, axis=0)
            return bbox

        def raw2anno(json_path: str) -> Dict[str, Union[str, int, Dict]]:
            raw = load_json(json_path)
            h, w = get_image_size(raw)
            shapes = [_cvt_to_poly[shape.shape_type](shape) for shape in get_shapes(raw)]
            mapped_labels = [map_label(s.label) for s in shapes]
            polys = [[sum(shape.points, [])] if label is not None else None
                     for shape, label in zip(shapes, mapped_labels)]
            bboxs = [get_bbox(shape) if label is not None else None
                     for shape, label in zip(shapes, mapped_labels)]
            cvt_ann = {
                'masks': [poly for poly in polys if poly is not None],
                'bboxes': np.stack([bbox for bbox in bboxs if bbox is not None], axis=0),
                'labels': np.array([label for label in mapped_labels if label is not None], dtype=np.int64),
            }
            return {
                'filename': f'{osp.dirname(json_path)}/{get_image_path(raw)}',
                'width': w,
                'height': h,
                'ann': cvt_ann,
            }

        assert osp.isdir(ann_file)

        dirs = [d for d, _, _ in os.walk(ann_file)]
        jsons = [f'{d}/{f}' for d in dirs for f in os.listdir(d) if f.endswith('.json')]

        data_infos = [raw2anno(p) for p in jsons]
        return data_infos


    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        assert 'proposal_fast' not in metric
        # import json
        # import time
        # import mmcv
        # result_files, _ = self.format_results(results)
        # predictions = mmcv.load(result_files['bbox'])
        # predictions_seg = mmcv.load(result_files['segm'])
        # cocoDt = self.coco.loadRes(predictions)
        # cocoDtSeg = self.coco.loadRes(predictions_seg)
        # stamp = str(time.time())
        # with open(f'{stamp}_gt.json', 'w') as f:
        #     json.dump(self.coco.dataset, f, indent=2)
        # with open(f'{stamp}_dt.bbox.json', 'w') as f:
        #     json.dump(cocoDt.dataset, f, indent=2)
        # with open(f'{stamp}_dt.segm.json', 'w') as f:
        #     json.dump(cocoDtSeg.dataset, f, indent=2)
        return self._evaluate(results, metric, logger, jsonfile_prefix, classwise,
                              proposal_nums, iou_thrs, metric_items)

    _evaluate = CocoDataset.evaluate
    format_results = CocoDataset.format_results
    results2json = CocoDataset.results2json
    _segm2json = CocoDataset._segm2json
    _det2json = CocoDataset._det2json
    _proposal2json = CocoDataset._proposal2json
    xyxy2xywh = CocoDataset.xyxy2xywh
