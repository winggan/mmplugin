from mmdet.datasets.builder import PIPELINES
from mmdet.core import BaseInstanceMasks
import os
from typing import Dict, Union, Any
import numpy as np
import cv2


@PIPELINES.register_module()
class ShowAnnotation:

    SHIFT = 4
    SCALE = 1 << SHIFT
    THICK = 1
    COLOR = (255, 255, 255)

    def __init__(self, output_prefix: str, interval: int = -1,
                 show_bbox: bool = False, show_mask: bool = True) -> None:
        self.output_prefix = output_prefix
        self.show_bbox = bool(show_bbox)
        self.show_mask = bool(show_mask)
        self.counter = 0
        self.interval_counter = 0
        self.interval = int(interval)

    def _get_img_name(self) -> str:
        self.counter += 1
        return f'{self.output_prefix}_{os.getpid()}_{self.counter}.jpg'

    def __call__(self, results: Dict[str, Union[np.ndarray, Any]]
                 ) -> Dict[str, Union[np.ndarray, Any]]:
        if (self.interval < 0 or
            self.interval_counter % self.interval == 0):
            self._show_anno(results)

        self.interval_counter += 1
        return results

    def _show_anno(self, results: Dict[str, Union[np.ndarray, Any]]
                   ) -> Dict[str, Union[np.ndarray, Any]]:
        imgs = [results[key] for key in
                results.get('img_fields', ['img'])]

        skip_anno = False
        try:
            masks = [results[key] for key in
                     results.get('mask_fields', ['gt_masks'])]

            bboxes = [results[key] for key in
                    results.get('bbox_fields', ['gt_bboxes'])]

            labels = [results[key] for key in
                      results.get('label_fields', ['gt_labels'])]
        except KeyError:
            skip_anno = True

        img = imgs[0]
        if not skip_anno:
            mask: BaseInstanceMasks = masks[0]
            np_mask = mask.to_ndarray()
            bbox = bboxes[0]
            label = labels[0]
        else:
            label = np.empty((0,), dtype=np.int64)

        if img.shape[2] > 3:
            img = img[:, :, 0:3]

        draw = img.copy()
        for inst in range(label.shape[0]):
            cv2.rectangle(draw,
                          (bbox[inst][0:2] * self.SCALE)
                          .astype(np.int32).tolist(),
                          (bbox[inst][2:4] * self.SCALE)
                          .astype(np.int32).tolist(),
                          self.COLOR,
                          self.THICK,
                          cv2.LINE_AA,
                          self.SHIFT)
            cont, _ = cv2.findContours(np_mask[inst],
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(draw, cont, -1,
                             self.COLOR,
                             self.THICK,
                             cv2.LINE_8)

            cv2.putText(draw, str(label[inst]),
                        (int(max(bbox[inst][0::2])),
                         int(min(bbox[inst][1::2]))),
                        cv2.FONT_HERSHEY_PLAIN, 2,
                        self.COLOR,
                        self.THICK)

        filename = self._get_img_name()
        cv2.imwrite(filename, draw)
