from typing import Iterable, Dict, List, Any
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import cv2

from .view import ViewerHandler, get_web_palette


def anno_to_viewitem(anno: Dict[str, Any], cats: Dict[int, Dict[str, Any]]
                     ) -> Dict[str, Any]:
    if ('segmentation' not in anno or
        anno['segmentation'] is None):
        # bbox
        shape = 'Polygon'
        x, y, w, h = anno['bbox']
        points = [x, y, x + w, y, x + w, y + h, x, y + h]

    elif isinstance(anno['segmentation'], dict):
        bimask = cocomask.decode(anno['segmentation'])
        mask = np.where(bimask, np.uint8(255), np.uint8(0))
        points = [[float(v) for v in con.reshape(-1)]
                  for con in cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                  )[0]]
        shape = 'MultiPolygon'

    else:
        points = anno['segmentation']
        shape = 'MultiPolygon'

    cat_id = anno['category_id']

    return {
        'tag': cats[cat_id]['name'] + '|' + str(cat_id) + '\n' +
        '\n'.join(f'{key}={value}' for key, value in dict(
            id=anno['id'],
            area=anno['area'],
            iscrowd=anno['iscrowd'],
        ).items()),
        'shape_type': shape,
        'points': points,
        'viewpoint': 'viewpoint',
        'category': 'defect',
    }


class CocoHandler(ViewerHandler):

    anno = COCO()
    img_prefix = ''
    id2index: Dict[int, int] = {}
    index2id: Dict[int, int] = {}
    palette: Dict[str, str] = {}

    @classmethod
    def set_context(cls, ann_file: str, img_prefix: str = '') -> None:
        cls.anno = COCO(ann_file)
        cls.img_prefix = img_prefix if img_prefix == '' else img_prefix + '/'
        cls.id2index = {img['id']: idx for idx, img in enumerate(
            sorted(cls.anno.imgs.values(),
                   key=lambda img: (img['file_name'], img['width'], img['height']))
        )}
        cls.index2id = {value: key for key, value in cls.id2index.items()}
        all_color = get_web_palette(255)[1:][::-1]
        cls.palette = {cat['name']: all_color[cat['id']]
                       for cat in cls.anno.cats.values()}

    @classmethod
    def all_item_ids(cls) -> Iterable[str]:
        return (str(cls.index2id[idx]) for idx in range(cls.num_items()))

    @classmethod
    def prev_item_id(cls, item_id: str) -> str:
        idx = cls.get_index(item_id)
        idx -= 1
        if idx <= 0:
            idx = len(cls.index2id) - 1
        return str(cls.index2id[idx])

    @classmethod
    def next_item_id(cls, item_id: str) -> str:
        idx = cls.get_index(item_id)
        idx += 1
        if idx >= len(cls.id2index):
            idx = 0
        return str(cls.index2id[idx])

    @classmethod
    def get_index(cls, item_id: str) -> int:
        return cls.id2index[int(item_id)]

    @classmethod
    def num_items(cls) -> int:
        return len(cls.anno.imgs)

    @classmethod
    def get_layers(cls, item_id: str) -> Dict[str, str]:
        return {'image': cls.anno.imgs[int(item_id)]['file_name']}

    @classmethod
    def get_palette(cls, item_id: str) -> Dict[str, str]:
        return cls.palette

    @classmethod
    def get_shapes(cls, item_id: str) -> List[Dict[str, Any]]:
        return [anno_to_viewitem(anno, cls.anno.cats)
                for anno in cls.anno.imgToAnns[int(item_id)]]

    @classmethod
    def get_image_path(cls, filename: str) -> str:
        return cls.img_prefix + filename

    @classmethod
    def get_title(cls, item_id: str) -> str:
        return cls.anno.imgs[int(item_id)]['file_name']


if __name__ == '__main__':
    from argparse import ArgumentParser
    import http.server as svr

    parser = ArgumentParser()
    parser.add_argument('ann_file', type=str, help='coco annotation json file')
    parser.add_argument('--img_prefix', type=str, default='', help='path to load image')
    parser.add_argument('--addr', type=str, default='127.0.0.1', help='address to bind')
    parser.add_argument('--port', type=int, default=33693)
    args = parser.parse_args()

    CocoHandler.set_context(args.ann_file, args.img_prefix)

    server = svr.HTTPServer((args.addr, args.port), CocoHandler)
    server.serve_forever()
