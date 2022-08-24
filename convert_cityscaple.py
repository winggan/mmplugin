from typing import List, Optional, Dict
from glob import glob
import os
from dataclasses import dataclass
from mmdet.datasets import CityscapesDataset
from PIL import Image
import numpy as np

from mmplugin.coco_utils import DetCategory, Image, DetAnnotation, COCOAnnotationFile
from pycocotools import mask
from cityscapesscripts.helpers.annotation import Annotation, CsPoly, CsObjectType


@dataclass
class ImageWithSegm(Image):
    segm_file: Optional[str] = None


def anno_cs2coco(obj: CsPoly, cls_mapping: Dict[str, DetCategory], h: int, w: int) -> Optional[DetAnnotation]:
    from cityscapesscripts.helpers.labels import name2label
    label   = obj.label
    polygon = obj.polygon

    # If the object is deleted, skip it
    if obj.deleted:
        return None

    # if the label is not known, but ends with a 'group' (e.g. cargroup)
    # try to remove the s and see if that works
    # also we know that this polygon describes a group
    isGroup = False
    if ( not label in name2label ) and label.endswith('group'):
        label = label[:-len('group')]
        isGroup = True

    if label not in cls_mapping:
        return None

    labelTuple = name2label[label]
    assert labelTuple.hasInstances

    cat_info = cls_mapping[label]

    polygons = [sum(([float(pt.x), float(pt.y)] for pt in polygon), [])]
    rle = mask.frPyObjects(polygons, h, w)[0]
    bbox = tuple(float(val) for val in mask.toBbox(rle))
    area = float(mask.area(rle))
    json_rle = {key: value.decode('ascii') if isinstance(value, bytes)
                else value for key, value in rle.items()}
    # return DetAnnotation(-1, -1, cat_info.id, polygons,
    #                      area=area, bbox=bbox, iscrowd=1 if isGroup else 0)
    return DetAnnotation(-1, -1, cat_info.id, json_rle,
                         area=area, bbox=bbox, iscrowd=1 if isGroup else 0)



if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('dataroot', type=str, help='root dir of the cityscapes dataset '
        'which contains \'gtFine\' folder or \'gtCoarse\' folder')
    parser.add_argument('--with_fine', action='store_true')
    parser.add_argument('--with_coarse', action='store_true')
    parser.add_argument('--split', type=str, choices=('train', 'val', 'test'), nargs='+')
    parser.add_argument('--simplify', action='store_true')

    args = parser.parse_args()

    anno_jsons: List[str] = []
    if args.with_fine:
        anno_jsons += glob(os.path.join(args.dataroot, "gtFine", "*", "*", "*_gt*_polygons.json"))

    if args.with_coarse:
        anno_jsons += glob(os.path.join(args.dataroot, "gtCoarse", "*", "*", "*_gt*_polygons.json"))

    print('found annontations:', len(anno_jsons))
    print('split:', args.split)
    anno_jsons = sorted(anno_jsons)

    coco = COCOAnnotationFile()
    classes: List[str] = list(CityscapesDataset.CLASSES)

    cls_map = {cls: DetCategory(cls_id, cls, None)
               for cls_id, cls in enumerate(classes, start=1)}
    coco.categories.extend(cls_map.values())
    if args.simplify:
        sim_map = {
            key: 'human' if key in ('person', 'rider') else
            ('bike' if key in ('motorcycle', 'bicycle') else 'motorvehicle')
            for key in classes
        }
        cats = {cat: DetCategory(cat_id, cat, None)
                for cat_id, cat in enumerate(sorted(set(sim_map.values())), start=1)}
        coco.categories.clear()
        coco.categories.extend(cats.values())
        cls_map = {cls: cats[sim_map[cls]] for cls in classes}

    anno_counter = 1
    for img_id, anno_path in enumerate(anno_jsons, start=1):
        if all(f'/{split}/' not in anno_path for split in args.split):
            continue  # skip the split not included
        anno = Annotation()
        anno.fromJsonFile(anno_path)
        assert anno.objectType == CsObjectType.POLY

        rel_path = anno_path.replace(args.dataroot, '')
        if rel_path.startswith('/') or rel_path.startswith('\\'):
            rel_path = rel_path[1:]
        assert not rel_path.startswith('/') and not rel_path.startswith('\\')
        img_path = rel_path.replace('_polygons.json', '.png')\
            .replace('gtFine', 'leftImg8bit').replace('gtCoarse', 'leftImg8bit')

        coco.images.append(ImageWithSegm(img_id, anno.imgWidth, anno.imgHeight, img_path,
                                         segm_file=rel_path.replace('_polygons.json',
                                                                    '_labelTrainIds.png')))

        coco_annos: List[DetAnnotation] = []
        for obj in anno.objects:
            coco_anno = anno_cs2coco(obj, cls_map, anno.imgHeight, anno.imgWidth)
            if coco_anno is not None:
                coco_anno.image_id = img_id
                coco_anno.id = anno_counter
                anno_counter += 1
                # coco.annotations.append(coco_anno)
                coco_annos.append(coco_anno)

        assert len(coco_annos) < 65536
        anno_mask = np.zeros((anno.imgHeight, anno.imgWidth),
                             dtype=np.uint8 if len(coco_annos) < 256 else np.uint16)
        for idx, coco_anno in enumerate(coco_annos, start=1):
            anno_mask[mask.decode(coco_anno.segmentation).view(np.bool)] = idx
        for idx, coco_anno in enumerate(coco_annos, start=1):
            coco_anno.segmentation = {key: value.decode('ascii') if isinstance(value, bytes)
                                      else value for key, value in
                                      mask.encode((anno_mask == idx).copy('F')).items()}
            

        coco.annotations.extend(coco_annos)

    with open(f'{args.dataroot}/cityscapes_instances_' +
              '-'.join((['gtFine'] if args.with_fine else []) +
                       (['gtCoarse'] if args.with_coarse else [])) + '_' +
              ('sim_' if args.simplify else '') +
              ''.join(sorted(args.split)) + '.json', 'w') as f:
        coco.dump(f)
