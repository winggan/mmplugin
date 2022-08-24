from pycocotools.coco import COCO
from pycocotools import mask
from dataclasses import dataclass, is_dataclass, asdict, field
from typing import List, Tuple, Optional, Union, TextIO, Dict
import json


"""
coco_anno_json{
"info": info, 
"images": [image], 
"categories": categories,
"annotations": [annotation], 
"licenses": [license],
}

image{
"id": int, 
"width": int, 
"height": int, 
"file_name": str, 
"license": int, 
"flickr_url": str, 
"coco_url": str, 
"date_captured": datetime,
}

info{
"year": int, 
"version": str, 
"description": str, 
"contributor": str, 
"url": str, 
"date_created": datetime,
}

license{
"id": int, 
"name": str, 
"url": str,
}

"""
@dataclass
class Image:
    id: int 
    width: int
    height: int
    file_name: str
    license: Optional[int] = None
    flickr_url: Optional[str] = None
    coco_url: Optional[str] = None
    date_captured: Optional[str] = None  # datetime str


@dataclass
class Info:
    year: int
    version: str
    description: str
    contributor: str
    url: str 
    date_created: str  # datetime str


@dataclass
class License:
    id: int
    name: str
    url: str


@dataclass
class BaseCategory:
    pass


@dataclass
class BaseAnnotation:
    pass


Categories = List[BaseCategory]


@dataclass
class COCOAnnotationFile:
    info: Optional[Info] = None
    images: List[Image] = field(default_factory=list)
    categories: Categories = field(default_factory=list)
    annotations: List[BaseAnnotation] = field(default_factory=list)
    licenses: List[License] = field(default_factory=list)

    def dump(self, f: TextIO):
        json.dump(self, f, cls=DataclassJSONEncoder)

    def dumps(self) -> str:
        return json.dumps(self, cls=DataclassJSONEncoder)

    def toCOCO(self, is_mm_wrapper: bool = True) -> COCO:
        try:
            from mmdet.datasets.api_wrappers import COCO as MMCOCO
        except ImportError:
            MMCOCO = None
            is_mm_wrapper = False

        coco_cls = MMCOCO if is_mm_wrapper else COCO
        ret = coco_cls()
        ret.dataset = json.loads(self.dumps())
        ret.createIndex()
        if is_mm_wrapper:
            ret.img_ann_map = ret.imgToAnns
            ret.cat_img_map = ret.catToImgs

        return ret


"""
For detection / instance segmentation
annotation{
"id": int,
"image_id": int,
"category_id": int,
"segmentation": RLE or [polygon],
"area": float,
"bbox": [x,y,width,height],
"iscrowd": 0 or 1,
}

categories[{
"id": int,
"name": str,
"supercategory": str,
}]
"""
@dataclass
class RLE:
    size: Tuple[int, int]  # h, w
    counts: str
    # Original bytes is similar to LEB128 but using
    #     6 bits/char and ascii chars 48-111, hence
    #     it is safe to ascii-decode to str

    def to_coco_rle(self) -> Dict[str, Union[str, Tuple[int, int]]]:
        return {
            'size': self.size,
            'counts': self.counts,
        }

    @staticmethod
    def from_coco_rle(rle: Dict[str, Union[
        bytes, str, Tuple[int, int]]
    ]) -> 'RLE':
        return RLE(rle['size'],
                   rle['counts'].decode('ascii')
                   if isinstance(rle['counts'], bytes)
                   else rle['counts'])


Polygon = List[float]


@dataclass
class DetAnnotation(BaseAnnotation):
    id: int
    image_id: int
    category_id: int
    segmentation: Union[RLE, List[Polygon]]
    area: float
    bbox: Tuple[float, float, float, float]  # [x, y, w, h]
    iscrowd: int  # 0 or 1,   


@dataclass
class DetCategory(BaseCategory):
    id: int
    name: str
    supercategory: Optional[str] = None


class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


def polygons2rle(polys: List[Polygon], h: int, w: int) -> RLE:
    rle_objs = mask.frPyObjects(polys, h, w)
    if len(rle_objs) > 1:
        rle_obj = mask.merge(rle_objs)
    else:
        rle_obj = rle_objs[0]
    return RLE.from_coco_rle(rle_obj)


def rle_area(rle: RLE) -> float:
    return rles_area([rle])[0]


def rles_area(rles: List[RLE]) -> List[float]:
    np_areas = mask.area([
        rle.to_coco_rle() for rle in rles
    ])
    return [float(area) for area in np_areas]


# TODO: more annotations to go
