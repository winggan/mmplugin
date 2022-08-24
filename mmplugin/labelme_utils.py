import json
from typing import Tuple, Dict, Any, List, NamedTuple, Optional
import numpy as np


class Shape(NamedTuple):
    label: str
    points: List[List[float]]
    group_id: Optional[str]
    shape_type: str
    flags: Dict[str, str]


SUPPORTED_SHAPE_TYPES = {
    'polygon',
    'linestrip',
    'rectangle',
    'circle',
}


def parse_grid(grid_str: str) -> Tuple[int, int]:
    ret = tuple(int(v) for v in grid_str.split('x')[0:2])
    assert len(ret) == 2 and ret[0] > 0 and ret[1] > 0
    return ret


def get_shapes(data: Dict[str, Any]) -> List[Shape]:
    assert isinstance(data, dict)
    assert 'shapes' in data and isinstance(data['shapes'], list)
    shapes = [Shape(**s) for s in data['shapes']]
    assert all(s.shape_type in SUPPORTED_SHAPE_TYPES and
               all(len(t) == 2 for t in s.points) for s in shapes)
    return shapes


def reset_shapes(data: Dict[str, Any], shapes: List[Shape]) -> None:
    assert isinstance(data, dict)
    assert 'shapes' in data and isinstance(data['shapes'], list)
    data['shapes'].clear()
    data['shapes'].extend(s._asdict() for s in shapes)


def get_image_size(data: Dict[str, Any]) -> Tuple[int, int]:
    assert isinstance(data, dict)
    h = data.get("imageHeight")
    w = data.get("imageWidth")
    assert h is not None and h > 0
    assert w is not None and w > 0
    return h, w


def get_image_path(data: Dict[str, Any]) -> str:
    assert isinstance(data, dict)
    path = data.get('imagePath')
    assert isinstance(path, str) and len(path) > 0
    return path


def load_json(path: str) -> Any:
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(data, f)
