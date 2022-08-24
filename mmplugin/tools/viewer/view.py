from typing import Dict, NamedTuple, Optional, Iterable, List, Any, Tuple
import http.server as svr
import os.path as osp
import json
from abc import ABCMeta, abstractmethod
import socketserver

# class DirsForView(NamedTuple):
#     name: str
#     product_ident: str
#     json: str
#     json_dir: str
#     image_dir: str


def get_palette(num_cls: int) -> List[int]:
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def get_web_palette(num_cls: int) -> List[str]:
    pal = get_palette(num_cls)
    return [f'#{pal[idx]:02x}{pal[idx + 1]:02x}{pal[idx + 2]:02x}'
            for idx in range(0, num_cls * 3, 3)]


class ViewerHandler(svr.BaseHTTPRequestHandler, metaclass=ABCMeta):

    REPLACE_PREFIX = '{prefix}/'
    FABRIC_JS_PATH = osp.dirname(osp.abspath(__file__)) + '/fabric.js'
    VIEW_HTML_PATH = osp.dirname(osp.abspath(__file__)) + '/view.html'
    LAYERS = '{__LAYERS__}'
    PALETTE = '{__PALETTE__}'
    DEFECTS = '{__DEFECTS__}'
    TITLE = '{__TITLE__}'
    PREV = '{__PREV__}'
    NEXT = '{__NEXT__}'
    INDEX = '{__INDEX__}'
    TOTAL = '{__TOTAL__}'

    def __init__(self, request: bytes, client_address: Tuple[str, int], server: socketserver.BaseServer) -> None:
        super().__init__(request, client_address, server)
        assert all('/' not in name for name in self.all_item_ids())

    @classmethod
    def first_item_id(cls) -> str:
        return next(iter(cls.all_item_ids()))

    @classmethod
    @abstractmethod
    def all_item_ids(cls) -> Iterable[str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def prev_item_id(cls, item_id: str) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def next_item_id(cls, item_id: str) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_index(cls, item_id: str) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def num_items(cls) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_title(cls, item_id: str) -> str:
        raise NotImplementedError

    @classmethod
    def get_layers(cls, item_id: str) -> Dict[str, str]:
        raise NotImplementedError

    @classmethod
    def get_palette(cls, item_id: str) -> Dict[str, str]:
        raise NotImplementedError

    @classmethod
    def get_shapes(cls, item_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @classmethod
    def get_item(cls, item_id: str) -> Tuple[Dict[str, str], Dict[str, str], List[Dict[str, Any]]]:
        return cls.get_layers(item_id), cls.get_palette(item_id), cls.get_shapes(item_id)

    @classmethod
    @abstractmethod
    def get_image_path(cls, filename: str) -> str:
        raise NotImplementedError

    def _send_file(self, contenttype: str, filename: str):
        self.send_response(200)
        self.send_header('Content-type', contenttype)
        self.end_headers()
        with open(filename, 'rb') as f:
            self.wfile.write(f.read())

    def do_GET(self):
        print(f'got request url: {self.path}')
        assert self.path.startswith('/')
        parts = self.path[1:].split('/', 2)
        if parts[0] == 'fabric.js':
            self._send_file('text/javascript', self.FABRIC_JS_PATH)
            return
        
        elif parts[0] == '':
            parts = [self.first_item_id()]

        elif any(parts[0] == item_id for item_id in self.all_item_ids()):
            pass

        else:
            self.send_response_only(404)
            return

        name = parts[0]

        if len(parts) == 1:
            # request a view html by name
            print(f'requesting view {name} (url: {self.path})')

            layers, palette, defects = self.get_item(name)
            layers = {key: f'{name}/image/{value}' for key, value in layers.items()}

            with open(self.VIEW_HTML_PATH, 'r') as f:
                html = f.read()

            print(html.find(self.LAYERS))

            html = html\
                .replace(self.LAYERS, json.dumps(layers))\
                .replace(self.PALETTE, json.dumps(palette))\
                .replace(self.DEFECTS, json.dumps(defects))\
                .replace(self.TITLE, self.get_title(name))\
                .replace(self.TOTAL, str(self.num_items()))\
                .replace(self.INDEX, str(self.get_index(name)))\
                .replace(self.PREV, self.prev_item_id(name))\
                .replace(self.NEXT, self.next_item_id(name))

            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))

        elif len(parts) == 2:
            self.send_response_only(404)

        elif parts[1] == 'image':
            filename = self.get_image_path(parts[2])
            self._send_file(f'image/{osp.splitext(filename)[1][1:]}', filename)

        else:
            self.send_response_only(404)
            return


if __name__ == '__main__':
    raise RuntimeError('code in __main__ are only for reference')
    from argparse import ArgumentParser

    parser = ArgumentParser(description='''
A defect viewer that support switching layers of the same viewpoint while
preserving the rendered defect instances (in polygons or rectangles with
a tag to show the score and other attributes), based on html SimpleHttpServer
with fabric.js support for frontend rendering.

For rendering a viewpoint, it needs a json file that contains information
about what and where to render the defects instance, and from which files to
load the layers.

The json files should follow the format below:
{
    "layers": layer_dict,
    "palette": palette,
    "defects": [draw_info, draw_info, ...]
}

layer_dict: { key1: value1, key2: value2, ... }
    each key/value pair representing a layer, with key to be the layer name and
    value to be the path of image file to load

palette: {key1: value1, key2, value2, ...}
    each key is the name of a type of defect, and the corresponding value is the
    color that instances of the type of defect are rendered in

draw_info: object with format below, representing a defect instance to be rendered

    draw_info: None  // no need not draw
        | caption    // caption of a viewpoint or a product
        | shape      // draw given shape on specific viewpoint
    caption: str
    shape: (tag, shape_type, points, viewpoint, category)
    tag: str  // text information attached to the shape
    shape_type: enum(Line, Rectangle, Polygon, MultiPolygon)
    points: List[float] 
        | List[List[float]]  // must meets the needs of given shape_type
    viewpoint: which viewpoint to draw on
    category: which style should be applied to this shape, current

    Note that for category currently only 'defect' and non-'defect' are supported
    ''')
    parser.add_argument('path', type=str, help='where to load the json files')
    parser.add_argument('--image_path', type=str, default='', help=
        '\'^{prefix}/\' in values in the layer_dict are replaced with this argument '
        '(plus the qrcode of the product that the view belongs to if single_view is '
        'not set)')
    parser.add_argument('--single_view', action='store_true', help=
        'if set, only view a single viewpoint and the argument \'image_path\' is '
        'intepreted as a directory contains the image files of layers, instead '
        'of a dataset directory, whose subdirectories contains thoes image files')
    parser.add_argument('--ident_regex', type=str, default=None, help=
        'the regex to extract product from the name of the json file to view, if not set '
        'extract the product by taking the first part when spliting the file name with \'_\'')
    parser.add_argument('--addr', type=str, default='127.0.0.1', help='address to bind')
    parser.add_argument('--pattern', type=str, default='*_draw.json', help=
        'file name pattern of the json files to view')
    parser.add_argument('--port', type=int, default=33693)

    args = parser.parse_args()

    ViewerHandler.set_context(args.path, args.image_path, args.single_view,
                              args.pattern, args.ident_regex)
    server = svr.HTTPServer((args.addr, args.port), ViewerHandler)
    server.serve_forever()
