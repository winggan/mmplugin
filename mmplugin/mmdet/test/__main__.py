import os.path as osp
import mmdet
from ...utils import ddp_launch, mm_check_launcher

if __name__ == '__main__':
    test_script = osp.join(
        osp.dirname(osp.abspath(mmdet.__file__)),
        '.mim', 'tools', 'test.py'
    )
    assert osp.exists(test_script)

    mm_check_launcher()

    ddp_launch(test_script)

