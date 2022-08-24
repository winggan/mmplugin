from copy import deepcopy
from typing import Tuple, Union
from mmdet.apis.inference import init_detector
from mmcv import Config
import torch
import mmplugin.mmdet  # NOQA


@torch.jit.script
def uncompress(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return data.float().mul(scale)


def compress(data: torch.Tensor, force: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    if data.dtype == torch.float16:
        return data, data.new_ones(())
    EXP_MIN = -14
    EXP_MAX = 15
    exp = data.abs().log2_().floor_().long()
    exp_min = exp.min()
    exp_max = exp.max()

    if exp_max - exp_min <= EXP_MAX - EXP_MIN:
        exp_shift = EXP_MIN - exp_min
    else:
        if not force:
            return data, data.new_ones(())
        else:
            exp_shift = EXP_MAX - exp_max

    scale = exp_shift.to(data.dtype).exp2()
    data = data.mul(scale).half()
    return data, (-exp_shift).to(data.dtype).exp2()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('ckpt', type=str)
    parser.add_argument('outweight', type=str)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--force_fp16', action='store_true')

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    det = init_detector(cfg)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    assert isinstance(ckpt, dict)
    assert isinstance(ckpt.get('state_dict', None), dict)
    ckpt.pop('optimizer')

    if args.fp16:
        from torch.nn import Conv2d, Linear

        state = ckpt['state_dict']

        for key, submod in torch.nn.Module.named_modules(det):
            if key == '':
                continue
            if isinstance(submod, Conv2d):
                state[key + '.weight'] = compress(state[key + '.weight'], args.force_fp16)
            elif isinstance(submod, Linear):
                state[key + '.weight'] = compress(state[key + '.weight'], args.force_fp16)
            else:
                continue

        for key in state:
            if isinstance(state[key], torch.Tensor):
                state[key] = (state[key], state[key].new_ones(()))

        ckpt['state_dict_value_hook'] = uncompress.save_to_buffer()

    torch.save(ckpt, args.outweight)
