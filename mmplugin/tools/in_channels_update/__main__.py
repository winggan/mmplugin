from enum import IntEnum
from typing import List, Dict, Any, Optional
import torch
from torch import Tensor
from itertools import groupby


class ChannelType(IntEnum):
    R = 1  # red
    G = 2  # green
    B = 3  # blue
    Y = 4  # gray


def parse_layout(layout: str) -> List[ChannelType]:
    assert all(k in ChannelType.__members__.keys()
               for k in layout.upper())
    return [ChannelType.__members__[k]
            for k in layout.upper()]


def access_by_key(state: Dict[str, Any], key_seq: Optional[str]
                  ) -> Dict[str, Tensor]:
    assert isinstance(state, dict)
    if key_seq is None:
        return state
    for key in key_seq.split(','):
        state = state[key]
        assert isinstance(state, dict)
    assert all(isinstance(v, Tensor) for v in state.values())
    assert all(isinstance(k, str) for k in state.keys())
    return state


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--input_layout', type=str)
    parser.add_argument('--output_layout', type=str)
    parser.add_argument('--access_key', type=str, default=None)

    args = parser.parse_args()

    weight = torch.load(args.input, map_location='cpu')
    state = access_by_key(weight, args.access_key)

    in_layout = parse_layout(args.input_layout)
    out_layout = parse_layout(args.output_layout)

    first_conv_cands = {k: v for k, v in state.items()
                        if v.ndim == 4 and v.shape[1] == len(in_layout)}

    assert len(first_conv_cands) == 1, \
        'cannot determine the weight of first conv: ' + \
        ','.join(first_conv_cands.keys())

    first_conv_key = next(iter(first_conv_cands.keys()))
    first_conv_weight = next(iter(first_conv_cands.values()))
    print(f'found {first_conv_key}: {first_conv_weight.shape}')

    # process state
    cnt_by_type = {t: len(list(g)) for t, g in groupby(sorted(in_layout))}






    print(f'updated shape: {updated_weight.shape}')

    state[first_conv_key] = updated_weight

    torch.save(weight, args.output)    

if __name__ == '__main__':
    main()
