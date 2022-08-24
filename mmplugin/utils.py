import sys
from typing import List, Tuple, Any, Optional
import atexit
import subprocess
from torch.distributed.launch import main


def _get_flag(flag: str, flag_type: type, argv: List[str]) -> Tuple[Optional[Any], List[str]]:
    if len(flag) == 1:
        search = f'-{flag}'
    else:
        search = f'--{flag}'

    if search in argv:
        index = argv.index(search)
        value = flag_type(argv[index + 1])
        return value, argv[:index] + argv[index + 2:]

    elif any(value.startswith(search) and value[len(search)] == '='
             for value in argv):
        index = [idx for idx, value in enumerate(argv)
                 if value.startswith(search) and value[len(search) == '=']][0]
        value = flag_type(argv[index][len(search) + 1])
        return value, argv[:index] + argv[index + 1:]

    else:
        return None, argv


def _print_extra_help():
    print('''
EXTRA FLAGS to control multi-gpu processing via torch.disributed.launch backend:

  --nnodes NNODES       The number of nodes to use for distributed training
  --node_rank NODE_RANK
                        The rank of the node for multi-node distributed
                        training
  --nproc_per_node NPROC_PER_NODE
                        The number of processes to launch on each node, for
                        GPU training, this is recommended to be set to the
                        number of GPUs in your system so that each process can
                        be bound to a single GPU.
  --master_addr MASTER_ADDR
                        Master node (rank 0)'s address, should be either the
                        IP address or the hostname of node 0, for single node
                        multi-proc training, the --master_addr can simply be
                        127.0.0.1
  --master_port MASTER_PORT
                        Master node (rank 0)'s free port that needs to be used
                        for communication during distributed training
    ''')


def ddp_launch(exe: str, is_module: bool = False):
    ori_argv = sys.argv.copy()

    if '-h' in ori_argv or '--help' in ori_argv:
        subprocess.run([sys.executable, exe] + ori_argv[1:])
        _print_extra_help()
        sys.exit(0)

    print('origin argv:', ori_argv)

    nnodes, argv = _get_flag('nnodes', int, ori_argv)
    node_rank, argv = _get_flag('node_rank', int, argv)
    nproc_per_node, argv = _get_flag('nproc_per_node', int, argv)
    master_addr, argv = _get_flag('master_addr', str, argv)
    master_port, argv = _get_flag('master_port', int, argv)

    argv = argv[0:1] + \
        ([] if nnodes is None else ['--nnodes', str(nnodes)]) + \
        ([] if node_rank is None else ['--node_rank', str(node_rank)]) + \
        ([] if nproc_per_node is None else ['--nproc_per_node', str(nproc_per_node)]) + \
        ([] if master_addr is None else ['--master_addr', str(master_addr)]) + \
        ([] if master_port is None else ['--master_port', str(master_port)]) + \
        (['-m'] if is_module else []) + [exe] + argv[1:]

    print('processed argv:', argv)

    sys.argv = argv
    atexit.register(_print_extra_help)
    main()
    atexit.unregister(_print_extra_help)
    sys.exit(0)


def mm_check_launcher():
    if '--launcher' in sys.argv:
        idx = sys.argv.index('--launcher')
        assert 'pytorch' == sys.argv[idx + 1]
    elif any(v.startswith('--launcher') for v in sys.argv):
        for v in sys.argv:
            if v.startswith('--launcher'):
                assert v.endswith('pytorch')
    else:
        sys.argv.extend(['--launcher', 'pytorch'])
