import sys
from torch.distributed.launch import main, parse_args


if __name__ == '__main__':
    """
    enhanced version of dist_train.sh so it can be installed into
    the environment and called as __main__ directly
    """
    import atexit

    def _fail_to_determine_config_file():
        print('\n\n'
              '============= !!! NOTICE !!! =============\n'
              'failed to determine config file\n'
              'note that any parameters of mmplugin.mmseg.train OTHER THAN\n'
              'config file should be put AFTER the config file, so that\n'
              'dist_train can recognize the parameters for DDP\n')

    print('origin argv:', sys.argv)
    print('determine config file ...')
    atexit.register(_fail_to_determine_config_file)
    try_args = parse_args()
    atexit.unregister(_fail_to_determine_config_file)

    config_file = try_args.training_script
    print(f'treat {config_file} as the training config file')

    if '--launcher' in sys.argv:
        idx = sys.argv.index('--launcher')
        assert 'pytorch' == sys.argv[idx + 1]
    elif any(v.startswith('--launcher') for v in sys.argv):
        for v in sys.argv:
            if v.startswith('--launcher'):
                assert v.endswith('pytorch')
    else:
        sys.argv.extend(['--launcher', 'pytorch'])

    config_file_index = sys.argv.index(config_file)

    sys.argv = sys.argv[0:config_file_index] + \
        ['-m', 'mmplugin.mmseg.train'] + sys.argv[config_file_index:]
    print('argv:', sys.argv)

    main()
