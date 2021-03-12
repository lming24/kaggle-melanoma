"""
Module to parse json config files and resume checkpoints
"""

import os
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
import logger.logger as logger
from lib.utils import read_json, write_json
import lib.env


class ConfigParser:
    """
    Class to parse configuration json file. Handles hyperparameters for training,
    initializations of modules, checkpoint saving and logging module.
    :param config: Dict containing configurations, hyperparameters for training.
        contents of `config.json` file for example.
    :param resume: String, path to the checkpoint being loaded.
    :param modification: Dict keychain:value, specifying position values to be
        replaced from config dict.
    :param run_id: Unique Identifier for training processes. Used to save checkpoints
        and training log. Timestamp is being used as default
    """
    def __init__(self, config, resume=None, modification=None, run_id=None):
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_env = os.getenv(lib.env.SAVE_ENV)
        if not save_env:
            raise ValueError("Please set environment variable SAVE_DIR before running..")

        save_dir = Path(save_env).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        exper_name = self.config['name']
        if not run_id:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%Y.%m.%d_%H.%M.%S')
        self._checkpoint_dir = save_dir / exper_name / run_id / 'models'
        self._log_dir = save_dir / exper_name / run_id / 'log'

        # make directory for saving checkpoints and log.
        self.checkpoint_dir.mkdir(parents=True, exist_ok=False)
        self.log_dir.mkdir(parents=True, exist_ok=False)
        os.environ[lib.env.CHECKPOINT_ENV] = str(self.checkpoint_dir.resolve())
        os.environ[lib.env.LOG_ENV] = str(self.log_dir.resolve())

        # save updated config file to the checkpoint dir
        write_json(self.config, self.checkpoint_dir / 'config.json')

        # configure logging module
        logger.setup_logging(self.log_dir)

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

    @classmethod
    def _init_obj(cls, module, module_cfg, *args, **kwargs):
        """
        Initializes an object from `module` using parameters from `module_cfg`
        """
        module_name = module_cfg['type']
        module_args = dict(module_cfg['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        if not isinstance(module, list):
            module = [module]
        class_constructor = None
        for mod in module:
            try:
                class_constructor = getattr(mod, module_name)
                break
            except AttributeError:
                pass
        if class_constructor is None:
            raise AttributeError(f"Attribute {module_name} could not be found in {module}")
        return class_constructor(*args, **module_args)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a class handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_cfg = self[name]
        return ConfigParser._init_obj(module, module_cfg, *args, **kwargs)

    def init_list_of_objs(self, name, module, *args, **kwargs):
        """
        Initializes a list of objects based on a list of configurations found in
        the config file under the key `name`
        """
        list_of_cfgs = self[name]
        return [ConfigParser._init_obj(module, module_cfg, *args, **kwargs) for module_cfg in list_of_cfgs]

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        if not isinstance(module, list):
            module = [module]
        fnt_pointer = None
        for mod in module:
            try:
                fnt_pointer = getattr(mod, module_name)
                break
            except AttributeError:
                pass
        if fnt_pointer is None:
            raise AttributeError(f"Attribute {name} could not be found in {module}")
        return partial(fnt_pointer, *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    # setting read-only attributes
    @property
    def config(self):
        """Get underlying config file."""
        return self._config

    @property
    def checkpoint_dir(self):
        """Get the model checkpoint directory"""
        return self._checkpoint_dir

    @property
    def log_dir(self):
        """Get the log directory"""
        return self._log_dir


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    """
    Updates the dictionary `config` with modifications from `modification` dict.
    Structure of the modification dict is {target: value} where target is a ';'
    separated list of nested keys where `value` needs to be placed.
    """
    if modification is None:
        return config

    for key, value in modification.items():
        if value is not None:
            _set_by_path(config, key, value)
    return config


def _get_opt_name(flags):
    """
    Returns the variable that an argument is stored in based on a list of flags

    e.g ['-t', '--test'] would be stored in a variable args.test
    """
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """
    Set a value in a nested object in tree by sequence of keys.

    e.g for keys = ['test1', 'test2'], this is
        equivalent to tree['test1']['test2'] = value
    """
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """
    Access a nested object in tree by sequence of keys.

    e.g. for keys = ['test1', 'test2'], it returns tree['test1']['test2']
    """
    return reduce(getitem, keys, tree)
