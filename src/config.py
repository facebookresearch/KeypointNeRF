import os
import json
import pathlib
import logging
import argparse
import subprocess

import yaml
import argcomplete

def create_parser():
    """ Create argument parser. """
    parser = argparse.ArgumentParser(description="Run KeypointNeRF.")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Configuration file",
    )
    parser.add_argument(
        "-d",
        "--data_root",
        type=str,
        default="/mnt/home/markomih/zju_mocap",
        help="ZJU-MoCap Data directory",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Set the logger verbosity level.",
    )
    parser.add_argument(
        "--vis_split", default="test",
        choices=["train", "test", "val"],
    )
    parser.add_argument(
        "--run_val", action='store_true',
    )
    parser.add_argument(
        "--fast_dev_run", action='store_true'
    )
    parser.add_argument(
        "--render_back", action='store_true'
    )
    parser.add_argument(
        "--render_front", action='store_true'
    )
    parser.add_argument(
        "--render_profile", action='store_true',
        help="Whether to render only as single front image"
    )
    parser.add_argument(
        "--render_interpolated_video", action='store_true'
    )
    parser.add_argument(
        "--render_spiral_video", action="store_true"
    )
    parser.add_argument(
        "--render_spiralcentered_video", action="store_true"
    )
    parser.add_argument(
        "--render_spiralcentered_iphonevideo", action="store_true"
    )
    parser.add_argument(
        "--model_ckpt", type=str, default=None
    )
    parser.add_argument(
        "--num_gpus",
        default=1,
        type=int,
        help="The number of GPU devices.",
    )
    argcomplete.autocomplete(parser)
    return parser


def load_cfg(path):
    """ Load configuration file.
    Args:
        path (str): model configuration file.
    """
    if path.endswith('.json'):
        with open(path, 'r') as file:
            cfg = json.load(file)
    elif path.endswith('.yml') or path.endswith('.yaml'):
        with open(path, 'r') as file:
            cfg = yaml.safe_load(file)
    else:
        raise ValueError('Invalid config file.')

    # insert global path variables
    data_root = os.environ.get('IRECON_DATA_ROOT', None)
    log_root = os.environ.get('IRECON_LOG_ROOT', None)
    if data_root is not None and 'dataset' in cfg:
        if 'data_root' in cfg['dataset']:
            cfg['dataset']['data_root'] = os.path.join(data_root, cfg['dataset']['data_root'])
    if log_root is not None and 'out_dir' in cfg:
        cfg['out_dir'] = os.path.join(log_root, cfg['out_dir'])

    return cfg


def save_config(dst_directory, config):
    """ Saves config file.
    Args:
        dst_directory (str): Directory to store input config.
        config (dict): Config dictionary.
    """
    pathlib.Path(dst_directory).mkdir(parents=True, exist_ok=True)
    dst_path = os.path.join(dst_directory, 'config.json')

    config['git_head'] = get_git_commit_head()

    # save commit head for experiment reproducibility
    with open(dst_path, 'w') as file:
        json.dump(config, file, indent=4)


def get_git_commit_head():
    """ Get git commit. """
    try:
        head = subprocess.check_output("git rev-parse HEAD", stderr=subprocess.DEVNULL, shell=True)
        return head.decode('utf-8').strip()
    except (subprocess.SubprocessError, UnicodeEncodeError):
        logger = logging.getLogger('irecon')
        logger.warning('Git commit is not saved.')
        return ''


def get_model(cfg):
    """ Instantiate reconstruction model. """
    from .model import KeypointNeRFLightningModule
    model = KeypointNeRFLightningModule.from_config(cfg, cfg.get('method', None))
    return model
