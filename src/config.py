# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
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
        "--config",
        type=str,
        help="Configuration file",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="ZJU-MoCap Data directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str, default=None,
        required=False,
        help="Overwrite the log directory that is specified in the config file.",
    )
    parser.add_argument(
        "--run_val", action='store_true',
    )
    parser.add_argument(
        "--fast_dev_run", action='store_true'
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

    return cfg

def save_config(dst_directory, config):
    """ Saves config file.
    Args:
        dst_directory (str): Directory to store input config.
        config (dict): Config dictionary.
    """
    pathlib.Path(dst_directory).mkdir(parents=True, exist_ok=True)
    dst_path = os.path.join(dst_directory, 'config.json')

    # save git head to ensure reproducibility
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
