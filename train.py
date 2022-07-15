# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch

from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import Trainer, loggers

from src import config


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly(True)
    
    # load configuration
    parser = config.create_parser()
    args = parser.parse_args(None)
    cfg = config.load_cfg(args.config)
    cfg['dataset']['data_root'] = args.data_root
    if args.out_dir is not None:
        cfg['out_dir'] = args.out_dir

    cfg['expname'] = cfg.get('expname', 'default')
    config.save_config(os.path.join(cfg['out_dir'], cfg['expname']), cfg)

    # create model
    model = config.get_model(cfg)

    val_key = cfg["training"].get("model_selection_metric", 'val_PSNR')
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{cfg["out_dir"]}/{cfg["expname"]}/ckpts/',
        filename='model-{epoch:04d}-{%s:.4f}' % val_key,
        verbose=True,
        monitor=val_key,
        mode=cfg["training"].get("model_selection_mode", 'max'),
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=True,
    )
    last_ckpt = os.path.join(checkpoint_callback.dirpath, f"{checkpoint_callback.CHECKPOINT_NAME_LAST}.ckpt")
    if not os.path.exists(last_ckpt):
        last_ckpt = None
    if args.model_ckpt is not None:  # overwrite last ckpt if specified model path
        last_ckpt = args.model_ckpt

    resume_from_checkpoint = cfg.get('resume_from_checkpoint', last_ckpt)

    # create trainer
    logger = loggers.TestTubeLogger(
        save_dir=cfg["out_dir"],
        name=cfg['expname'],
        debug=False,
        create_git_tag=False
    )
    trainer = Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=1)],
        resume_from_checkpoint=resume_from_checkpoint,
        logger=logger,
        gpus=args.num_gpus,
        num_sanity_val_steps=0,
        benchmark=True,
        detect_anomaly=True,
        # terminate_on_nan=False,
        accumulate_grad_batches=cfg["training"].get("accumulate_grad_batches", 1),
        fast_dev_run=args.fast_dev_run,
        strategy="ddp" if args.num_gpus != 1 else None,
        **cfg["training"].get('pl_cfg', {})
    )

    # run training
    if args.run_val:
        trainer.test(model, ckpt_path=resume_from_checkpoint, verbose=True)
    else:
        trainer.fit(model)
        model.save_ckpt()


