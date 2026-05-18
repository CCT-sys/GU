
import logging
import argparse
from argparse import ArgumentParser
import json
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger



from method.hs_unlearn import llama_fuzzy

from utils import MetricTracker
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy

from method.llama_lora import llama_lora
from method.gradient_ascent import GA
from method.RMU import RMU
from method.npo_adapter import NPO_adapter
from method.llama_adapter import LlamaAdapter
from method.fisher_lora_HL import FisherLoraHingeLoss

import torch
import os
os.environ["WANDB_DISABLE_CODE"] = "true"

if __name__ == '__main__':
    # Parsing Arguments
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config is None:
        raise NameError("Include a config file in the argument please.")

    # Getting configurations
    config_path = arg_.config
    with open(config_path) as config_file:
        config = json.load(config_file)
    config = argparse.Namespace(**config)

    # Init configs that are not given
    if 'seed' not in config:
        seed = 42
    if 'privacy_method' not in config:
        config.privacy_method = None
    if 'train_sets' not in config:
        config.train_sets = ""
    if 'valid_sets' not in config:
        config.valid_sets = []
    if 'valid_subset_path' not in config:
        config.valid_subset_path = None
    if 'valid_type_path' not in config:
        config.valid_type_path = None
    if 'learning_rate' not in config:
        config.learning_rate = 5e-5
    if 'negative_loss' not in config:
        config.negative_loss = True
    if 'gradient_accumulation_steps' not in config:
        config.gradient_accumulation_steps = 1
    if 'num_train_epochs' not in config:
        config.num_train_epochs = 0
    if 'num_workers' not in config:
        config.num_workers = 0
    if 'wandb_log' not in config:
        config.wandb_log = False
    if 'strategy' not in config:
        config.strategy = None
    if 'fp16' not in config:
        config.fp16 = False
    if 'check_validation_only' not in config:
        config.check_validation_only = False
    if 'check_val_every_n_epoch' not in config:
        config.check_val_every_n_epoch = 1
    if 'target_length' not in config:
        config.target_length = None


    pl.seed_everything(seed, workers=True)
    os.makedirs("checkpoints", exist_ok=True)
    # Set console logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s (%(filename)s:%(lineno)d) : %(message)s'
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Set wandb logger
    if config.wandb_log:
        wandb_logger = WandbLogger(
            project=config.wandb_project,
            name=config.wandb_run_name,
            entity='',
            log_model=False)
    else:
        wandb_logger = None



    # Setting for pytorch lightning trainer
    precision = 16 if getattr(config, "fp16", False) else "bf16"
    train_params = dict(
        accumulate_grad_batches=config.gradient_accumulation_steps,
        accelerator='gpu',
        devices=config.ngpu,
        max_epochs=int(config.num_train_epochs),
        precision=precision,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        enable_checkpointing=False,
        logger=wandb_logger,
        strategy=config.strategy,
        num_sanity_val_steps=0,
        limit_val_batches=1,
        gradient_clip_val=1, 
        log_every_n_steps=1
    )

    trainer = pl.Trainer(**train_params)

    model = llama_fuzzy(config)
    trainer.fit(model)


    export_dir = getattr(config, "hf_export_dir", None)
    if export_dir:
       
        if trainer.is_global_zero:
            merge_lora = bool(getattr(config, "hf_export_merge_lora", True))
            export_dtype = str(getattr(config, "hf_export_dtype", "bf16"))
            model.export_hf_checkpoint(export_dir, merge_lora=merge_lora, dtype=export_dtype)
            print(f"[EXPORT] HF checkpoint saved to: {export_dir}")

    
