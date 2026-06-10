import os
import glob
import h5py
import numpy as np

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

from stormer.data.multi_step_datamodule import MultiStepDataRandomizedModule
from stormer.models.iterative_module import GlobalForecastIterativeModule
from stormer.models.hub.stormer import Block, WeatherEmbedding
from stormer.models.callbacks.save_model_config import SaveModelConfig
from stormer.models.callbacks.training_timing import TrainingTiming

from lightning.pytorch.loggers.csv_logs import CSVLogger

try:
    from lightning.pytorch.loggers.wandb import WandbLogger
except Exception:
    WandbLogger = None

class CustomCLI(LightningCLI):
    def _instantiate_trainer(self, config, callbacks):
        key = "callbacks"
        if key in config:
            if config[key] is None:
                config[key] = []
            elif not isinstance(config[key], list):
                config[key] = [config[key]]
            config[key].extend(callbacks)
            if key in self.trainer_defaults:
                value = self.trainer_defaults[key]
                config[key] += value if isinstance(value, list) else [value]
            if self.save_config_callback and not config.get("fast_dev_run", False):
                config_callback = self.save_config_callback(
                    self._parser(self.subcommand),
                    self.config.get(str(self.subcommand), self.config),
                    **self.save_config_kwargs,
                )
                config[key].append(config_callback)
        else:
            rank_zero_warn(
                f"The `{self.trainer_class.__qualname__}` class does not expose the `{key}` argument so they will"
                " not be included."
            )
        
        if config['strategy'] == 'fsdp':
            fsdp_strategy = FSDPStrategy(
                sharding_strategy="SHARD_GRAD_OP",
                activation_checkpointing_policy={Block, WeatherEmbedding},
                auto_wrap_policy={Block, WeatherEmbedding}
            )
            config['strategy'] = fsdp_strategy
        
        return self.trainer_class(**config)

def main():
    # Set multiprocessing sharing strategy to file_system to avoid "too many open files" issues
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = CustomCLI(
        model_class=GlobalForecastIterativeModule,
        datamodule_class=MultiStepDataRandomizedModule,
        seed_everything_default=42,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)
    
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_transforms(*cli.datamodule.get_transforms())

    # Sanity check: ensure latitude vector length matches one spatial axis
    try:
        lat = cli.model.lat
        train_dir = os.path.join(cli.datamodule.hparams.root_dir, 'train')
        sample_files = sorted(glob.glob(os.path.join(train_dir, '*.h5')))
        if sample_files:
            with h5py.File(sample_files[0], 'r') as f:
                # Try to pick the first available variable present in the configured variables
                input_group = f['input']
                keys = [k for k in input_group.keys() if k != 'time']
                # Prefer a configured variable if present
                preferred = [k for k in keys if k in cli.datamodule.hparams.variables]
                key = preferred[0] if preferred else keys[0]
                H, W = input_group[key].shape[-2], input_group[key].shape[-1]
            if len(lat) not in (H, W):
                raise ValueError(
                    f"Latitude length {len(lat)} does not match sample tensor dims H={H}, W={W}."
                )
            lat_min = float(np.min(lat))
            lat_max = float(np.max(lat))
            if lat_min < -90.5 or lat_max > 90.5:
                raise ValueError(
                    f"Latitude values must be within [-90, 90], got range [{lat_min:.2f}, {lat_max:.2f}]."
                )
    except Exception as e:
        # Fail early with a clear message
        raise RuntimeError(f"Latitude/shape mismatch check failed: {e}")
    
    cli.model.set_base_intervals_and_lead_times(
        cli.datamodule.hparams.list_train_intervals,
        cli.datamodule.hparams.val_lead_times
    )

    # Handle both single logger and multiple loggers
    if isinstance(cli.trainer.logger, list):
        # Multiple loggers: get name from first logger
        logger_name = cli.trainer.logger[0].name if hasattr(cli.trainer.logger[0], 'name') else 'default'
    else:
        # Single logger
        logger_name = cli.trainer.logger.name if hasattr(cli.trainer.logger, 'name') else 'default'

    smoke_only_last = os.getenv("SMOKE_CHECKPOINT_ONLY_LAST", "0") == "1"
    for i in range(len(cli.trainer.callbacks)):
        if isinstance(cli.trainer.callbacks[i], ModelCheckpoint):
            save_top_k = cli.trainer.callbacks[i].save_top_k
            save_last = cli.trainer.callbacks[i].save_last
            save_weights_only = cli.trainer.callbacks[i].save_weights_only
            if smoke_only_last:
                save_top_k = 0
                save_last = True
                save_weights_only = False
            cli.trainer.callbacks[i] = ModelCheckpoint(
                dirpath=os.path.join(cli.trainer.default_root_dir, logger_name, 'checkpoints'),
                monitor=cli.trainer.callbacks[i].monitor,
                mode=cli.trainer.callbacks[i].mode,
                save_top_k=save_top_k,
                save_last=save_last,
                verbose=cli.trainer.callbacks[i].verbose,
                filename=cli.trainer.callbacks[i].filename,
                auto_insert_metric_name=cli.trainer.callbacks[i].auto_insert_metric_name,
                save_weights_only=save_weights_only,
            )

    # Add training timing and SaveModelConfig callbacks
    cli.trainer.callbacks.append(TrainingTiming())
    cli.trainer.callbacks.append(SaveModelConfig())

    # Handle logger replacement for single or multiple loggers
    if isinstance(cli.trainer.logger, list):
        # Multiple loggers: only replace WandbLogger instances
        new_loggers = []
        for logger in cli.trainer.logger:
            if WandbLogger is not None and isinstance(logger, WandbLogger):
                new_loggers.append(WandbLogger(
                    name=logger_name,
                    project=logger._wandb_init['project'],
                    save_dir=os.path.join(cli.trainer.default_root_dir, logger_name)
                ))
            else:
                # Keep other loggers as-is (CSVLogger, etc.)
                new_loggers.append(logger)
        cli.trainer.logger = new_loggers
    elif WandbLogger is not None and isinstance(cli.trainer.logger, WandbLogger):
        # Single WandbLogger: replace it
        cli.trainer.logger = WandbLogger(
            name=logger_name,
            project=cli.trainer.logger._wandb_init['project'],
            save_dir=os.path.join(cli.trainer.default_root_dir, logger_name)
        )
    # For single CSVLogger or other logger types, keep as-is

    ignore_existing_ckpt = os.getenv("IGNORE_EXISTING_CKPT", "0") == "1"
    if ignore_existing_ckpt:
        ckpt_resume_path = None
    elif os.path.exists(os.path.join(cli.trainer.default_root_dir, logger_name, 'checkpoints', 'last.ckpt')):
        ckpt_resume_path = os.path.join(cli.trainer.default_root_dir, logger_name, 'checkpoints', 'last.ckpt')
    else:
        ckpt_resume_path = None

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_resume_path)


if __name__ == "__main__":
    main()
