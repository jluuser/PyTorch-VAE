import os
import yaml
import argparse
import numpy as np
import time 
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from dataset import CurveDataModule
#from pytorch_lightning.tuner import Tuner



parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vq_vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model, config['exp_params'])

# Fix: ensure pin_memory handles both int and list cases
gpus = config['trainer_params']['gpus']
pin_memory = gpus != 0 if isinstance(gpus, int) else len(gpus) > 0

data = CurveDataModule(**config["data_params"])

data.setup()
# Create checkpoint directory at project root
project_root = Path(__file__).resolve().parent
ckpt_dir = project_root / '902_checkpoints'
ckpt_dir.mkdir(exist_ok=True)

trainer = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(
                         dirpath=str(ckpt_dir),          # save to project_root/checkpoints
                         filename='epoch{epoch:02d}',
                         save_top_k=-1,                   # no top-k
                         every_n_epochs=5,                # save every 5 epochs
                         save_on_train_epoch_end=True,    # trigger on train epoch end
                         save_last=True,                  # keep last.ckpt
                         verbose=True
                     )    
                          
                     
                 ],
                 gradient_clip_val=5.0,
                 #auto_lr_find=True,
                 **config['trainer_params'])
#trainer.tune(experiment, datamodule=data)
print(f"Using learning rate = {experiment.lr:.2e}")

#Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
#Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
start = time.time()  
trainer.fit(experiment, datamodule=data)
end = time.time()
print(f"Training finished in {(end - start)/60:.2f} minutes")  
'''
python run.py --config configs/vae.yaml
'''  