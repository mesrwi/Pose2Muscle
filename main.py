import wandb
import torch
import pytorch_lightning as pl
from lightning.pytorch import loggers as pl_loggers
from data_module import DataModule
from args import args
from net.P2M_GTM import Pose2Muscle
from utils.utils import initialize_model

if __name__ == "__main__":
    wandb.init(project=args['project_name'],
               name=args['task_name'],
               config=args)
    wandb_logger = pl_loggers.WandbLogger(log_model=True)
    
    pl.seed_everything(seed=args['seed'], workers=True)
    torch.set_float32_matmul_precision('high')

    dm = DataModule(args)
    dm.setup()
    train_loader = dm.train_dataloader()
    valid_loader = dm.valid_dataloader()

    model = Pose2Muscle(args)
    initialize_model(model)
    wandb_logger.watch(model)

    trainer = pl.Trainer(max_epochs=300,
                         accelerator='auto',
                         devices=[1],
                         logger=wandb_logger,
                         log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)