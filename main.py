import wandb
import pytorch_lightning as pl
from lightning.pytorch import loggers as pl_loggers
from data_module import DataModule
from args import args
from net.pose2muscle import Pose2Muscle

if __name__ == "__main__":
    wandb.init(project=args['project_name'],
            name=args['task_name'],
            config=args)
    wandb_logger = pl_loggers.WandbLogger(log_model=True)
    
    pl.seed_everything(seed=args['seed'], workers=True)

    dm = DataModule(args)
    dm.setup()
    train_loader = dm.train_dataloader()
    valid_loader = dm.valid_dataloader()

    model = Pose2Muscle(args)
    wandb_logger.watch(model)

    trainer = pl.Trainer(max_epochs=10,
                         accelerator='auto',
                         logger=wandb_logger,
                         log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)