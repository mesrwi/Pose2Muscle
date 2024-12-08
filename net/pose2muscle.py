import torch
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError, R2Score
import torch.nn as nn
import pytorch_lightning as pl
from .pose_embedding import PoseEmbedding
from .encoder import TransformerEncoder

class Pose2Muscle(pl.LightningModule):
    def __init__(self, args):
        super(Pose2Muscle, self).__init__()

        # model
        self.pose_embedding = PoseEmbedding(args)
        self.encoder = TransformerEncoder(args)
        self.subject_embedding = nn.Linear(8, 128)
        self.regression_head = nn.Linear(512 + 128, 7)

        # configs
        self.learning_rate = args['lr']
        self.criterion = nn.MSELoss()

    def forward(self, pose3d, subject):
        batch_size, seq_len, num_joints, channel = pose3d.size()
        pose3d = pose3d.view(batch_size, seq_len, num_joints*channel).unsqueeze(1).permute(0, 1, 3, 2)

        x_pose = self.pose_embedding(pose3d)
        x_pose = x_pose.permute(0, 2, 1)
        encoder_out = self.encoder(x_pose)

        x_subject = self.subject_embedding(subject)
        x_subject = x_subject.unsqueeze(1).expand(batch_size, seq_len, 128)

        x = torch.cat([encoder_out, x_subject], dim=2)
        out = self.regression_head(x)

        return out

    def training_step(self, batch, batch_idx):
        subject, pose3d, emg_values = batch
        y_hat = self(pose3d, subject)
        loss = self.criterion(y_hat, emg_values)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        subject, pose3d, emg_values = batch
        y_hat = self(pose3d, subject)
        loss = self.criterion(y_hat, emg_values)
        
        # MAE, MAPE, R2 Score
        metrics = MeanAbsoluteError(), MeanAbsolutePercentageError(), R2Score()
        mae = metrics[0](y_hat.cpu(), emg_values.cpu())
        mape = metrics[1](y_hat.cpu(), emg_values.cpu())
        # r2_1 = metrics[2](y_hat[:, :, 0].cpu(), emg_values[:, :, 0].cpu())
        # r2_2 = metrics[2](y_hat[:, :, 1].cpu(), emg_values[:, :, 1].cpu())
        # r2_3 = metrics[2](y_hat[:, :, 2].cpu(), emg_values[:, :, 2].cpu())
        # r2_4 = metrics[2](y_hat[:, :, 3].cpu(), emg_values[:, :, 3].cpu())
        # r2_5 = metrics[2](y_hat[:, :, 4].cpu(), emg_values[:, :, 4].cpu())
        # r2_6 = metrics[2](y_hat[:, :, 5].cpu(), emg_values[:, :, 5].cpu())
        # r2_7 = metrics[2](y_hat[:, :, 6].cpu(), emg_values[:, :, 6].cpu())

        self.log("val_mse", loss, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        self.log("val_mape", mape, prog_bar=True)
        # self.log("val_r2_1", r2_1, prog_bar=True)
        # self.log("val_r2_2", r2_2, prog_bar=True)
        # self.log("val_r2_3", r2_3, prog_bar=True)
        # self.log("val_r2_4", r2_4, prog_bar=True)
        # self.log("val_r2_5", r2_5, prog_bar=True)
        # self.log("val_r2_6", r2_6, prog_bar=True)
        # self.log("val_r2_7", r2_7, prog_bar=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return self.optimizer