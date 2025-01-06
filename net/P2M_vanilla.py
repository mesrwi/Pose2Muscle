import torch
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError, R2Score
import torch.nn as nn
import pytorch_lightning as pl
from .pose_embedding import PoseEmbedding
from .subject_embedding import SubjectEmbedding
from .encoder import TransformerEncoder

class Pose2Muscle(pl.LightningModule):
    def __init__(self, args):
        super(Pose2Muscle, self).__init__()

        # model
        self.pose_embedding = PoseEmbedding(args)
        self.encoder = TransformerEncoder(args)
        self.subject_embedding = nn.Linear(8, 32)
        # self.regression_head = nn.Sequential(
        #     nn.Linear(args['pose_embedding_dim'] + args['subject_embedding_dim'], 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, len(args['target_muscle']))
        # )
        self.regression_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

        # configs
        self.learning_rate = args['lr']
        self.criterion = nn.MSELoss()

    def forward(self, pose3d, subject):
        batch_size, seq_len, num_joints, channel = pose3d.size()
        pose3d = pose3d.view(batch_size, seq_len, num_joints*channel).unsqueeze(1).permute(0, 1, 3, 2)

        x_pose = self.pose_embedding(pose3d)
        x_pose = x_pose.permute(0, 2, 1)
        encoder_out = self.encoder(x_pose) # encoder_out: [batch_size, seq_len, d_model]

        x_subject = self.subject_embedding(subject)
        x_subject = x_subject.unsqueeze(1).expand(batch_size, seq_len, 32)

        x = torch.cat([encoder_out, x_subject], dim=2) # x: [batch_size, seq_len, d_model*2]
        out = self.regression_head(x) # out: [batch_size, seq_len, num_target]

        return out

    def training_step(self, batch, batch_idx):
        filename, subject, pose3d, emg_values = batch
        y_hat = self(pose3d, subject)
        loss = self.criterion(y_hat, emg_values)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        filename, subject, pose3d, emg_values = batch
        y_hat = self(pose3d, subject)
        loss = self.criterion(y_hat, emg_values)
        
        # MAE, MAPE, R2 Score
        metrics = MeanAbsoluteError(), MeanAbsolutePercentageError(), R2Score()
        mae = metrics[0](y_hat.cpu(), emg_values.cpu())
        mape = metrics[1](y_hat.cpu(), emg_values.cpu())

        self.log("val_mse", loss, prog_bar=True, sync_dist=True)
        self.log("val_mae", mae, prog_bar=True, sync_dist=True)
        self.log("val_mape", mape, prog_bar=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return self.optimizer