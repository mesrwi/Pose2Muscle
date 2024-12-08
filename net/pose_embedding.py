import torch.nn as nn
from utils.utils import get_activation

class PoseEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embedding_dim = args['pose_embedding_dim']
        self.input_dim = 17 * 3
        self.window_size = 9 # args['window_size']
        self.half_window = self.window_size // 2
        
        self.conv = nn.Conv2d(in_channels=1, 
                              out_channels=self.embedding_dim, 
                              kernel_size=(51, self.window_size), 
                              stride=(1, 1), 
                              padding=(0, self.half_window))
        
        self.norm = nn.BatchNorm1d(self.embedding_dim)
        self.activation = get_activation(args['pose_embedding_activation'])
        
    def forward(self, pose):
        x = self.conv(pose)
        x = x.squeeze(2)
        x = self.norm(x)
        x = self.activation(x)

        return x

    def __repr__(self):
        return "%s(embedding_dim=%d, input_dim=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.input_dim,
        )