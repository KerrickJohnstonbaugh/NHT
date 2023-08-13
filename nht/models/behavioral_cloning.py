import numpy as np
import torch
import torch.nn as nn
from nht.models.MLP import MLP, MultiHeadMLP, MultiHeadMLP2
import pytorch_lightning as pl


class MLP_BehavioralCloning(pl.LightningModule):
    def __init__(self,  
                u_dim=6,       
                c_dim=17,      
                hiddens=[256, 256],
                act = 'tanh',
                lr=1e-4,
                ):

        super().__init__()
        self.save_hyperparameters()

        self.n = u_dim
        self.neural_net = MLP(inputs = c_dim, hiddens=hiddens, out = self.n, activation=act)
            
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    #def get_map(self, context):
    def forward(self, context):

        out = self.neural_net(context)

        return out

    def training_step(self, train_batch, batch_idx):
        #TODO: inputs should really be formed by [conditionals, outputs], not separate value by itself
        u, context = train_batch

        u_hat = self.forward(context)

        loss = self.loss_fn(u_hat, u)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        u, context = val_batch

        u_hat = self.forward(context)

        loss = self.loss_fn(u_hat, u)

        self.log('val_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor =0.99)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'train_loss'
                    }
                }