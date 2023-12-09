import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, kl_divergence as kl
import torch.nn.functional as F
from torch_geometric.nn import APPNP, GCNConv, GATConv, MessagePassing, TransformerConv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Downsample = 1000
Downsample2 = 500

LAYER_1 = 1600
LAYER_2 = 320

SUB_1 = 64 #100 #64
SUB_2 = 32 #50 #32
class VAE(nn.Module):
    #Standard VAE class
    def __init__(self, input_dim, z_dim):
        super(VAE, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.BatchNorm1d(num_features=1000),
            nn.PReLU(),
            nn.Linear(1000, 100),
            nn.BatchNorm1d(num_features=100),
            nn.PReLU()
        )
        self.mean_enc = nn.Linear(100, z_dim)
        self.var_enc = nn.Linear(100, z_dim)
        self.Decoder = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.BatchNorm1d(num_features=100),
            # nn.LeakyReLU(0.2, True),
            nn.PReLU(),
            nn.Linear(100, 1000),
            nn.BatchNorm1d(num_features=1000),
            nn.PReLU(),
            nn.Linear(1000, input_dim),
        )
    
    def forward(self, x, no_rec=False):
        if no_rec:
            out = self.Encoder(x)
            mean = self.mean_enc(out)
            log_var = self.var_enc(out)
            return mean, log_var
        else:
            out = self.Encoder(x)
            mean = self.mean_enc(out)
            log_var = self.var_enc(out)
            z = Normal(mean, torch.exp(log_var)).rsample()
            rec = self.Decoder(z)
            return mean, log_var, z, rec

class GraceCell(nn.Module):

    def __init__(self,
                 in_dim1,
                 in_dim2,
                 out_dim,
                 projection_dim,
                 dropout=0.5,
                 tau=0.5,
                 conv_layer=GCNConv,
                 head=1,
                 cross=None):
        super(GraceCell, self).__init__()
        if conv_layer == GATConv:
            self.conv1 = conv_layer(in_dim1, out_dim * 2 // head, heads=head)
            self.conv2 = conv_layer(out_dim * 2,
                                    out_dim * 2 // head,
                                    heads=head)
            self.conv3 = conv_layer(out_dim * 2, out_dim // head, heads=head)

            self.conv4 = conv_layer(in_dim2, out_dim * 2 // head, heads=head)
            self.conv5 = conv_layer(out_dim * 2,
                                    out_dim * 2 // head,
                                    heads=head)
            self.conv6 = conv_layer(out_dim * 2, out_dim // head, heads=head)
        elif conv_layer == TransformerConv and cross == 'attr':
            self.conv1 = conv_layer(in_dim1, out_dim * 2)
            self.conv2 = conv_layer(out_dim * 2,
                                    out_dim * 2,
                                    edge_dim=1,
                                    beta=True)
            self.conv3 = conv_layer(out_dim * 2,
                                    out_dim,
                                    edge_dim=1,
                                    beta=True)

            self.conv4 = conv_layer(in_dim2, out_dim * 2)
            self.conv5 = conv_layer(out_dim * 2,
                                    out_dim * 2,
                                    edge_dim=1,
                                    beta=True)
            self.conv6 = conv_layer(out_dim * 2,
                                    out_dim,
                                    edge_dim=1,
                                    beta=True)
        else:
            self.conv1 = conv_layer(in_dim1, out_dim * 2)
            self.conv2 = conv_layer(out_dim * 2, out_dim * 2)
            self.conv3 = conv_layer(out_dim * 2, out_dim)

            self.conv4 = conv_layer(in_dim2, out_dim * 2)
            self.conv5 = conv_layer(out_dim * 2, out_dim * 2)
            self.conv6 = conv_layer(out_dim * 2, out_dim)

        if cross == 'direct':
            pass

        self.dropout = dropout
        self.tau = tau
        self.cross = cross

        self.fc1 = nn.Linear(out_dim, projection_dim)
        self.fc2 = nn.Linear(projection_dim, out_dim)

    def forward(self, x: torch.Tensor, aug_x: torch.Tensor,
                edge_index1: torch.Tensor, edge_index2: torch.Tensor):

        # because we comment the relu in the conv layer, so this forward function don't have the nonlinear brought by relu
        if self.cross == None:
            # x1 = F.relu(self.conv1(x, edge_index1))
            x1, att_1_1 = self.conv1(x,
                                     edge_index1,
                                     return_attention_weights=True)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            # x1 = F.relu(self.conv2(x1, edge_index1))
            x1, att_1_2 = self.conv2(x1,
                                     edge_index1,
                                     return_attention_weights=True)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x1 = self.conv3(x1, edge_index1)

            # x2 = F.relu(self.conv4(aug_x, edge_index2))
            x2, att_2_1 = self.conv4(aug_x,
                                     edge_index2,
                                     return_attention_weights=True)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
            # x2 = F.relu(self.conv5(x2, edge_index2))
            x2, att_2_2 = self.conv5(x2,
                                     edge_index2,
                                     return_attention_weights=True)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
            x2 = self.conv6(x2, edge_index2)
        elif self.cross == 'attr':
            x1, att_1_1 = self.conv1(x,
                                     edge_index1,
                                     return_attention_weights=True)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2, att_2_1 = self.conv4(aug_x,
                                     edge_index2,
                                     return_attention_weights=True)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)

            x1, att_1_2 = self.conv2(x1,
                                     edge_index=att_2_1[0],
                                     edge_attr=att_2_1[1],
                                     return_attention_weights=True)

            x2, att_2_2 = self.conv5(x2,
                                     edge_index=att_1_1[0],
                                     edge_attr=att_1_1[1],
                                     return_attention_weights=True)

            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x1 = self.conv3(x1, edge_index=att_2_2[0], edge_attr=att_2_2[1])

            x2 = F.dropout(x2, p=self.dropout, training=self.training)
            x2 = self.conv6(x2, edge_index=att_1_2[0], edge_attr=att_1_2[1])
        elif self.cross == 'direct':
            # x1, att_1_1 = self.conv1(x,
            #                          edge_index1,
            #                          return_attention_weights=True)
            # x1 = F.dropout(x1, p=self.dropout, training=self.training)
            # x2, att_2_1 = self.conv4(aug_x,
            #                          edge_index2,
            #                          return_attention_weights=True)
            # x2 = F.dropout(x2, p=self.dropout, training=self.training)
            pass

        else:
            raise ValueError('cross should be None, attr or direct')

        return x1, x2

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def dig_loss(self, z1: torch.Tensor, z2: torch.Tensor, labels=None):
        dot = torch.mm(z1, z2.t())
        if labels is None:
            labels = torch.eye(dot.size(0)).to(device)
        return F.binary_cross_entropy_with_logits(dot, labels)

    def loss(self, z1, z2, adj=None, mean=True):
        """
        input adj: torch.Tensor (n, n) as ground truth
        default: adj = torch.eye(n)
        """
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        # the semi_loss is original loss for grace
        # however, seemed to be not appropriate for our task

        # l1 = self.semi_loss(h1, h2)
        # l2 = self.semi_loss(h2, h1)
        l1 = self.dig_loss(h1, h2, adj)
        l2 = self.dig_loss(h2, h1, adj)

        ret = (l1 + l2) / 2
        if mean:
            ret = ret.mean()
        else:
            ret = ret.sum()
        return ret
