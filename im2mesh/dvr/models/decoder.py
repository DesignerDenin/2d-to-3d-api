import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    ''' Decoder class.

    As discussed in the paper, we implement the OccupancyNetwork
    f and TextureField t in a single network. It consists of 5
    fully-connected ResNet blocks with ReLU activation.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of ResNet blocks
        out_dim (int): output dimension (e.g. 1 for only
            occupancy prediction or 4 for occupancy and
            RGB prediction)
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=512, leaky=False, n_blocks=5, out_dim=4):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.out_dim = out_dim

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, c=None, batchwise=True, only_occupancy=False,
                only_texture=False, **kwargs):

        assert((len(p.shape) == 3) or (len(p.shape) == 2))

        net = self.fc_p(p)
        for n in range(self.n_blocks):
            if self.c_dim != 0 and c is not None:
                net_c = self.fc_c[n](c)
                if batchwise:
                    net_c = net_c.unsqueeze(1)
                net = net + net_c

            net = self.blocks[n](net)

        out = self.fc_out(self.actvn(net))

        if only_occupancy:
            if len(p.shape) == 3:
                out = out[:, :, 0]
            elif len(p.shape) == 2:
                out = out[:, 0]
        elif only_texture:
            if len(p.shape) == 3:
                out = out[:, :, 1:4]
            elif len(p.shape) == 2:
                out = out[:, 1:4]

        out = out.squeeze(-1)
        return out


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
