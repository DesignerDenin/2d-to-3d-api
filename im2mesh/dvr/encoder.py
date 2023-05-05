import torch.nn as nn
from torchvision import models, transforms
from im2mesh.common import normalize_imagenet
import PIL.Image as Image

class ConvEncoder(nn.Module):
    def __init__(self, c_dim=128):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv1 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2)
        self.fc_out = nn.Linear(512, c_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        net = self.conv0(x)
        net = self.conv1(self.actvn(net))
        net = self.conv2(self.actvn(net))
        net = self.conv3(self.actvn(net))
        net = self.conv4(self.actvn(net))
        net = net.view(batch_size, 512, -1).mean(2)
        out = self.fc_out(self.actvn(net))
        return out


class Resnet18(nn.Module):
    
    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet18()
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out
    
encoder_dict = {
    'simple_conv': ConvEncoder,
    'resnet18': Resnet18,
}