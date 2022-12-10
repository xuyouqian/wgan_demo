from torch import nn

import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    """
    Input shape: (batch, in_dim)
    Output shape: (batch, 3, 64, 64)
    """

    def __init__(self, in_dim, feature_dim=64):
        super().__init__()

        # input: (batch, 100)
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            self.dconv_bn_relu(feature_dim * 8, feature_dim * 4),  # (batch, feature_dim * 16, 8, 8)
            self.dconv_bn_relu(feature_dim * 4, feature_dim * 2),  # (batch, feature_dim * 16, 16, 16)
            self.dconv_bn_relu(feature_dim * 2, feature_dim),  # (batch, feature_dim * 16, 32, 32)
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),  # double height and width
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2(y)
        y = self.l3(y)
        return y


g = Generator(64)

net_state_dict = g.state_dict()


for name,value in g.named_parameters():
    print(name)
    print(torch.mean(value).detach())
    print(torch.std(value).detach())
    print('---------------------------------')
'''

l1.0.weight
tensor(-7.5399e-06)
tensor(0.0721)
---------------------------------
l1.1.weight    有batch normal
tensor(0.9999)
tensor(0.0201)
---------------------------------
l1.1.bias
tensor(0.)
tensor(0.)
---------------------------------
l2.0.0.weight
tensor(2.4493e-06)
tensor(0.0200)
---------------------------------
l2.0.1.weight
tensor(1.0000)
tensor(0.0204)
---------------------------------
l2.0.1.bias
tensor(0.)
tensor(0.)
---------------------------------
l2.1.0.weight
tensor(3.6846e-06)
tensor(0.0200)
---------------------------------
l2.1.1.weight
tensor(0.9982)
tensor(0.0164)
---------------------------------
l2.1.1.bias
tensor(0.)
tensor(0.)
---------------------------------
l2.2.0.weight
tensor(2.6561e-05)
tensor(0.0200)
---------------------------------
l2.2.1.weight
tensor(0.9997)
tensor(0.0227)
---------------------------------
l2.2.1.bias
tensor(0.)
tensor(0.)
---------------------------------
l3.0.weight
tensor(-8.4211e-05)
tensor(0.0202)

'''