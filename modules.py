import torch
import torch.nn as nn
from util import make_1ch, make_3ch
from torch.nn import init as init
import torch.nn.functional as F
import pickle

class ResidualBlockNoBN(nn.Module):

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class DSEM(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_feats=(64, 128, 256),
                 num_blocks=(2, 2, 2),
                 downscales=(2, 2, 1)):
        super(DSEM, self).__init__()

        num_stage = len(num_feats) ##4 
        
        self.conv_first = nn.Conv2d(num_in_ch, num_feats[0], 3, 1, 1)##3 --> 64
        body = list()
        for stage in range(num_stage):
            for _ in range(num_blocks[stage]):
                body.append(ResidualBlockNoBN(num_feats[stage]))
            if downscales[stage] == 1:
                if stage < num_stage - 1 and num_feats[stage] != num_feats[stage + 1]:
                    body.append(nn.Conv2d(num_feats[stage], num_feats[stage + 1], 3, 1, 1))
                continue
            elif downscales[stage] == 2:
                body.append(nn.Conv2d(num_feats[stage], num_feats[min(stage + 1, num_stage - 1)], 3, 2, 1))
            else:
                raise NotImplementedError

        self.body=nn.Sequential(*body)
        actv = nn.Sigmoid

        self.fc_degree=nn.Sequential(
                nn.Linear(num_feats[-1], 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
                actv(),
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x_out = self.conv_first(x)
        feat = self.body(x_out)
        feat = self.avg_pool(feat)
        feat = feat.squeeze(-1).squeeze(-1)
        degrees=self.fc_degree(feat).squeeze(-1)
        return degrees    


class DARM(nn.Module):
    def __init__(self, channels=3, features=128):
        super(DARM, self).__init__()
        
        # extract shallow feature
        self.head = nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1)

        # using two RCAB to extract deep feature 
        self.head1 =nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True)
        )      
        self.conv_du = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(features, features // 16, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(features // 16, features, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.head2 =nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True)
        )     
        self.conv_du2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(features, features // 16, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(features // 16, features, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        # a tail conv to generate coefficients 
        self.tail = nn.Conv2d(features, 72, kernel_size=3, stride=1, padding=1)
        
        # load basic Gaussian filters
        self.kernel = pickle.load(open('filters/kernel_72_k5.pkl', 'rb'))
        self.kernel = torch.from_numpy(self.kernel).float().view(-1, 1, 5, 5).cuda()


    def forward(self, x):
        B, C, H, W = x.size() 
    
        weight = self.head(x)
        weight = self.head1(weight)
        weight = weight * self.conv_du(weight)        
        weight = self.head2(weight) 
        weight = weight * self.conv_du2(weight) 
        weight = self.tail(weight).view(B, 1, -1, H, W)

        x_pad = F.pad(x, pad=(2, 2, 2, 2), mode='reflect')
        pad_H, pad_W = x_pad.size()[2:]
        x_pad = x_pad.view(B * 3, 1, pad_H, pad_W)
        x_com = F.conv2d(x_pad, weight=self.kernel, bias=None, stride=1, padding=0, groups=1).view(B, 3, -1, H,  W)  # 2, 3, 24, 58, 58
        out = torch.sum(weight*x_com, dim=2)

        return out

class ComponentDecConv(nn.Module):
    def __init__(self, k_path, k_size):
        super(ComponentDecConv, self).__init__()

        kernel = pickle.load(open(k_path, 'rb'))
        kernel = torch.from_numpy(kernel).float().view(-1, 1, k_size, k_size)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        out = F.conv2d(x, weight=self.weight, bias=None, stride=1, padding=0, groups=1)
        return out

        
class Generator_G(nn.Module):
    def __init__(self, scale, features=64):
        super(Generator_G, self).__init__()
        struct = [7, 5, 3, 1, 1, 1]
        self.G_kernel_size = 13
        # First layer
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=1, bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            if struct[layer] == 3: # Downsample on the first layer with kernel_size=1
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], stride=scale, bias=False)]
            else:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=struct[-1], bias=False)

    def forward(self, x):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        
        x = make_1ch(x) ##(2,3,128,128)-->(6,1,128,128)
        x = self.first_layer(x)
        x = self.feature_block(x) 
        out = self.final_layer(x)
        return make_3ch(out)


class Discriminator_D(nn.Module):

    def __init__(self, layers=7, features=64, D_kernel_size=7):
        super(Discriminator_D, self).__init__()

        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels=3, out_channels=features, kernel_size=D_kernel_size, bias=True))
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, layers - 1):
            feature_block += [nn.utils.spectral_norm(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, bias=True)),
                              nn.BatchNorm2d(features),
                              nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channels=features, out_channels=1, kernel_size=1, bias=True)),
                                         nn.Sigmoid())
        
        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = 128 - self.forward(torch.FloatTensor(torch.ones([1, 3, 128, 128]))).shape[-1]
        
    def forward(self, x):
        x = self.first_layer(x)
        x = self.feature_block(x)
        out = self.final_layer(x)
        return out



def weights_init_D(m):
    """ initialize weights of the discriminator """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif class_name.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_G(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        n = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
        m.weight.data.normal_(1/n, 1/n)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)

def weights_init_DARM(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)

def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)     









