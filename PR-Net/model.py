import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50
import torch.nn.functional as F
from einops import rearrange
import math


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)



class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class STA_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(STA_module, self).__init__()

        self.FC11 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC11.apply(weights_init_kaiming)
        self.FC12 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC12.apply(weights_init_kaiming)
        self.FC13 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC13.apply(weights_init_kaiming)
        self.FC1 = nn.Conv2d(channel//4, channel, kernel_size=1)
        self.FC1.apply(weights_init_kaiming)

        self.FC21 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC21.apply(weights_init_kaiming)
        self.FC22 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC22.apply(weights_init_kaiming)
        self.FC23 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC23.apply(weights_init_kaiming)
        self.FC2 = nn.Conv2d(channel//4, channel, kernel_size=1)
        self.FC2.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(p=0.01)

        self.stab = STAB(channel, stoken_size=[8,8])


    def forward(self, x):

        x = x + self.stab(x)

        # DCB
        x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x))/3
        x1 = self.FC1(F.relu(x1))
        x2 = (self.FC21(x) + self.FC22(x) + self.FC23(x))/3
        x2 = self.FC2(F.relu(x2))
        out = torch.cat((x, x1, x2), 0)
        out = self.dropout(out)
        return out


class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim),)
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0), nn.BatchNorm2d(self.high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)
        
        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        
        return z

class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(nn.Conv2d(self.low_dim//self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(high_dim),)
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        # g_x = self.g(x_l).contiguous().view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(x_l).view(B, self.low_dim, -1)
        # phi_x = self.phi(x_l).contiguous().view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim//self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        return z


class MIFE_module(nn.Module):
    def __init__(self, high_dim, low_dim, flag):
        super(MIFE_module, self).__init__()

        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)


    def forward(self, x, x0):


        h1 = self.CNL(x, x0)
        h2 = self.PNL(h1+x, x0)
        z = h1+h2

        return z



class pr_net(nn.Module):
    def __init__(self,  class_num, dataset, arch='resnet50'):
        super(pr_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        
        self.dataset = dataset
        if self.dataset == 'regdb': # For regdb dataset, we remove the MIFE3 block and layer4.
            pool_dim = 1024
            self.STA = STA_module(512)
            self.MIFE1 = MIFE_module(256, 64, 0)
            self.MIFE2 = MIFE_module(512, 256, 1)
        else:
            pool_dim = 2048
            self.STA = STA_module(1024)
            self.MIFE1 = MIFE_module(256, 64, 0)
            self.MIFE2 = MIFE_module(512, 256, 1)
            self.MIFE3 = MIFE_module(1024, 512, 1)

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.FFFT = FFFT(512, h=48, w=10)


        
    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)


        x_ = x
        x = self.base_resnet.base.layer1(x_)
        x_ = self.MIFE1(x, x_)


        x = self.base_resnet.base.layer2(x_)
        x_ = self.MIFE2(x, x_)
        if self.dataset == 'regdb':  # For regdb dataset, we remove the MIFE3 block and layer4.

            x_ = x_ + self.FFFT(x, spatial_size=(48, 18))

            x_ = self.STA(x_)

            x = self.base_resnet.base.layer3(x_)
        else:

            x_ = x_ + self.FFFT(x, spatial_size=(48, 18))

            x = self.base_resnet.base.layer3(x_)
            x_ = self.MIFE3(x, x_)


            x_ = self.STA(x_)


            x = self.base_resnet.base.layer4(x_)

        
        xp = self.avgpool(x)
        x_pool = xp.view(xp.size(0), xp.size(1))
        
        feat = self.bottleneck(x_pool)

        if self.training:
            xps = xp.view(xp.size(0), xp.size(1), xp.size(2)).permute(0, 2, 1)
            xp1, xp2, xp3 = torch.chunk(xps, 3, 0)
            xpss = torch.cat((xp2, xp3), 1)
            loss_ort = torch.triu(torch.bmm(xpss, xpss.permute(0, 2, 1)), diagonal = 1).sum() / (xp.size(0))

            return x_pool, self.classifier(feat), loss_ort
        else:
            return self.l2norm(x_pool), self.l2norm(feat)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class FFFT(nn.Module):
    def __init__(self, dim, h=96, w=19):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        h_f, w_f = x.shape[2], x.shape[3]
        x = to_3d(x)

        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)

        x = to_4d(x, h_f, w_f)

        return x

class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b * c, 1, h, w), self.weights, stride=1, padding=self.kernel_size // 2)
        return x.reshape(b, c * 9, h * w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,
                                                                                           dim=2)  # (B, num_heads, head_dim, N)

        attn = (k.transpose(-1, -2) @ q) * self.scale

        attn = attn.softmax(dim=-2)  # (B, h, N, N)
        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class STAB(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()

        self.n_iter = n_iter
        self.stoken_size = stoken_size

        self.scale = dim ** - 0.5

        self.unfold = Unfold(3)
        self.fold = Fold(3)

        self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop=attn_drop, proj_drop=proj_drop)

    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size

        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        _, _, H, W = x.shape

        hh, ww = H // h, W // w

        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww))  # (B, C, hh, ww)

        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh * ww, h * w, C)

        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale  # (B, hh*ww, h*w, 9)

                affinity_matrix = affinity_matrix.softmax(-1)  # (B, hh*ww, h*w, 9)

                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)

                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)

                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(
                        B, C, hh, ww)

                    stoken_features = stoken_features / (affinity_matrix_sum + 1e-12)  # (B, C, hh, ww)

        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)

        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(B, C, hh, ww)

        stoken_features = stoken_features / (affinity_matrix_sum.detach() + 1e-12)  # (B, C, hh, ww)

        stoken_features = self.stoken_refine(stoken_features)

        stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)  # (B, hh*ww, C, 9)

        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2)  # (B, hh*ww, C, h*w)

        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]

        return pixel_features

    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        stoken_features = self.stoken_refine(stoken_features)
        return stoken_features

    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)