
import torch.nn as nn
import math
import torch


from IndicPhotoOCR.detection import east_config as cfg
from IndicPhotoOCR.detection import east_utils as utils


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            # [6, 320, 1, 1],
        ]

        # building first layer
        # assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = x.mean(3).mean(2)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet(pretrained=True, **kwargs):
    """
    Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV2()
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(cfg.pretrained_basemodel_path,map_location=torch.device('cpu'), weights_only=True)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # state_dict = torch.load(cfg.pretrained_basemodel_path)  # add map_location='cpu' if no gpu
        # model.load_state_dict(state_dict)

    return model


class East(nn.Module):
    def __init__(self):
        super(East, self).__init__()
        self.mobilenet = mobilenet(True)
        # self.si for stage i
        self.s1 = nn.Sequential(*list(self.mobilenet.children())[0][0:4])
        self.s2 = nn.Sequential(*list(self.mobilenet.children())[0][4:7])
        self.s3 = nn.Sequential(*list(self.mobilenet.children())[0][7:14])
        self.s4 = nn.Sequential(*list(self.mobilenet.children())[0][14:17])

        self.conv1 = nn.Conv2d(160+96, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128+32, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(64+24, 64, 1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv9 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv10 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear')

        # utils.init_weights([self.conv1,self.conv2,self.conv3,self.conv4,
        #                        self.conv5,self.conv6,self.conv7,self.conv8,
        #                        self.conv9,self.conv10,self.bn1,self.bn2,
        #                        self.bn3,self.bn4,self.bn5,self.bn6,self.bn7])

    def forward(self, images):
        images = utils.mean_image_subtraction(images)

        f0 = self.s1(images)
        f1 = self.s2(f0)
        f2 = self.s3(f1)
        f3 = self.s4(f2)

        # _, f = self.mobilenet(images)
        h = f3  # bs 2048 w/32 h/32
        g = (self.unpool1(h))  # bs 2048 w/16 h/16
        c = self.conv1(torch.cat((g, f2), 1))
        c = self.bn1(c)
        c = self.relu1(c)

        h = self.conv2(c)  # bs 128 w/16 h/16
        h = self.bn2(h)
        h = self.relu2(h)
        g = self.unpool2(h)  # bs 128 w/8 h/8
        c = self.conv3(torch.cat((g, f1), 1))
        c = self.bn3(c)
        c = self.relu3(c)

        h = self.conv4(c)  # bs 64 w/8 h/8
        h = self.bn4(h)
        h = self.relu4(h)
        g = self.unpool3(h)  # bs 64 w/4 h/4
        c = self.conv5(torch.cat((g, f0), 1))
        c = self.bn5(c)
        c = self.relu5(c)

        h = self.conv6(c)  # bs 32 w/4 h/4
        h = self.bn6(h)
        h = self.relu6(h)
        g = self.conv7(h)  # bs 32 w/4 h/4
        g = self.bn7(g)
        g = self.relu7(g)

        F_score = self.conv8(g)  # bs 1 w/4 h/4
        F_score = self.sigmoid1(F_score)
        geo_map = self.conv9(g)
        geo_map = self.sigmoid2(geo_map) * 512
        angle_map = self.conv10(g)
        angle_map = self.sigmoid3(angle_map)
        angle_map = (angle_map - 0.5) * math.pi / 2

        F_geometry = torch.cat((geo_map, angle_map), 1)  # bs 5 w/4 h/4

        return F_score, F_geometry


model=East()
