import math
import torch
from torch import nn
from timm.models.registry import register_model

__all__ = ['darknet37', 'darknet53', 'darknet19']

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet19(nn.Module):
    def __init__(self, num_classes):
        super(Darknet19, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 32)
        self.conv1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = conv_batch(32, 64)
        self.conv2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = conv_batch(64, 128)
        self.conv3_2 = conv_batch(128, 64, 1, 0)
        self.conv3_3 = conv_batch(64, 128)
        self.conv3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = conv_batch(128, 256)
        self.conv4_2 = conv_batch(256, 128, 1, 0)
        self.conv4_3 = conv_batch(128, 256)
        self.conv4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = conv_batch(256, 512)
        self.conv5_2 = conv_batch(512, 256, 1, 0)
        self.conv5_3 = conv_batch(256, 512)
        self.conv5_4 = conv_batch(512, 256, 1, 0)
        self.conv5_5 = conv_batch(256, 512)
        self.conv5_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6_1 = conv_batch(512, 1024)
        self.conv6_2 = conv_batch(1024, 512, 1, 0)
        self.conv6_3 = conv_batch(512, 1024)
        self.conv6_4 = conv_batch(1024, 512, 1, 0)
        self.conv6_5 = conv_batch(512, 1024)

        self.conv7 = nn.Conv2d(1024, 1000, 1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._initialize_weights()

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
    
    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')['state_dict']
        model_dict = self.state_dict()
        for i in model_dict:
            map_i = 'module.' + i

            if 'fc' in i or 'classifier' in i:
                continue
            
            print(i)

            if len(self.state_dict()[i].size()) != 0:
                self.state_dict()[i].copy_(param_dict[map_i])

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_pool(out)

        out = self.conv2(out)
        out = self.conv2_pool(out)

        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.conv3_3(out)
        out = self.conv3_pool(out)

        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.conv4_3(out)
        out = self.conv4_pool(out)

        out = self.conv5_1(out)
        out = self.conv5_2(out)
        out = self.conv5_3(out)
        out = self.conv5_4(out)
        out = self.conv5_5(out)
        out = self.conv5_pool(out)

        out = self.conv6_1(out)
        out = self.conv6_2(out)
        out = self.conv6_3(out)
        out = self.conv6_4(out)
        out = self.conv6_5(out)

        out = self.conv7(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        return out


class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)
        self._initialize_weights()

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
    
    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')['state_dict']
        model_dict = self.state_dict()
        for i in model_dict:
            map_i = 'module.' + i

            if 'fc' in i or 'classifier' in i:
                continue
            
            print(i)

            if len(self.state_dict()[i].size()) != 0:
                self.state_dict()[i].copy_(param_dict[map_i])

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


class Darknet37(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet37, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=4)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=4)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)
        
        self._initialize_weights()
    
    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')['state_dict']
        model_dict = self.state_dict()

        for i in model_dict:
            map_i = 'module.' + i

            if 'fc' in i or 'classifier' in i:
                continue
            
            print(i)

            if len(self.state_dict()[i].size()) != 0:
                self.state_dict()[i].copy_(param_dict[map_i])

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        stage8 = self.residual_block3(out)
        out = self.conv5(stage8)
        stage16 = self.residual_block4(out)
        out = self.conv6(stage16)
        stage32 = self.residual_block5(out)

        out = self.global_avg_pool(stage32)
        out = out.view(-1, 1024)
        out = self.fc(out)
        return out
    
    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

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

class Darknet31(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet31, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=2)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=2)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

@register_model
def darknet31(pretrained = False, **kwargs):
    return Darknet31(DarkResidualBlock, num_classes = 1000)
    
@register_model
def darknet37(pretrained = False, **kwargs):
    return Darknet37(DarkResidualBlock, num_classes = 1000)

@register_model
def darknet53(pretrained = False, **kwargs):
    return Darknet53(DarkResidualBlock, num_classes = 1000)

@register_model
def darknet19(pretrained = False, **kwargs):
    return Darknet19(num_classes = 1000)


if __name__ == '__main__':
    # for darknet19
    model = darknet19(1000)
    model.load_param("darknet19_model_best.pth.tar")

    # for darknet37
    # model = darknet37(1000)
    # model.load_param("model_best.pth.tar")

    # for darknet53
    # model = darknet53(1000)
    # model.load_param("darknet53_model_best.pth.tar")

    model.cuda()
    model.eval()

    input = torch.autograd.Variable(torch.ones(1, 3, 224, 224)).cuda()
    output = model(input)
