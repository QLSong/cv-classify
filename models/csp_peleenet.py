import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import time
from timm.models.registry import register_model


class Conv_bn_relu(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, pad=1, use_relu=True, dilation=1):
        super(Conv_bn_relu, self).__init__()
        self.use_relu = use_relu
        if self.use_relu:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False, dilation=1),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False, dilation=1),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.convs(x)
        return out


class StemBlock(nn.Module):
    def __init__(self, inp=3, num_init_features=32, dilation=1):
        super(StemBlock, self).__init__()

        self.stem_1 = Conv_bn_relu(inp, num_init_features, 3, 2, 1, dilation=dilation)

        self.stem_2a = Conv_bn_relu(num_init_features, int(num_init_features / 2), 1, 1, 0, dilation=dilation)

        self.stem_2b = Conv_bn_relu(int(num_init_features / 2), num_init_features, 3, 2, 1, dilation=dilation)

        self.stem_2p = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stem_3 = Conv_bn_relu(num_init_features * 2, num_init_features, 1, 1, 0, dilation=dilation)

    def forward(self, x):
        stem_1_out = self.stem_1(x)

        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)

        stem_2p_out = self.stem_2p(stem_1_out)

        out = self.stem_3(torch.cat((stem_2b_out, stem_2p_out), 1))

        return out


class DenseBlock(nn.Module):
    def __init__(self, inp, inter_channel, growth_rate, dilation=1):
        super(DenseBlock, self).__init__()

        self.cb1_a = Conv_bn_relu(inp, inter_channel, 1, 1, 0, dilation=dilation)
        self.cb1_b = Conv_bn_relu(inter_channel, growth_rate, 3, 1, 1, dilation=dilation)

        self.cb2_a = Conv_bn_relu(inp, inter_channel, 1, 1, 0, dilation=dilation)
        self.cb2_b = Conv_bn_relu(inter_channel, growth_rate, 3, 1, 1, dilation=dilation)
        self.cb2_c = Conv_bn_relu(growth_rate, growth_rate, 3, 1, 1, dilation=dilation)

    def forward(self, x):
        cb1_a_out = self.cb1_a(x)
        cb1_b_out = self.cb1_b(cb1_a_out)

        cb2_a_out = self.cb2_a(x)
        cb2_b_out = self.cb2_b(cb2_a_out)
        cb2_c_out = self.cb2_c(cb2_b_out)

        out = torch.cat((x, cb1_b_out, cb2_c_out), 1)

        return out


class TransitionBlock(nn.Module):
    def __init__(self, inp, oup, with_pooling=True, dilation=1):
        super(TransitionBlock, self).__init__()
        if with_pooling:
            self.tb = nn.Sequential(Conv_bn_relu(inp, oup, 1, 1, 0, dilation=dilation),
                                    nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.tb = Conv_bn_relu(inp, oup, 1, 1, 0, dilation=dilation)

    def forward(self, x):
        out = self.tb(x)
        return out

class CSPTransitionBlock(nn.Module):
    def __init__(self, inp, oup, dilation=1):
        super(CSPTransitionBlock, self).__init__()
        self.tb = Conv_bn_relu(inp, oup, 1, 1, 0, dilation=dilation)

    def forward(self, x):
        out = self.tb(x)
        return out

class CSPPeleeNet(nn.Module):
    def __init__(self, num_classes=1000, num_init_features=32, growthRate=32, nDenseBlocks=[3, 4, 8, 6],
                 bottleneck_width=[1, 2, 4, 4], csp_p1_rate=None, **kwargs):
        super(CSPPeleeNet, self).__init__()
                
        self.csp_p1_rate = csp_p1_rate
        if self.csp_p1_rate is not None:
            assert len(nDenseBlocks) == len(bottleneck_width) == len(csp_p1_rate)

        self.stages = nn.Sequential()
        self.num_classes = num_classes
        self.num_init_features = num_init_features

        inter_channel = list()
        total_filter = list()
        dense_inp = list()
        
        if self.csp_p1_rate is not None: 
            self.csp_p1_inp = list()

        self.half_growth_rate = int(growthRate / 2)

        # building stemblock
        self.stages.add_module('stage_0', StemBlock(3, num_init_features))

        for i, b_w in enumerate(bottleneck_width):

            inter_channel.append(int(self.half_growth_rate * b_w / 4) * 4)

            if i == 0:
                total_filter.append(num_init_features + growthRate * nDenseBlocks[i])
                if self.csp_p1_rate is not None:
                    input_channels = int(self.num_init_features * (1 - self.csp_p1_rate[i]) + 0.5)
                    csp_p1_inp = self.num_init_features - input_channels
                else:
                    input_channels = self.num_init_features
            else:
                total_filter.append(total_filter[i - 1] + growthRate * nDenseBlocks[i])
                if self.csp_p1_rate is not None:
                    input_channels = int(total_filter[i - 1] * (1 - self.csp_p1_rate[i]))
                    csp_p1_inp = total_filter[i - 1] - input_channels
                else:
                    input_channels = total_filter[i - 1]
            dense_inp.append(input_channels)
            if self.csp_p1_rate is not None:
                self.csp_p1_inp.append(csp_p1_inp)
                
            if self.csp_p1_rate is not None:
                with_pooling = False
                dilation = 1
            else:
                if i == len(nDenseBlocks) - 1:
                    with_pooling = False
                else:
                    with_pooling = True
                if i == len(nDenseBlocks) - 2:
                    with_pooling = False

                if i == (len(bottleneck_width) - 1):
                    dilation = 2
                else:
                    dilation = 1
            
            # building middle stageblock
            self.stages.add_module('stage_{}'.format(i + 1), self._make_dense_transition(dense_inp[i], total_filter[i],
                                                                                        inter_channel[i],
                                                                                        nDenseBlocks[i],
                                                                                        with_pooling=with_pooling,
                                                                                        dilation=dilation))
        
        if self.csp_p1_rate is not None:
            # self.stage1_csp_tb = TransitionBlock(dense_inp, total_filter, with_pooling, dilation=dilation))
            self.stage1_csp_tb = TransitionBlock(self.csp_p1_inp[0]+total_filter[0], total_filter[0], with_pooling=True, dilation=1)
            self.stage2_csp_tb = TransitionBlock(self.csp_p1_inp[1]+total_filter[1], total_filter[1], with_pooling=True, dilation=1)
            self.stage3_csp_tb = TransitionBlock(self.csp_p1_inp[2]+total_filter[2], total_filter[2], with_pooling=True, dilation=1)
            self.stage4_csp_tb = TransitionBlock(self.csp_p1_inp[3]+total_filter[3], total_filter[3], with_pooling=False, dilation=1)  # with_pooling False -> True, is change 32 -> 64
            # self.stage5_csp_tb = TransitionBlock(self.csp_part[4]+total_filter[4], total_filter[4], with_pooling=False, dilation=1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(total_filter[len(nDenseBlocks)-1], self.num_classes)
        )

        self.return_features_num_channels = total_filter
        self._initialize_weights()
        

    def _make_dense_transition(self, dense_inp, total_filter, inter_channel, ndenseblocks, with_pooling=True,
                               dilation=1):
        layers = []

        for i in range(ndenseblocks):
            layers.append(DenseBlock(dense_inp, inter_channel, self.half_growth_rate, dilation=dilation))
            dense_inp += self.half_growth_rate * 2

        # Transition Layer without Compression
        layers.append(TransitionBlock(dense_inp, total_filter, with_pooling, dilation=dilation))

        return nn.Sequential(*layers)
    
    def _make_csp_dense_transition(self, cps_rate, dense_inp, total_filter, inter_channel, ndenseblocks, with_pooling=True,
                               dilation=1):
        layers = []

        for i in range(ndenseblocks):
            layers.append(DenseBlock(dense_inp, inter_channel, self.half_growth_rate, dilation=dilation))
            dense_inp += self.half_growth_rate * 2

        # Transition Layer without Compression
        layers.append(TransitionBlock(dense_inp, total_filter, with_pooling, dilation=dilation))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stages[0](x)
        
        x_part1, x_part2 = x[:, : self.csp_p1_inp[0], :, :], x[:, self.csp_p1_inp[0]: , :, :]
        stage1_x = self.stages[1](x_part2)
        stage1_tb = torch.cat([x_part1, stage1_x], dim=1)
        stage1_csp_tb = self.stage1_csp_tb(stage1_tb)
        
        stage1_csp_part1, stage1_csp_part2 = stage1_csp_tb[:, : self.csp_p1_inp[1], :, :], stage1_csp_tb[:, self.csp_p1_inp[1]: , :, :]
        stage2_x = self.stages[2](stage1_csp_part2)
        stage2_tb = torch.cat([stage1_csp_part1, stage2_x], dim=1)
        stage2_csp_tb = self.stage2_csp_tb(stage2_tb)
        
        stage2_csp_part1, stage2_csp_part2 = stage2_csp_tb[:, : self.csp_p1_inp[2], :, :], stage2_csp_tb[:, self.csp_p1_inp[2]: , :, :]
        stage3_x = self.stages[3](stage2_csp_part2)
        stage3_tb = torch.cat([stage2_csp_part1, stage3_x], dim=1)
        stage3_csp_tb = self.stage3_csp_tb(stage3_tb)
        
        stage3_csp_part1, stage3_csp_part2 = stage3_csp_tb[:, : self.csp_p1_inp[3], :, :], stage3_csp_tb[:, self.csp_p1_inp[3]: , :, :]
        stage4_x = self.stages[4](stage3_csp_part2)
        stage4_tb = torch.cat([stage3_csp_part1, stage4_x], dim=1)
        stage4_csp_tb = self.stage4_csp_tb(stage4_tb)
        # return stage1_csp_tb, stage2_csp_tb, stage3_csp_tb, stage4_csp_tb

        x = F.avg_pool2d(stage4_csp_tb, kernel_size=7)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')['state_dict']
        model_dict = self.state_dict()
        for i in model_dict:
            map_i = 'module.' + i

            if 'stage4_csp_tb.tb' in map_i:
                map_i = map_i.replace('stage4_csp_tb.tb.0', 'stage4_csp_tb.tb')

            if 'fc' in i or 'classifier' in i:
                continue
            
            print(i)
            if len(self.state_dict()[i].size()) != 0:
                self.state_dict()[i].copy_(param_dict[map_i])


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

@register_model
def csppeleenet(pretrained = False, **kwargs):
    return CSPPeleeNet(num_classes=1000, csp_p1_rate=[0.25, 0.25, 0.25, 0.25]).cuda()

if __name__ == '__main__':
    model = CSPPeleeNet(num_classes=1000, csp_p1_rate=[0.25, 0.25, 0.25, 0.25]).cuda()

    model.load_param("csp_pelee_0.25_model_best.pth.tar")
    
    model.cuda()
    model.eval()

    input = torch.autograd.Variable(torch.ones(1, 3, 224, 224)).cuda()
    output = model(input)
