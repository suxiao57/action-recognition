import torch
from torch import nn
from torch.nn import functional as F

#非对称金字塔
class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 4, 8, 16), dimension=1):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=1):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, t, v = feats.size()
        feats = feats.view(-1, c, t)
        #feats = feats.permute(0,1,3,2)#nv,c,t -->nv,c,s
        #feats = feats.view(-1,v,t)
        priors = [((stage(feats)).reshape(n, c, -1,v)).view(n, c, -1) for stage in self.stages]
        #priors = [(((stage(feats)).reshape(n,c,v,-1)).permute(0,1,3,2)).reshape(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)#n,c,s  s=110
        return center



class NONLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=False):
        super(NONLocalBlock2D, self).__init__()

        self.dimension = 2
        self.sub_sample = sub_sample     #sub_sample决定是否池化
        self.in_channels = in_channels
        self.inter_channels = in_channels//2
        ##

        self.W = nn.ModuleList()
        self.g = nn.ModuleList()
        self.theta = nn.ModuleList()
        self.phi = nn.ModuleList()
        for i in range(2):
            self.W.append(nn.Sequential(nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                                                 kernel_size=1, stride=1, padding=0),
                                        nn.Dropout(0.5,inplace=True)
                                        ))
            self.g.append(nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                                             kernel_size=1, stride=1, padding=0),
                                        nn.Dropout(0.5,inplace=True)
                                        ))
            self.theta.append(nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0),
                              nn.Dropout(0.5, inplace=True)
                              ))
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.psp = PSPModule(sizes=(1,4,8,16))

        # conv_init(self.g)
        # conv_init(self.W)
        # conv_init(self.theta)
        # conv_init(self.phi)
        # bn_init(self.bn, 1)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)   输入的是(N*M,C,T,V)=(batch_size,C,T,V)
        :return:
        '''

        ## 时间
        batch_size = x.size(0)
        g_x_t = self.g[0](x)    #   batch_sizex(C/2)xTxV
        #g_x_t = self.pool(g_x_t)      # batch_sizex(C/2)x(T/2)x(V/2)

        g_x_t = self.psp(g_x_t)  # batch_sizex(C/2)xS
        phi_x_t = g_x_t # batch_sizex(C/2)xS
        g_x_t = g_x_t.permute(0, 2, 1)    #交换维度相当于转置     batch_sizexSx(C/2) value
        theta_x_t = self.theta[0](x)     # batch_sizex(C/2)xTxV
        theta_x_t = theta_x_t.view(batch_size, self.inter_channels, -1)  # batch_sizex(C/2)xTV
        theta_x_t = theta_x_t.permute(0, 2, 1)     #batch_sizexTVx(C/2) query
        f_t = torch.matmul(theta_x_t, phi_x_t)     #乘法   batch_sizexTVx(C/2)* batch_sizex(C/2)xS == batch_sizexTVxS
        f_t = F.softmax(f_t, dim=-2)    #归一化
        y_t = torch.matmul(f_t, g_x_t)  # batch_sizexTVxS * batch_sizexSx(C/2) = batch_sizexTVxc/2
        y_t = F.softmax(y_t, dim=-2)  # 归一化
        y_t = y_t.permute(0, 2, 1).contiguous()   # batch_sizex(C/2)xTV
        y_t = y_t.view(batch_size, self.inter_channels, *x.size()[2:])    #batch_sizex(C/2)xTxV
        y_t = self.W[0](y_t)    # batch_sizexCxTxV
        '''## 空间
        x_v = x.permute(0,1,3,2)   #现在输入是(batch_size,C,V,T)
        g_x_v = self.g[1](x_v)    # batch_sizex(C/2)xVxT
        g_x_v = self.pool(g_x_v)  # batch_sizex(C/2)x(V/2)x(T/2)
        g_x_v = self.psp(g_x_v)  # batch_sizex(C/2)xS
        g_x_v = g_x_v.permute(0, 2, 1)
        theta_x_v = self.theta[1](x_v)  # batch_sizex(C/2)xTxV
        theta_x_v = theta_x_v.view(batch_size, self.inter_channels, -1)  # batch_sizex(C/2)xVT
        theta_x_v = theta_x_v.permute(0, 2, 1)  # batch_sizexVTx(C/2)
        phi_x_v = self.theta[1](x_v)    # batch_sizex(C/2)xVxT
        phi_x_v = self.pool(phi_x_v)   # batch_sizex(C/2)x(V/2)x(T/2)
        phi_x_v = self.psp(phi_x_v)#batch_sizex(C/2)xS
        f_v = torch.matmul(theta_x_v, phi_x_v)  # 乘法   batch_sizexVTxS
        f_v = F.softmax(f_v, dim=-2)  # 归一化
        y_v = torch.matmul(f_v, g_x_v)  # batch_sizexVTxS*batch_sizex(C/2)xS
        y_v = F.softmax(y_v, dim=-2)  # 归一化
        y_v = y_v.permute(0, 2, 1).contiguous()  # batch_sizex(C/2)xVT
        y_v = y_v.view(batch_size, self.inter_channels, *x_v.size()[2:])  # batch_sizex(C/2)xVxT
        y_v = y_v.permute(0, 1, 3, 2)   # batch_sizex(C/2)xTxV
        y_v = self.W[1](y_v)  # batch_sizexCxTxV'''
        z = y_t + x
        z = self.bn(z)
        z = self.relu(z)

        return z




def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')    # 使用均匀分布填充张量
    nn.init.constant_(conv.bias, 0)                         # 用给定值val填充输入张量


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)




