import torch.nn as nn
from torch import cat as cat
# from DSC import DSC
import torch
from torchsummary import summary
from torch.nn import init

' Branch0 block '
class Branch0(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch0, self).__init__()
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        self.bt0 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bt0(x0)
        return x0

' Branch1 block '
class Branch1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch1, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)#图像的尺寸不变
        self.bt1 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bt1(x1)
        return x1

' Branch2 block '
class Branch2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch2, self).__init__()
        self.conv2_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt2_1 = nn.BatchNorm2d(out_ch)
        self.rl2_1 = nn.LeakyReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt2_2 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x2 = self.conv2_1(x)
        x2 = self.bt2_1(x2)
        x2 = self.rl2_1(x2)
        x2 = self.conv2_2(x2)
        x2 = self.bt2_2(x2)
        return x2

' Branch3 block '
class Branch3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch3, self).__init__()
        self.conv3_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt3_1 = nn.BatchNorm2d(out_ch)
        self.rl3_1 = nn.LeakyReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt3_2 = nn.BatchNorm2d(out_ch)
        self.rl3_2 = nn.LeakyReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt3_3 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x3 = self.conv3_1(x)
        x3 = self.bt3_1(x3)
        x3 = self.rl3_1(x3)
        x3 = self.conv3_2(x3)
        x3 = self.bt3_2(x3)
        x3 = self.rl3_2(x3)
        x3 = self.conv3_3(x3)
        x3 = self.bt3_3(x3)
        return x3

' Branch4 block '
class Branch4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch4, self).__init__()
        self.conv4_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_1 = nn.BatchNorm2d(out_ch)
        self.rl4_1 = nn.LeakyReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_2 = nn.BatchNorm2d(out_ch)
        self.rl4_2 = nn.LeakyReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_3 = nn.BatchNorm2d(out_ch)
        self.rl4_3 = nn.LeakyReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_4 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x4 = self.conv4_1(x)
        x4 = self.bt4_1(x4)
        x4 = self.rl4_1(x4)
        x4 = self.conv4_2(x4)
        x4 = self.bt4_2(x4)
        x4 = self.rl4_2(x4)
        x4 = self.conv4_3(x4)
        x4 = self.bt4_3(x4)
        x4 = self.rl4_3(x4)
        x4 = self.conv4_4(x4)
        x4 = self.bt4_4(x4)
        return x4

' Residual block with Inception module '
class ResB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResB, self).__init__()
        self.branch0 = Branch0(in_ch, out_ch)
        self.branch1 = Branch1(in_ch, out_ch // 4)
        self.branch2 = Branch2(in_ch, out_ch // 4)
        self.branch3 = Branch3(in_ch, out_ch // 4)
        self.branch4 = Branch4(in_ch, out_ch // 4)
        self.rl = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = cat((x1, x2, x3, x4), dim=1)
        x6 =  x0 + x5
        x7= self.rl(x6)
        return  x7

' Downsampling block '
class DownB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownB, self).__init__()
        self.res = ResB(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
        x1 = self.res(x)
        x2 = self.pool(x1)
        return x2, x1

' Upsampling block '
class UpB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpB, self).__init__()
        self.up = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=3, stride=2, padding = 1, output_padding = 1 )
        self.res = ResB(out_ch*2, out_ch)
    def forward(self, x, x_):
        x1 = self.up(x)
        x2 = cat((x1 , x_), dim=1)
        x3 = self.res(x2)
        return x3

' Output layer '#将最后一层变成融合层
class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
       # self.Sigmoid = nn.Sigmoid()
    def forward(self, x):
        x1 = self.conv(x)
  
        return x1

' Architecture of Res-UNet '

class UNet(nn.Module):
    def __init__(self, x=1, y=1):
        super(UNet, self).__init__()
        self.down1 = DownB(x, 8)# [-1, 8, 256, 256]
        self.down12 = DownB(8, 64) #[-1, 64, 128, 128]
        self.down2 = DownB(64, 128)#[-1, 128, 64, 64] 
        self.down3 = DownB(128, 256)# [-1, 256, 32, 32]  
        self.down4 = DownB(256, 512)#[-1, 512, 16, 16]
        self.res = ResB(512, 1024)# [-1, 1024, 8, 8]  
        # self.dsc=DSC(1024)   

        self.up1 = UpB(1024, 512)
        self.up2 = UpB(512, 256)
        self.up3 = UpB(256, 128)
        self.up4 = UpB(128, 64)
        self.up34 = UpB(64, 8)
        self.outc = Outconv(8, y)##返回2个channel,channel1是主路depth,channel2是支路1,wrapped_low

    def forward(self, x):
        x1, x1_ = self.down1(x)
        x12, x12_ = self.down12(x1)
        x2, x2_ = self.down2(x12)
        x3, x3_ = self.down3(x2)
        x4, x4_ = self.down4(x3)
        x5  = self.res(x4)
        # x55=self.dsc(x5)

        x6  = self.up1(x5, x4_)
        x7  = self.up2(x6, x3_)
        x8  = self.up3(x7, x2_)
        x9  = self.up4(x8, x12_)
        x9_10  = self.up34(x9, x1_)
        x10 = self.outc(x9_10)# [-1, 2, 256, 256] 


        return  x10 



class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),

            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAM(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x*self.ca(x)
        out = out*self.sa(out)
        return out + residual



class WHOLE_NET(nn.Module):
    def __init__(self):
        super(WHOLE_NET, self).__init__()
        self.fenzi = UNet(1, 1)
        self.fenmu = UNet(1, 1)

        self.unwrapped = UNet(3, 1)
            

    def forward(self, x):
        xfenzi_1= self.fenzi(x)
        xfenmu_1= self.fenmu(x)
        xfusion2= torch.cat([x,xfenzi_1, xfenmu_1], dim=1)
        x_unwrapped = self.unwrapped(xfusion2)

        return  xfenzi_1, xfenmu_1,x_unwrapped
    
if __name__ == '__main__':

# 打印模型结构和每一层的输入输出大小
  input_size = (1,256, 256) # 模型输入张量的形状
  torch.cuda.set_device(0)#选择GPU0进行单独训练
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")#此处选了第一块GPU，也可以不选
#   input_tensor = torch.randn(1, 256, 256).cuda().to(torch.float32)
  input_tensor = torch.randn(1, 1, 256, 256).to(device)
  model=WHOLE_NET()
  net = model.to(device)
  x_unwrapped,xfenzi_1, xfenmu_1= net(input_tensor) 

#   summary(net, input_size=input_size) 
  print(x_unwrapped.size())