import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

# 6.9定稿版本
# 参考：
# arxiv 1505.04597
# arxiv 1801.05746，官方实现：https://github.com/ternaus/TernausNet
# https://blog.csdn.net/github_36923418/article/details/83273107
# pixelshuffle参考: arxiv 1609.05158

backbone ='resnet101'
# backbone = 'resnet34'
class DecoderBlock(nn.Module):
    """
    U-Net中的解码模块

    采用每个模块一个stride为1的3*3卷积加一个上采样层的形式

    上采样层可使用'deconv'、'pixelshuffle', 其中pixelshuffle必须要mid_channels=4*out_channles

    定稿采用pixelshuffle

    BN_enable控制是否存在BN,定稿设置为True
    """
    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='pixelshuffle', BN_enable=True):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable
    
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if self.BN_enable:
            self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

        if self.upsample_mode=='deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels = out_channels,

                                                kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode=='pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=2)
        if self.BN_enable:
            self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x=self.conv(x)
        if self.BN_enable:
            x=self.norm1(x)
        x=self.relu1(x)
        x=self.upsample(x)
        if self.BN_enable:
            x=self.norm2(x)
        x=self.relu2(x)
        return x

class Resnet_Unet(nn.Module):
    """
    定稿使用resnet50作为backbone

    BN_enable控制是否存在BN,定稿设置为True
    """
    def __init__(self, in_channels=1, out_channels=1,backbone='resnet101',BN_enable=True, resnet_pretrain=True):
        super().__init__()
        self.BN_enable = BN_enable
        self.in_channels = in_channels
        self.out_channels = out_channels        
        # encoder部分
        # 使用resnet34或50预定义模型，由于单通道入，因此自定义第一个conv层，同时去掉原fc层
        # 剩余网络各部分依次继承
        # 经过测试encoder取三层效果比四层更佳，因此降采样、升采样各取4次
        if backbone=='resnet34':
            resnet = models.resnet34(pretrained=False)
            filters=[64,64,128,256,512]
            print('using the resnet34 ')
        elif backbone=='resnet18':      
            resnet = models.resnet18( pretrained=False )
            filters=[64,64,128,256,512]      
            print('using the resnet18 ')            
        elif backbone=='resnet50':
            resnet = models.resnet50(pretrained=False)
            filters=[64,256,512,1024,2048]
            print('using the resnet50 ')            
        elif backbone=='resnet101':
            resnet = models.resnet101(pretrained=False)
            filters=[64,256,512,1024,2048]
            print('using the resnet101 ')            
        self.firstconv = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        # decoder部分
        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3]*4, out_channels=filters[3], BN_enable=self.BN_enable)
        self.decoder1 = DecoderBlock(in_channels=filters[3]+filters[2], mid_channels=filters[2]*4, out_channels=filters[2], BN_enable=self.BN_enable)
        self.decoder2 = DecoderBlock(in_channels=filters[2]+filters[1], mid_channels=filters[1]*4, out_channels=filters[1], BN_enable=self.BN_enable)
        self.decoder3 = DecoderBlock(in_channels=filters[1]+filters[0], mid_channels=filters[0]*4, out_channels=filters[0], BN_enable=self.BN_enable)
        if self.BN_enable:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0],out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), 
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=self.out_channels, kernel_size=1),
                nn.Sigmoid()
                )
        else:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0],out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=self.out_channels, kernel_size=1), 
                nn.Sigmoid()
                )

    def forward(self,x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        center = self.center(e3)

        d2 = self.decoder1(torch.cat([center,e2],dim=1))
        d3 = self.decoder2(torch.cat([d2,e1], dim=1))
        d4 = self.decoder3(torch.cat([d3,x], dim=1))

        return self.final(d4)


class WHOLE_NET(nn.Module):
    def __init__(self):
        super(WHOLE_NET, self).__init__()
        self.fenzi = Resnet_Unet(in_channels=1, out_channels=1)
        self.fenmu = Resnet_Unet(in_channels=1, out_channels=1)
        self.wrapped = Resnet_Unet(in_channels=3, out_channels=1)
        self.unwrapped = Resnet_Unet(in_channels=4, out_channels=1)
            

    def forward(self, x):
        xfenzi_1= self.fenzi(x)
        xfenmu_1= self.fenmu(x)
        xfusion1= torch.cat([x,xfenzi_1, xfenmu_1], dim=1)
        x_wrapped = self.wrapped(xfusion1)
        xfusion2= torch.cat([x,xfenzi_1, xfenmu_1,x_wrapped], dim=1)
        x_unwrapped = self.unwrapped(xfusion2)

        return  xfenzi_1, xfenmu_1, x_wrapped,x_unwrapped


# if __name__ == '__main__':

# # 打印模型结构和每一层的输入输出大小
#   input_size = (1,256, 256) # 模型输入张量的形状
#   torch.cuda.set_device(0)#选择GPU0进行单独训练
#   use_cuda = torch.cuda.is_available()
#   device = torch.device("cuda" if use_cuda else "cpu")#此处选了第一块GPU，也可以不选
#   input_tensor = torch.randn(1, 256, 256).cuda().to(torch.float32)
#   model=Resnet_Unet()
#   net = Resnet_Unet().to(device)
#   summary(net, input_size=input_size) 