import torch
import torch.nn as nn
from models.att.da_att import CPAM

class conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(conv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        self.conv = nn.Sequential(

            nn.Conv2d(out_c, out_c, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_c, out_c, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),

        )    
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        res = x
        x = self.conv(x)
        out = x + res
        out = self.relu(out)
        return x

class up(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
                               padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=False),
            
        )

    def forward(self, x):
        x = self.up(x)
        return x


class down(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2,
                      padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True)
           
        )

    def forward(self, x):
        x = self.down(x)
        return x


class encoder(nn.Module):
    def __init__(self, out_c, dp=0):
        super(encoder, self).__init__()
        
        self.conv = conv(out_c, out_c, dp=dp)
        self.down = down(out_c, out_c*2)

    def forward(self,  x):
        x = self.conv(x)
        x_down = self.down(x)
        return x, x_down


class decoder(nn.Module):
    def __init__(self, out_c, fuse_n=2, dp=0.2, is_up = True, is_CPAM = True):
        super(decoder, self).__init__()
        self.is_up = is_up
        self.fuse_n = fuse_n
        self.is_CPAM = is_CPAM
        assert self.fuse_n > 0 and self.fuse_n <= 4
        if self.fuse_n > 1:
            self.fuse = nn.Sequential(
                nn.Conv2d(out_c*self.fuse_n, out_c, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1, inplace=True)
            )
        if self.is_CPAM is True:
            self.cpam = CPAM(out_c,1)
        self.conv = conv(out_c, out_c, dp=dp)
        if is_up is True:
            self.up = up(out_c,out_c//2)
        
    def forward(self,  x):
        if self.fuse_n >=2:
            x = self.fuse(x)
    
        output = self.conv(x)
        if self.is_CPAM is True:
            output = self.cpam(output)
       
        if self.is_up is True:
            up = self.up(output)
        else:
            up = None
        
        return up, output
class MAS(nn.Module):
    def __init__(self, in_c,class_num,dp=0):
        super(MAS,self).__init__()
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.up4 = up(in_c,in_c//2)
        self.up3 = up(in_c//2,in_c//4)
        self.up2 = up(in_c//4,in_c//8)
        self.conv4 = nn.Sequential(

            nn.Conv2d(in_c, in_c, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv3 = nn.Sequential(

            nn.Conv2d(in_c//2, in_c//2, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(in_c//2),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.sv3 = up(in_c//2,in_c//4)
        self.conv2 = nn.Sequential(

            nn.Conv2d(in_c//4, in_c//4, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(in_c//4),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True)
        )


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c//8, in_c//8, kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(in_c//8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dp),
            nn.Conv2d(in_c//8,class_num, kernel_size=1,
                      padding=0, stride=1, bias=True)
           
        )
    def forward(self,x4,x3,x2,x1):
        x = self.conv4(x4)
        x = self.up4(x)*self.gamma3 + x3 
        x = self.conv3(x)
        x = self.up3(x)*self.gamma2+x2
        x = self.conv2(x)
        x = self.up2(x)*self.gamma1+x1
        x = self.conv1(x)
        return x



class MAA_Net(nn.Module):
    def __init__(self,  num_classes=1,num_channels=1, feature_scale=2,  dropout=0.2):
        super(MAA_Net, self).__init__()
        # self.out_ave = out_ave
        filters = [64, 128, 256, 512]
        filters = [int(x / feature_scale) for x in filters]

        
        self.first_conv = nn.Sequential(
            nn.Conv2d(num_channels, filters[0], kernel_size=1,
                      padding=0, stride=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.encoder1 = encoder(filters[0],dp=dropout)
        self.encoder2 = encoder(filters[1],dp=dropout)
        self.encoder3 = encoder(filters[2],dp=dropout)
        self.middle = decoder(filters[3], fuse_n=1, dp=dropout)


        self.decoder3 = decoder(filters[2], fuse_n=2, dp=dropout)
        self.decoder2 = decoder(filters[1], fuse_n=2, dp=dropout)
        self.decoder1 = decoder(filters[0], fuse_n=2, dp=dropout,is_up = False)
       
        self.mas = MAS(filters[3],num_classes,dp =dropout)
       
    def forward(self, x):
        x = self.first_conv(x)
        x1,d1=self.encoder1(x)
        x2,d2=self.encoder2(d1)
        x3,d3=self.encoder3(d2)
     
        up4,output4 = self.middle(d3)
        up3,output3 = self.decoder3(torch.cat([up4, x3], dim=1))
        up2,output2 = self.decoder2(torch.cat([up3, x2], dim=1))
        _,output1 = self.decoder1(torch.cat([up2,x1], dim=1))
        
        output = self.mas(output4,output3,output2,output1)
       

        return output
        
        




        
    



