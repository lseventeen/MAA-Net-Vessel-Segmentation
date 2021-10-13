import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(conv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Sequential(

            # nn.Conv2d(out_c, out_c, kernel_size=1,
            #           padding=0, bias=False),
            # nn.BatchNorm2d(out_c),
            # nn.Dropout2d(dp),
            # nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_c, out_c, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_c, out_c, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),

        )
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        if self.in_c != self.out_c:
            self.diminsh_c = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1,
                          padding=0, stride=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1, inplace=True)
                
            )

    def forward(self, x):

        if self.in_c != self.out_c:
            x = self.diminsh_c(x)
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


class block(nn.Module):
    def __init__(self, in_c, out_c,  dp=0, is_up=False, is_down=False):
        super(block, self).__init__()
        self.is_up = is_up

        self.is_down = is_down
        self.conv = conv(in_c, out_c, dp=dp)

        if self.is_up == True:
            self.up = up(out_c, out_c//2)
        if self.is_down == True:
            self.down = down(out_c, out_c*2)

    def forward(self,  x):
        x = self.conv(x)
        if self.is_up == False and self.is_down == False:
            return x

        elif self.is_up == True and self.is_down == False:
            x_up = self.up(x)
            return x, x_up
        elif self.is_up == False and self.is_down == True:
            x_down = self.down(x)
            return x, x_down
        else:
            x_up = self.up(x)
            x_down = self.down(x)
            return x, x_up, x_down


class FRNet2(nn.Module):
    def __init__(self,  num_classes=1,num_channels=1, feature_scale=2,  dropout=0.2, out_ave=True):
        super(FRNet2, self).__init__()
        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.block1_3 = block(
            num_channels, filters[0],  dp=dropout, is_up=False, is_down=True)
        self.block1_2 = block(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=True)
        self.block1_1 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True)
        self.block10 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True)
        self.block11 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True)
        self.block12 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False)
        self.block13 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False)

        self.block2_2 = block(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=True)
        self.block2_1 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=True)
        self.block20 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=True)
        self.block21 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False)
        self.block22 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False)

        self.block3_1 = block(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=True)
        self.block30 = block(
            filters[2]*2, filters[2],  dp=dropout, is_up=True, is_down=False)
        self.block31 = block(
            filters[2]*3, filters[2],  dp=dropout, is_up=True, is_down=False)

        self.block40 = block(filters[3], filters[3],
                             dp=dropout, is_up=True, is_down=False)

        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x1_3, x_down1_3 = self.block1_3(x)

        x1_2, x_down1_2 = self.block1_2(x1_3)
        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)

        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(
            torch.cat([x_down1_2, x2_2], dim=1))
        x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2)

        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        x20, x_up20, x_down20 = self.block20(
            torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))
        _, x_up40 = self.block40(x_down3_1)

        x11, x_down11 = self.block11(torch.cat([x10, x_up20], dim=1))
        x21, x_up21 = self.block21(torch.cat([x_down10, x20, x_up30], dim=1))
        _, x_up31 = self.block31(torch.cat([x_down20, x30, x_up40], dim=1))

        x12 = self.block12(torch.cat([x11, x_up21], dim=1))
        _, x_up22 = self.block22(torch.cat([x_down11, x21, x_up31], dim=1))

        x13 = self.block13(torch.cat([x12, x_up22], dim=1))
        if self.out_ave == True:
            output = (self.final1(x1_1)+self.final2(x11)+self.final3(x13))/3
        else:
            output = self.final3(x13)

        return output


# class FRNet_Fuse(nn.Module):
#     def __init__(self,  num_classes=1,num_channels=1, feature_scale=2,  dropout=0.2, out_ave=True):
#         super(FRNet_Fuse, self).__init__()
#         self.FRNet_s = FRNet(num_classes=num_classes,num_channels=num_channels, feature_scale=feature_scale,  dropout=dropout, out_ave=out_ave)
#         self.FRNet_l = FRNet(num_classes=num_classes,num_channels=num_channels, feature_scale=feature_scale,  dropout=dropout, out_ave=out_ave)
#         self.FRNet_f = FRNet(num_classes=num_classes,num_channels=num_channels*3, feature_scale=feature_scale,  dropout=dropout, out_ave=out_ave)
#     def forward(self, x):
#         s = self.FRNet_s(x)
#         l = self.FRNet_l(x)
#         f = self.FRNet_f(torch.cat([s, x, l], dim=1))
#         return f,l,s


