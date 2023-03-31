import torch
import torch.nn as nn
from torch.nn import init

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_ReLU=False, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_ReLU:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.InstanceNorm2d(out_size),
                                     nn.LeakyReLU(0.2, inplace=True)
                                    )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.InstanceNorm2d(out_size),
                                     nn.LeakyReLU(0.2, inplace=True)
                                    )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, is_ReLU=True)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class SKFusion(nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super(SKFusion, self).__init__()
		
		self.height = height
		d = max(int(dim/reduction), 4)
		
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.mlp = nn.Sequential(
			nn.Conv2d(dim, d, 1, bias=False), 
			nn.LeakyReLU(),
			nn.Conv2d(d, dim*height, 1, bias=False)
		)
		
		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats):
		B, C, H, W = in_feats[0].shape
		
		in_feats = torch.cat(in_feats, dim=1)
		in_feats = in_feats.view(B, self.height, C, H, W)
		
		feats_sum = torch.sum(in_feats, dim=1)
		attn = self.mlp(self.avg_pool(feats_sum))
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		out = torch.sum(in_feats*attn, dim=1)
		return out  

class UNet_Nested(nn.Module):

    def __init__(self, in_channels=3, n_classes=3, feature_scale=2, is_deconv=True, is_ReLU=False, is_ds=True):
        super(UNet_Nested, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_ReLU = is_ReLU
        self.is_ds = is_ds

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Skfusion
        self.SkFusion3 = SKFusion(filters[3], height=2, reduction=8)
        self.SkFusion2 = SKFusion(filters[2], height=2, reduction=8)
        self.SkFusion1 = SKFusion(filters[1], height=2, reduction=8)
        self.SkFusion0 = SKFusion(filters[0], height=2, reduction=8)

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_ReLU)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_ReLU)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_ReLU)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_ReLU)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_ReLU, n=4)

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 7, 1, 3)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 7, 1, 3)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 7, 1, 3)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 7, 1, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(filters[-1], 3),
                                nn.Tanh())

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)  
        maxpool0 = self.maxpool(X_00)  
        X_10 = self.conv10(maxpool0)  
        maxpool1 = self.maxpool(X_10)  
        X_20 = self.conv20(maxpool1)  
        maxpool2 = self.maxpool(X_20)  
        X_30 = self.conv30(maxpool2)  
        maxpool3 = self.maxpool(X_30)  
        X_40 = self.conv40(maxpool3)  

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        X_31 = self.SkFusion3([X_30, X_31])
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        X_22 = self.SkFusion2([X_21, X_22])
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        X_13 = self.SkFusion1([X_12, X_13])
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)
        X_04 = self.SkFusion0([X_03, X_04])

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        A = self.avg_pool(X_40)
        A = A.view(A.size(0), -1)
        A = self.fc(A).unsqueeze(2).unsqueeze(3)

        result4 = final_4 * inputs + (1 - final_4) * A
        result3 = final_3 * inputs + (1 - final_3) * A
        result2 = final_2 * inputs + (1 - final_2) * A
        result1 = final_1 * inputs + (1 - final_1) * A
        
        result = (result1 + result2 + result3 + result4) / 4

        return result

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi + x
        return out

if __name__ == '__main__':
    model = UNet_Nested()
    print(model)
    


    x = torch.randn(1, 3, 400, 400)
    y = model(x)