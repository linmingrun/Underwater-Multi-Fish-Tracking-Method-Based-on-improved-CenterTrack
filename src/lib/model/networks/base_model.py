from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CoordAtt, self).__init__()
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平全局池化
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直全局池化
        
        mid_channels = max(8, channels // reduction)        # 保证最小通道数
        
        self.conv1 = nn.Conv2d(channels, mid_channels, 1)  # 降维
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mid_channels, channels, 1) # 水平卷积
        self.conv_w = nn.Conv2d(mid_channels, channels, 1) # 垂直卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        
        # 水平池化 + 垂直池化
        x_h = self.avg_pool_h(x)        # [B, C, H, 1]
        x_w = self.avg_pool_w(x)        # [B, C, 1, W]
        x_w = x_w.permute(0, 1, 3, 2)   # [B, C, W, 1]
        
        # 拼接并降维
        y = torch.cat([x_h, x_w], dim=2) # [B, C, H+W, 1]
        y = self.conv1(y)                # [B, mid, H+W, 1]
        y = self.bn1(y)
        y = self.act(y)
        
        # 分解为水平和垂直部分
        h_part, w_part = torch.split(y, [h, w], dim=2)
        w_part = w_part.permute(0, 1, 3, 2) # [B, mid, 1, W]
        
        # 生成注意力权重
        att_h = self.sigmoid(self.conv_h(h_part)) # [B, C, H, 1]
        att_w = self.sigmoid(self.conv_w(w_part)) # [B, C, 1, W]
        
        return identity * att_h * att_w  # 结合位置和通道信息
    

class OcclusionAwareHead(nn.Module):
    def __init__(self, in_channel=64, occ_weight=1.0):
        super().__init__()

        self.occ_conv = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        self.correction = nn.Sequential(
            nn.Conv2d(in_channel, 64, 1),
            nn.InstanceNorm2d(64),
            CoordAtt(64),  
            nn.GELU()
        )

        self.alpha = nn.Parameter(torch.tensor(1.0)) 
        self.beta = nn.Parameter(torch.tensor(1.0))   
        self.register_buffer('base_occ_weight', torch.tensor(occ_weight))

    def forward(self, x, occ_gt=None):
        # 预测遮挡图
        occ_pred = self.occ_conv(x)
        
        # 特征校正
        corrected_feat = self.correction(x) * (1 - occ_pred)
        
        losses = {}
        if occ_gt is not None:
            # 计算Focal Loss
            bce = F.binary_cross_entropy(occ_pred, occ_gt, reduction='none')
            focal = (1 - occ_pred)**2 * bce
            
            # 计算Dice Loss
            inter = (occ_pred * occ_gt).sum(dim=(1,2,3))
            union = occ_pred.sum(dim=(1,2,3)) + occ_gt.sum(dim=(1,2,3))
            dice = 1 - (2 * inter + 1e-5) / (union + 1e-5)
            
            # 自适应权重调整
            current_loss = (focal.mean() / self.alpha.detach() + 
                          dice.mean() / self.beta.detach())
            self.alpha.data = self.alpha * (self.alpha.grad + 1e-8).sign().float()
            self.beta.data = self.beta * (self.beta.grad + 1e-8).sign().float()
            
            # 总损失计算
            total_loss = (self.base_occ_weight * 
                      (self.alpha * focal.mean() + self.beta * dice.mean()))
            losses['occ_loss'] = total_loss
            
        # 返回三个值：校正特征、遮挡预测、损失字典
        return corrected_feat, occ_pred, losses


class BaseModel(nn.Module):
    
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        self.opt = opt
        if opt is not None and getattr(opt, 'head_kernel', 3) != 3:
            head_kernel = opt.head_kernel
            print('Using head kernel:', head_kernel)
        else:
            head_kernel = 3

        self.num_stacks = num_stacks
        self.heads = heads

        for head_name, num_classes in self.heads.items():
            # 特殊处理 'occ' 头部
            if head_name == 'occ':
                # 从配置中获取遮挡权重，默认为1.0
                occ_weight = getattr(opt, 'occ_weight', 1.0) if opt else 1.0
                head_module = OcclusionAwareHead(
                    in_channel=last_channel, 
                    occ_weight=occ_weight
                )
                setattr(self, head_name, head_module)
                continue
            
            # 其他头部保持原逻辑
            conv_channels = head_convs.get(head_name, [])
            if len(conv_channels) > 0:
                conv = nn.Conv2d(last_channel, conv_channels[0], 
                                 kernel_size=head_kernel, 
                                 padding=head_kernel // 2, 
                                 bias=True)
                convs = [conv]
                for in_c, out_c in zip(conv_channels[:-1], conv_channels[1:]):
                    convs.append(nn.Conv2d(in_c, out_c, kernel_size=1, bias=True))
                out_conv = nn.Conv2d(conv_channels[-1], num_classes, 
                                     kernel_size=1, bias=True)
                layers = []
                for conv_layer in convs:
                    layers += [conv_layer, nn.ReLU(inplace=True)]
                layers.append(out_conv)
                head_module = nn.Sequential(*layers)
                if 'hm' in head_name and opt is not None:
                    head_module[-1].bias.data.fill_(opt.prior_bias)
                else:
                    fill_fc_weights(head_module)
            else:
                head_module = nn.Conv2d(last_channel, num_classes, 
                                        kernel_size=1, stride=1, padding=0, bias=True)
                if 'hm' in head_name and opt is not None:
                    head_module.bias.data.fill_(opt.prior_bias)
                else:
                    fill_fc_weights(head_module)
            setattr(self, head_name, head_module)

    def forward(self, x, pre_img=None, pre_hm=None, occ_gt=None):
        if (pre_hm is not None) or (pre_img is not None):
            feats = self.imgpre2feats(x, pre_img, pre_hm)
        else:
            feats = self.img2feats(x)
            
        outputs = []
        for s in range(self.num_stacks):
            stack_feats = feats[s]
            result = {}
            for head_name in self.heads:
                head_fn = getattr(self, head_name)
                
                # 特殊处理遮挡感知头部
                if head_name == 'occ':
                    corrected_feat, occ_pred, losses = head_fn(stack_feats, occ_gt)
                    result['corrected_feat'] = corrected_feat
                    result['occ_pred'] = occ_pred
                    result.update(losses)
                # 其他头部正常处理
                else:
                    result[head_name] = head_fn(stack_feats)
                    
            outputs.append(result)
        return outputs

    def img2feats(self, x):
        raise NotImplementedError

    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
        raise NotImplementedError