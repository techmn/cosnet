import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from cosnet_modules import FSB, BEM, LayerNorm
from mmseg.models.builder import BACKBONES

@BACKBONES.register_module()
class COSNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, img_size=224,
                 depths=[3, 3, 12, 3], dim=72, expan_ratio=4, num_stages=4, s_kernel_size=[5,5,3,3], 
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, head_init_scale=1., **kwargs):
        super().__init__()
        
        self.num_stages = num_stages
        self.num_classes = num_classes
        self.dims = []
        for ii in range(self.num_stages):
            self.dims.append(dim * (2**ii))

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        
        stem = nn.Conv2d(in_chans, self.dims[0], kernel_size=5, stride=4, padding=2)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=3, stride=2, padding=1)
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(FSB(dim=self.dims[i], s_kernel_size=s_kernel_size[i], drop_path=dp_rates[cur + j],
                                        layer_scale_init_value=layer_scale_init_value, expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]

        #self.norm = nn.LayerNorm(self.dims[-1], eps=1e-6)  # Final norm layer
        #self.head = nn.Linear(self.dims[-1], num_classes)
        #self.hdr_layer = BEM(self.dims[-2])

        self.apply(self._init_weights)

        '''
        if kwargs["classifier_dropout"] is not None:
            self.head_dropout = nn.Dropout(kwargs["classifier_dropout"])
        else:
            self.head_dropout = nn.Dropout(0.)

        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        '''

    def _init_weights(self, m):

        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            cur_state_dict = self.state_dict()
            checkpoint = torch.load(pretrained, map_location="cpu")
            model_keys = list(checkpoint["state_dict"].keys())

            # remove keys whose shape doesn't match
            for key in model_keys:
                if key not in cur_state_dict:
                    continue
                if checkpoint["state_dict"][key].shape != cur_state_dict[key].shape:
                    val = checkpoint["state_dict"][key]
                    val = F.interpolate(val, size=cur_state_dict[key].shape[2:], mode='bilinear', align_corners=True)
                    #del checkpoint["state_dict"][key]
                    checkpoint["state_dict"][key] = val

            msg = self.load_state_dict(checkpoint["state_dict"], strict=False)
            print(msg)

    def forward_features(self, x):
        feats = []
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feats.append(x)

        return feats

    def forward(self, x):
        f1, f2, f3, f4 = self.forward_features(x)
        
        #x = self.norm(f4.mean([-2, -1]))
        #x = self.head(self.head_dropout(x))
        #f5 = self.hdr_layer(f3)
        #return [f1, f2, f3, f4, f5]

        return [f1, f2, f3, f4]



###############################################################################
if __name__ == "__main__":
    # check parameters of backbone
    model = COSNet(in_chans=3, num_classes=1000, img_size=224,
                 depths=[3, 3, 12, 3], dim=72, expan_ratio=4, num_stages=4, s_kernel_size=[5,5,3,3], 
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, head_init_scale=1, classifier_dropout = 0.)
    #print(model)

    def count_parameters(model):
        total_trainable_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            total_trainable_params += params
        return total_trainable_params

    total_params = count_parameters(model)

    print(f"Total Trainable Params: {round(total_params * 1e-6, 2)} M")

    ###########################################
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    input_res = (3, 224, 224)
    input = torch.ones(()).new_empty((1, *input_res), dtype=next(model.parameters()).dtype,
                                     device=next(model.parameters()).device)
    #model.eval()
    flops = FlopCountAnalysis(model, input)
    print(flop_count_table(flops))