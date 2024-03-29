# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import Normalize


def denormalize(x):
    """Reverses the imagenet normalization applied to the input.

    Args:
        x (torch.Tensor - shape(N,3,H,W)): input tensor

    Returns:
        torch.Tensor - shape(N,3,H,W): Denormalized input
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook



class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
    ):
        """Init.
        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        # print("Params passed to Resize transform:")
        # print("\twidth: ", width)
        # print("\theight: ", height)
        # print("\tresize_target: ", resize_target)
        # print("\tkeep_aspect_ratio: ", keep_aspect_ratio)
        # print("\tensure_multiple_of: ", ensure_multiple_of)
        # print("\tresize_method: ", resize_method)

        self.__width = width
        self.__height = height

        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(
                f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, x):
        width, height = self.get_size(*x.shape[-2:][::-1])
        return nn.functional.interpolate(x, (int(height), int(width)), mode='bilinear', align_corners=True)

class AdaIn(nn.Module):
    def __init__(self, latent_size, out_channels):
        super(AdaIn, self).__init__()
        # self.mlp = EqualizedLinear(latent_size,
        #                            channels * 2,
        #                            gain=1.0, use_wscale=use_wscale)

        self.mlp = nn.Sequential(
                nn.Linear(latent_size, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, out_channels * 2))

    def forward(self, x, latent, mean_shift=0.0, var_shift=0.0, scale=1.0):
        style = self.mlp(latent)  # style => [batch_size, n_channels*2]
        #print("ADAIN MLP SHAPE", style.shape) #torch.Size([1, 2048])
        x = x.transpose(1, 2)  
        #print("ADAIN SHAPE", x.shape) #torch.Size([1, 768, 1024])
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        #print("STYLE", style.shape)
        mean = style[:, 1] - mean_shift
        var = style[:, 0] + 1. - var_shift

        #print("x shape:", x.shape)
        #print("mean shape:", mean.shape)
        #print("var shape:", var.shape)
        #print("scale value:", scale)  # scale is a scalar, so we directly print its value

        x = x * (var * scale) + mean
        return x
 

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x
    
class PrepForMidas(object):
    def __init__(self, resize_mode="minimal", keep_aspect_ratio=True, img_size=384, do_resize=True):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        net_h, net_w = img_size
        self.normalization = Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resizer = Resize(net_w, net_h, keep_aspect_ratio=keep_aspect_ratio, ensure_multiple_of=32, resize_method=resize_mode) \
            if do_resize else nn.Identity()

    def __call__(self, x):
        return self.normalization(self.resizer(x))

class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        print("Project Readout", x.shape)
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)
        print("Project Readout, After Reshape", x.shape)
        x = self.project(features)
        print("Projection Shape", x.shape)
        return x
   
class MidasCore(nn.Module):
    def __init__(self, midas, trainable=False, fetch_features=True, layer_names=('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1'), freeze_bn=False, keep_aspect_ratio=True,
                 img_size=384, **kwargs):
        """Midas Base model used for multi-scale feature extraction.

        Args:
            midas (torch.nn.Module): Midas model.
            trainable (bool, optional): Train midas model. Defaults to False.
            fetch_features (bool, optional): Extract multi-scale features. Defaults to True.
            layer_names (tuple, optional): Layers used for feature extraction. Order = (head output features, last layer features, ...decoder features). Defaults to ('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1').
            freeze_bn (bool, optional): Freeze BatchNorm. Generally results in better finetuning performance. Defaults to False.
            keep_aspect_ratio (bool, optional): Keep the aspect ratio of input images while resizing. Defaults to True.
            img_size (int, tuple, optional): Input resolution. Defaults to 384.
        """
        super().__init__()
        
        
        self.core = midas
        self.output_channels = None
        self.core_out = {}
        self.trainable = trainable
        self.fetch_features = fetch_features
        # midas.scratch.output_conv = nn.Identity()
        self.handles = []
        # self.layer_names = ['out_conv','l4_rn', 'r4', 'r3', 'r2', 'r1']
        self.layer_names = layer_names
        features=[256, 512, 1024, 1024]
        readout_oper = [
            ProjectReadout(in_features=1024, start_index=0) for out_feat in features
        ]
        self.set_trainable(trainable)
        self.set_fetch_features(fetch_features)

        self.prep = PrepForMidas(keep_aspect_ratio=keep_aspect_ratio,
                                 img_size=img_size, do_resize=kwargs.get('do_resize', True))
        self.adain = AdaIn(512,1024)
        
        if freeze_bn:
            self.freeze_bn()
        vit_features = 1024
        size=[384, 512]
        self.act_postprocess1 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
                ),
            )
        self.act_postprocess2 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )
        self.act_postprocess3 = nn.Sequential(
            readout_oper[2],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[2],
                kernel_size=1,
                stride=1,
                padding=0,
                ),
            )
        self.act_postprocess4 = nn.Sequential(
            readout_oper[3],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=features[3],
                out_channels=features[3],
                kernel_size=3,
                stride=2,
                padding=1,
                ),
            )

    def set_trainable(self, trainable):
        self.trainable = trainable
        if trainable:
            self.unfreeze()
        else:
            self.freeze()
        return self

    def set_fetch_features(self, fetch_features):
        self.fetch_features = fetch_features
        if fetch_features:
            if len(self.handles) == 0:
                self.attach_hooks(self.core)
        else:
            self.remove_hooks()
        return self

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.trainable = False
        return self

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.trainable = True
        return self

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self
    

    def forward(self, x, denorm=False, return_rel_depth=False):
        if denorm:
            x = denormalize(x)
        x = self.prep(x)
        
        print("Shape after prep: ", x.shape)
        z = torch.randn((x.shape[0], 512), device=x.device)


        x = self.core.pretrained.model.patch_embed(x)
        x = self.core.pretrained.model.pos_drop(x)
        print("PATCH EMBED SHAPE", x.shape) #torch.Size([1, 768, 1024])
        for i in range(0,24):
            x = self.core.pretrained.model.blocks[i].norm1(x)
            x = self.core.pretrained.model.blocks[i].attn.qkv(x)

            #Attention Score calculation
            batch_size, num_patches, feature_dim = x.shape
            d_k = feature_dim // 3
            q, k, v = x.reshape(batch_size, num_patches, 3, d_k).split(1, dim=2)
            q = q.squeeze(2)
            k = k.squeeze(2)
            v = v.squeeze(2)
            attention_scores = torch.matmul(q, k.transpose(-2, -1))
            attention_scores = attention_scores / np.sqrt(d_k)
            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            x = torch.matmul(attention_probs, v)
            print("Attention output shape:", x.shape)
            x = self.core.pretrained.model.blocks[i].attn.attn_drop(x)
            x = self.core.pretrained.model.blocks[i].attn.proj(x)
            x = self.core.pretrained.model.blocks[i].attn.proj_drop(x)
            
            #Post Attention
            x = self.core.pretrained.model.blocks[i].drop_path1(x)
            x = self.core.pretrained.model.blocks[i].norm2(x)
            x = self.core.pretrained.model.blocks[i].mlp(x)
            x = self.core.pretrained.model.blocks[i].drop_path2(x)
            #x.shape = torch.Size([1, 768, 1024])
            #AdaIN here
            if i%4 == 0 and i != 0:
                x = self.adain(x, z) #Adding AdaIN normalization every 4 layers
                x = x.transpose(1, 2)
                x = x.squeeze(1)
            print("SHAPE OUT OF ADAIN", x.shape)
            print(i,"BLOCK")
        
        print("SHAPE AT THE END OF BLOCKS", x.shape)
        print("FINSHED BLOCKS")
        unflatten = nn.Sequential(
            nn.Unflatten(
                2,
                torch.Size(
                    [
                        384 // self.core.pretrained.model.patch_size[1],
                        512 // self.core.pretrained.model.patch_size[0],
                    ]
                ),
            )
        )
        x = self.core.pretrained.model.norm(x)
        x = self.core.pretrained.model.fc_norm(x)
        x = self.core.pretrained.model.head_drop(x)
        # x = self.core.pretrained.model.head(x)
        print("Before PostProc", x.shape)
        print("FINISHED BEIT")
        #layer_1 = self.act_postprocess1(x)
        layer_1 = self.act_postprocess1[0:2](x)
        layer_2 = self.act_postprocess2[0:2](x)
        layer_3 = self.act_postprocess3[0:2](x)
        layer_4 = self.act_postprocess4[0:2](x)
        if layer_1.ndim == 3:
            layer_1 = unflatten(layer_1)
        if layer_2.ndim == 3:
            layer_2 = unflatten(layer_2)
        if layer_3.ndim == 3:
            layer_3 = unflatten(layer_3)
        if layer_4.ndim == 3:
            layer_4 = unflatten(layer_4)
        print("UNFLATENNING")
        layer_1 = self.act_postprocess1[3: len(self.act_postprocess1)](layer_1)
        layer_2 = self.act_postprocess2[3: len(self.act_postprocess2)](layer_2)
        layer_3 = self.act_postprocess3[3: len(self.act_postprocess3)](layer_3)
        layer_4 = self.act_postprocess4[3: len(self.act_postprocess4)](layer_4)

        print("Layer 1 Shape", layer_1.shape)

        print("FINISHED POST PROCESS")
        layer_1_rn = self.core.scratch.layer1_rn(layer_1)
        layer_2_rn = self.core.scratch.layer2_rn(layer_2)
        layer_3_rn = self.core.scratch.layer3_rn(layer_3)
        layer_4_rn = self.core.scratch.layer4_rn(layer_4)
        print("FINISHED RN LAYERS")
        path_4 = self.core.scratch.refinenet4(layer_4_rn)
        path_3 = self.core.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.core.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.core.scratch.refinenet1(path_2, layer_1_rn)
        print("FINISHED REFINEMENT")
        rel_depth = self.core.scratch.output_conv(path_1)
        print("GOT REL DEPTH")
        out = [self.core_out[k] for k in self.layer_names]
        print("CORE OUT")
        rel_depth = rel_depth.squeeze(1)
        print("REL DEPTH SHAPE", rel_depth.shape)
        print("OUT DEPTH", out[0].shape)
        if return_rel_depth:
            return rel_depth, out
        return out
    

    # def forward(self, x, denorm=False, return_rel_depth=False):
    #     if denorm:
    #         x = denormalize(x)
    #     x = self.prep(x)
        
    #     print("Shape after prep: ", x.shape)
    #     z = torch.randn((1, 512), device=x.device)
    #     self.core.pretrained.model.patch_size = [16, 16]

    #     x = self.core.pretrained.model.patch_embed(x)
    #     x = self.core.pretrained.model.pos_drop(x)
    #     resolution = (x.shape[1], x.shape[2])
    #     print('RESOLUTION', resolution)

    #     out = []
    #     for i, hook_idx in enumerate(self.hooks):
    #         if i < 3:
    #             layer_output = self.core.pretrained.model(x)
    #             print("LAYEROUR", layer_output.shape)
    #             adain_output = self.adains[i](layer_output, z)
    #             out.append(adain_output)
    #             x = adain_output  # Passing AdaIN output to the next layer
    #             resolution = (x.shape[1], x.shape[2])
    #     rel_depth = self.core.scratch(x)
    #     out = [self.core_out[k] for k in self.layer_names]
    #     if return_rel_depth:
    #         return rel_depth, out
    #     return out

    # def forward(self, x, denorm=False, return_rel_depth=False):
    #     with torch.no_grad():
    #         if denorm:
    #             x = denormalize(x)
    #         x = self.prep(x)
    #     z = torch.randn((1, 512), device=x.device)

    #         #print("Shape after prep: ", x.shape)
    #     with torch.set_grad_enabled(self.trainable):
    #         rel_depth = self.core.pretrained.model(x)
    #         hooks=[0, 4, 8, 11]
    #         self.core.pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation_BEiT("1"))

    #         print("Output from midas shape", rel_depth.shape)
    #         if not self.fetch_features:
    #             return rel_depth
    #     out = [self.core_out[k] for k in self.layer_names]

    #     if return_rel_depth:
    #         return rel_depth, out
    #     return out

    # def forward(self, x, denorm=False, return_rel_depth=False):
    #     if denorm:
    #         x = denormalize(x)
    #     x = self.prep(x)
    #     print("SHAPE", x.to('cpu').shape)
    #     # Sample latent code z for AdaIN
    #     z = torch.randn((1, 512), device=x.device)

    #     # Process through the MiDaS model's encoder, applying AdaIN after each layer
    #     adain_index = 0
    #     intermediate_outputs = []

    #     if hasattr(self.core, 'pretrained'):
    #         x = self.core.pretrained.model.patch_embed(x)
            
    #         x = self.core.pretrained.model.pos_drop(x)
    #         for i, blk in enumerate(self.core.pretrained.model.blocks):
    #             print("\n NUMBER", i)
    #             print(x.to('cpu').shape)
    #             x = blk(x)
    #             if i < len(self.adain_layers):
    #                 x = self.adain_layers[i](x, z)

    #         x = self.core.pretrained.norm(x)

    #     for name, module in self.core.named_children():
            
    #         x = module(x)
            
    #         # Check if the layer is one of the encoder layers where AdaIN should be applied
    #         if name in self.adain_layers_names:
    #             x = self.adain_layers[adain_index](x, z)
    #             adain_index += 1
            
    #         # Optionally store intermediate outputs for later use
    #         if name in self.layer_names:
    #             intermediate_outputs.append(x)

    #     # Assuming 'x' now holds the output from the last encoder layer after AdaIN application
    #     # Proceed with the rest of the network (decoder or other components)
    #     rel_depth = self.core(x)

    #     if not self.fetch_features:
    #         return rel_depth

    #     out = intermediate_outputs

    #     if return_rel_depth:
    #         return rel_depth, out
    #     return out


    def get_rel_pos_params(self):
        for name, p in self.core.pretrained.named_parameters():
            if "relative_position" in name:
                yield p

    def get_enc_params_except_rel_pos(self):
        for name, p in self.core.pretrained.named_parameters():
            if "relative_position" not in name:
                yield p

    def freeze_encoder(self, freeze_rel_pos=False):
        if freeze_rel_pos:
            for p in self.core.pretrained.parameters():
                p.requires_grad = False
        else:
            for p in self.get_enc_params_except_rel_pos():
                p.requires_grad = False
        return self

    def attach_hooks(self, midas):
        if len(self.handles) > 0:
            self.remove_hooks()
        if "out_conv" in self.layer_names:
            self.handles.append(list(midas.scratch.output_conv.children())[
                                3].register_forward_hook(get_activation("out_conv", self.core_out)))
        if "r4" in self.layer_names:
            self.handles.append(midas.scratch.refinenet4.register_forward_hook(
                get_activation("r4", self.core_out)))
        if "r3" in self.layer_names:
            self.handles.append(midas.scratch.refinenet3.register_forward_hook(
                get_activation("r3", self.core_out)))
        if "r2" in self.layer_names:
            self.handles.append(midas.scratch.refinenet2.register_forward_hook(
                get_activation("r2", self.core_out)))
        if "r1" in self.layer_names:
            self.handles.append(midas.scratch.refinenet1.register_forward_hook(
                get_activation("r1", self.core_out)))
        if "l4_rn" in self.layer_names:
            self.handles.append(midas.scratch.layer4_rn.register_forward_hook(
                get_activation("l4_rn", self.core_out)))

        return self

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        return self

    def __del__(self):
        self.remove_hooks()

    def set_output_channels(self, model_type):
        self.output_channels = MIDAS_SETTINGS[model_type]

    @staticmethod
    def build(midas_model_type="DPT_BEiT_L_384", train_midas=False, use_pretrained_midas=True, fetch_features=False, freeze_bn=True, force_keep_ar=False, force_reload=False, **kwargs):
        if midas_model_type not in MIDAS_SETTINGS:
            raise ValueError(
                f"Invalid model type: {midas_model_type}. Must be one of {list(MIDAS_SETTINGS.keys())}")
        if "img_size" in kwargs:
            kwargs = MidasCore.parse_img_size(kwargs)
        img_size = kwargs.pop("img_size", [384, 384])
        # print("img_size", img_size)
        # midas = torch.hub.load("intel-isl/MiDaS", midas_model_type,
        #                        pretrained=use_pretrained_midas, force_reload=force_reload)
        midas = torch.hub.load("AyaanShah2204/MiDaS", midas_model_type,
                               pretrained=use_pretrained_midas, force_reload=force_reload) # switcher to a better version?
        kwargs.update({'keep_aspect_ratio': force_keep_ar})
        midas_core = MidasCore(midas, trainable=train_midas, fetch_features=fetch_features,
                               freeze_bn=freeze_bn, img_size=img_size, **kwargs)
        midas_core.set_output_channels(midas_model_type)
        return midas_core

    @staticmethod
    def build_from_config(config):
        return MidasCore.build(**config)

    @staticmethod
    def parse_img_size(config):
        assert 'img_size' in config
        if isinstance(config['img_size'], str):
            assert "," in config['img_size'], "img_size should be a string with comma separated img_size=H,W"
            config['img_size'] = list(map(int, config['img_size'].split(",")))
            assert len(
                config['img_size']) == 2, "img_size should be a string with comma separated img_size=H,W"
        elif isinstance(config['img_size'], int):
            config['img_size'] = [config['img_size'], config['img_size']]
        else:
            assert isinstance(config['img_size'], list) and len(
                config['img_size']) == 2, "img_size should be a list of H,W"
        return config


nchannels2models = {
    tuple([256]*5): ["DPT_BEiT_L_384"],
    (512, 256, 128, 64, 64): ["MiDaS_small"]
}

# Model name to number of output channels
MIDAS_SETTINGS = {m: k for k, v in nchannels2models.items()
                  for m in v
                  }
