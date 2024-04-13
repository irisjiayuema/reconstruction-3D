set PYTHONPATH=%PYTHONPATH%;C:\Users\rajsh\Desktop\Generative Models\Final Project Stuff\reconstruction-3D\PatchFusion
set PYTHONPATH=%PYTHONPATH%;C:\Users\rajsh\Desktop\Generative Models\Final Project Stuff\reconstruction-3D\PatchFusion\external


python ./tools/test.py ./configs/patchfusion_zoedepth/zoedepth_general.py --ckp-path Zhyever/patchfusion_zoedepth --cai-mode r128 --cfg-option general_dataloader.dataset.rgb_image_dir='./examples/' --save --work-dir ./work_dir/predictions --test-type general

tools\train.py

./tools/dist_train.sh configs/patchfusion_zoedepth/zoedepth_coarse_pretrain_u4k.py 4 --work-dir ./work_dir/zoedepth_u4k --log-name coarse_pretrain_idk_anything --tag coarse,zoedep



torch-model-archiver --model-name CV_Model_CUDA --version 1.0 --model-file model.py --serialized-file network_with_regularization.pth --handler my_handler.py --export-path DISTR_MODEL



wget https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets/UnrealStereo4K_05.zip


 # def forward(self, x, denorm=False, return_rel_depth=False):
    #     if denorm:
    #         x = denormalize(x)
    #     x = self.prep(x)
        
    #     print("Shape after prep: ", x.shape)
    #     z = torch.randn((x.shape[0], 512), device=x.device)


    #     x = self.core.pretrained.model.patch_embed(x)
    #     x = self.core.pretrained.model.pos_drop(x)
    #     print("PATCH EMBED SHAPE", x.shape) #torch.Size([1, 768, 1024])
    #     for i in range(0,24):
    #         x = self.core.pretrained.model.blocks[i].norm1(x)
    #         x = self.core.pretrained.model.blocks[i].attn.qkv(x)

    #         #Attention Score calculation
    #         batch_size, num_patches, feature_dim = x.shape
    #         d_k = feature_dim // 3
    #         q, k, v = x.reshape(batch_size, num_patches, 3, d_k).split(1, dim=2)
    #         q = q.squeeze(2)
    #         k = k.squeeze(2)
    #         v = v.squeeze(2)
    #         attention_scores = torch.matmul(q, k.transpose(-2, -1))
    #         attention_scores = attention_scores / np.sqrt(d_k)
    #         attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    #         x = torch.matmul(attention_probs, v)
    #         print("Attention output shape:", x.shape)
    #         x = self.core.pretrained.model.blocks[i].attn.attn_drop(x)
    #         x = self.core.pretrained.model.blocks[i].attn.proj(x)
    #         x = self.core.pretrained.model.blocks[i].attn.proj_drop(x)
            
    #         #Post Attention
    #         x = self.core.pretrained.model.blocks[i].drop_path1(x)
    #         x = self.core.pretrained.model.blocks[i].norm2(x)
    #         x = self.core.pretrained.model.blocks[i].mlp(x)
    #         x = self.core.pretrained.model.blocks[i].drop_path2(x)
    #         #x.shape = torch.Size([1, 768, 1024])
    #         #AdaIN here
    #         if i%4 == 0 and i != 0:
    #             noise_level = 0.1  # Define your noise level
    #             noise = torch.randn_like(x) * 1.5 * noise_level
    #             x = x + noise
    #             x = self.adain(x, z) #Adding AdaIN normalization every 4 layers
    #             x = x.transpose(1, 2)
    #             x = x.squeeze(1)
    #             print(i)
    #         print("SHAPE OUT OF ADAIN", x.shape)
    #         print(i,"BLOCK")
        
    #     print("SHAPE AT THE END OF BLOCKS", x.shape)
    #     print("FINSHED BLOCKS")
    #     unflatten = nn.Sequential(
    #         nn.Unflatten(
    #             2,
    #             torch.Size(
    #                 [
    #                     384 // self.core.pretrained.model.patch_size[1],
    #                     512 // self.core.pretrained.model.patch_size[0],
    #                 ]
    #             ),
    #         )
    #     )
    #     x = self.core.pretrained.model.norm(x)
    #     x = self.core.pretrained.model.fc_norm(x)
    #     x = self.core.pretrained.model.head_drop(x)
    #     # x = self.core.pretrained.model.head(x)
    #     print("Before PostProc", x.shape)
    #     print("FINISHED BEIT")
    #     #layer_1 = self.act_postprocess1(x)
    #     layer_1 = self.act_postprocess1[0:2](x)
    #     layer_2 = self.act_postprocess2[0:2](x)
    #     layer_3 = self.act_postprocess3[0:2](x)
    #     layer_4 = self.act_postprocess4[0:2](x)
    #     if layer_1.ndim == 3:
    #         layer_1 = unflatten(layer_1)
    #     if layer_2.ndim == 3:
    #         layer_2 = unflatten(layer_2)
    #     if layer_3.ndim == 3:
    #         layer_3 = unflatten(layer_3)
    #     if layer_4.ndim == 3:
    #         layer_4 = unflatten(layer_4)
    #     print("UNFLATENNING")
    #     layer_1 = self.act_postprocess1[3: len(self.act_postprocess1)](layer_1)
    #     layer_2 = self.act_postprocess2[3: len(self.act_postprocess2)](layer_2)
    #     layer_3 = self.act_postprocess3[3: len(self.act_postprocess3)](layer_3)
    #     layer_4 = self.act_postprocess4[3: len(self.act_postprocess4)](layer_4)

    #     print("Layer 1 Shape", layer_1.shape)

    #     print("FINISHED POST PROCESS")
    #     layer_1_rn = self.core.scratch.layer1_rn(layer_1)
    #     layer_2_rn = self.core.scratch.layer2_rn(layer_2)
    #     layer_3_rn = self.core.scratch.layer3_rn(layer_3)
    #     layer_4_rn = self.core.scratch.layer4_rn(layer_4)
    #     print("FINISHED RN LAYERS")
    #     path_4 = self.core.scratch.refinenet4(layer_4_rn)
    #     path_3 = self.core.scratch.refinenet3(path_4, layer_3_rn)
    #     path_2 = self.core.scratch.refinenet2(path_3, layer_2_rn)
    #     path_1 = self.core.scratch.refinenet1(path_2, layer_1_rn)
    #     print("FINISHED REFINEMENT")
    #     rel_depth = self.core.scratch.output_conv(path_1)
    #     print("GOT REL DEPTH")
    #     out = [self.core_out[k] for k in self.layer_names]
    #     print("CORE OUT")
    #     rel_depth = rel_depth.squeeze(1)
    #     print("REL DEPTH SHAPE", rel_depth.shape)
    #     print("OUT DEPTH", out[0].shape)
    #     if return_rel_depth:
    #         return rel_depth, out
    #     return out
    

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