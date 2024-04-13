# pretrained Module(
#   (model): Beit(
#     (patch_embed): PatchEmbed(
#       (proj): Conv2d(3, 1024, kernel_size=(16, 16), stride=(16, 16))
#       (norm): Identity()
#     )
#     (pos_drop): Dropout(p=0.0, inplace=False)
#     (blocks): ModuleList(
#       (0-23): 24 x Block(
#         (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (qkv): Linear(in_features=1024, out_features=3072, bias=False)
#           (attn_drop): Dropout(p=0.0, inplace=False)
#           (proj): Linear(in_features=1024, out_features=1024, bias=True)
#           (proj_drop): Dropout(p=0.0, inplace=False)
#         )
#         (drop_path1): Identity()
#         (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
#         (mlp): Mlp(
#           (fc1): Linear(in_features=1024, out_features=4096, bias=True)
#           (act): GELU(approximate='none')
#           (drop1): Dropout(p=0.0, inplace=False)
#           (norm): Identity()
#           (fc2): Linear(in_features=4096, out_features=1024, bias=True)
#           (drop2): Dropout(p=0.0, inplace=False)
#         )
#         (drop_path2): Identity()
#       )
#     )
#     (norm): Identity()
#     (fc_norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
#     (head_drop): Dropout(p=0.0, inplace=False)
#     (head): Linear(in_features=1024, out_features=1000, bias=True)
#   )
#   (act_postprocess1): Sequential(
#     (0): ProjectReadout(
#       (project): Sequential(
#         (0): Linear(in_features=2048, out_features=1024, bias=True)
#         (1): GELU(approximate='none')
#       )
#     )
#     (1): Transpose()
#     (2): Unflatten(dim=2, unflattened_size=torch.Size([24, 24]))
#     (3): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
#     (4): ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(4, 4))
#   )
#   (act_postprocess2): Sequential(
#     (0): ProjectReadout(
#       (project): Sequential(
#         (0): Linear(in_features=2048, out_features=1024, bias=True)
#         (1): GELU(approximate='none')
#       )
#     )
#     (1): Transpose()
#     (2): Unflatten(dim=2, unflattened_size=torch.Size([24, 24]))
#     (3): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
#     (4): ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))
#   )
#   (act_postprocess3): Sequential(
#     (0): ProjectReadout(
#       (project): Sequential(
#         (0): Linear(in_features=2048, out_features=1024, bias=True)
#         (1): GELU(approximate='none')
#       )
#     )
#     (1): Transpose()
#     (2): Unflatten(dim=2, unflattened_size=torch.Size([24, 24]))
#     (3): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
#   )
#   (act_postprocess4): Sequential(
#     (0): ProjectReadout(
#       (project): Sequential(
#         (0): Linear(in_features=2048, out_features=1024, bias=True)
#         (1): GELU(approximate='none')
#       )
#     )
#     (1): Transpose()
#     (2): Unflatten(dim=2, unflattened_size=torch.Size([24, 24]))
#     (3): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
#     (4): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#   )
# )






# scratch Module(
#   (layer1_rn): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (layer2_rn): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (layer3_rn): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (layer4_rn): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (refinenet1): FeatureFusionBlock_custom(
#     (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
#     (resConfUnit1): ResidualConvUnit_custom(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (activation): ReLU()
#       (skip_add): FloatFunctional(
#         (activation_post_process): Identity()
#       )
#     )
#     (resConfUnit2): ResidualConvUnit_custom(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (activation): ReLU()
#       (skip_add): FloatFunctional(
#         (activation_post_process): Identity()
#       )
#     )
#     (skip_add): FloatFunctional(
#       (activation_post_process): Identity()
#     )
#   )
#   (refinenet2): FeatureFusionBlock_custom(
#     (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
#     (resConfUnit1): ResidualConvUnit_custom(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (activation): ReLU()
#       (skip_add): FloatFunctional(
#         (activation_post_process): Identity()
#       )
#     )
#     (resConfUnit2): ResidualConvUnit_custom(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (activation): ReLU()
#       (skip_add): FloatFunctional(
#         (activation_post_process): Identity()
#       )
#     )
#     (skip_add): FloatFunctional(
#       (activation_post_process): Identity()
#     )
#   )
#   (refinenet3): FeatureFusionBlock_custom(
#     (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
#     (resConfUnit1): ResidualConvUnit_custom(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (activation): ReLU()
#       (skip_add): FloatFunctional(
#         (activation_post_process): Identity()
#       )
#     )
#     (resConfUnit2): ResidualConvUnit_custom(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (activation): ReLU()
#       (skip_add): FloatFunctional(
#         (activation_post_process): Identity()
#       )
#     )
#     (skip_add): FloatFunctional(
#       (activation_post_process): Identity()
#     )
#   )
#   (refinenet4): FeatureFusionBlock_custom(
#     (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
#     (resConfUnit1): ResidualConvUnit_custom(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (activation): ReLU()
#       (skip_add): FloatFunctional(
#         (activation_post_process): Identity()
#       )
#     )
#     (resConfUnit2): ResidualConvUnit_custom(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (activation): ReLU()
#       (skip_add): FloatFunctional(
#         (activation_post_process): Identity()
#       )
#     )
#     (skip_add): FloatFunctional(
#       (activation_post_process): Identity()
#     )
#   )
#   (output_conv): Sequential(
#     (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): Interpolate()
#     (2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
#     (5): ReLU(inplace=True)
#     (6): Identity()
#   )
# )