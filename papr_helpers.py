from test import *

import torch
import torch.nn as nn
import numpy as np

# class PAPREmbedder:
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs
#         self.create_embedding_fn()
        
#     def create_embedding_fn(self):
#         embed_fns = []
#         spatial_dims = self.kwargs['input_dims']
#         feature_dims = self.kwargs.get('feature_dims', 0)  # Additional non-spatial feature dimensions
#         out_dim = 0

#         # Embed spatial dimensions
#         if self.kwargs['include_input']:
#             embed_fns.append(lambda x: x[..., :spatial_dims])  # Only apply to spatial part
#             out_dim += spatial_dims
            
#         # Optionally, embed additional feature dimensions directly without frequency modulation
#         if feature_dims > 0 and self.kwargs.get('include_features', False):
#             embed_fns.append(lambda x: x[..., spatial_dims:])  # Only apply to additional features
#             out_dim += feature_dims

#         max_freq = self.kwargs['max_freq_log2']
#         N_freqs = self.kwargs['num_freqs']
        
#         if self.kwargs['log_sampling']:
#             freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
#         else:
#             freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        
#         for freq in freq_bands:
#             for p_fn in self.kwargs['periodic_fns']:
#                 # Apply periodic functions only to the spatial part of the input
#                 embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x[..., :spatial_dims] * np.pi * freq))
#                 out_dim += spatial_dims
                
#         self.embed_fns = embed_fns
#         self.out_dim = out_dim
        
#     def embed(self, inputs):
#         return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# def get_papr_embedder(multires, include_features=False, feature_dims=0):
#     """
#     Creates an embedder for PAPR with options to include additional feature dimensions.
    
#     Parameters:
#     - multires: The maximum frequency logarithm base 2.
#     - include_features: Boolean indicating whether to include direct embedding of additional features.
#     - feature_dims: The number of additional feature dimensions.
    
#     Returns:
#     - A lambda function for embedding.
#     - The output dimensionality of the embedding.
#     """
#     embed_kwargs = {
#                 'include_input': True,
#                 'input_dims': 3,  # Assuming spatial coordinates
#                 'feature_dims': feature_dims,  # Additional non-spatial input dimensions
#                 'include_features': include_features,  # Whether to include those additional features directly
#                 'max_freq_log2': multires-1,
#                 'num_freqs': multires,
#                 'log_sampling': True,
#                 'periodic_fns': [torch.sin, torch.cos],
#     }
    
#     embedder_obj = PAPREmbedder(**embed_kwargs)
#     embed = lambda x, eo=embedder_obj: eo.embed(x)
#     return embed, embedder_obj.out_dim
    
    
def compute_space_carving_loss_papr(papr_depth, target_hypothesis, is_joint=False, mask=None, norm_p=2, threshold=0.0):
    # n_rays, n_points = papr_depth.shape
    # num_hypothesis = target_hypothesis.shape[0]
    # print(papr_depth.shape)
    # n_points = papr_depth.numel()
    # print(n_points)
    # print(target_hypothesis.shape)


    # if target_hypothesis.shape[-1] == 1:
    #     ### In the case where there is no caching of quantiles
    #     target_hypothesis_repeated = target_hypothesis.repeat(1, 1, n_points)
    # else:
    #     ### Each quantile here already picked a hypothesis
    #     target_hypothesis_repeated = target_hypothesis
    
    # print(papr_depth.shape)
    # print(target_hypothesis.shape)
    # exit()

    papr_depth_repeated = papr_depth.unsqueeze(-1).unsqueeze(0).repeat(20, 1, 1, 1)
    target_hypothesis_repeated = target_hypothesis

    ## L2 distance
    # distances = torch.sqrt((pred_depth - target_hypothesis_repeated)**2)
    distances = torch.norm(papr_depth_repeated.unsqueeze(-1) - target_hypothesis_repeated.unsqueeze(-1), p=norm_p, dim=-1)

    if mask is not None:
        mask = mask.unsqueeze(0).repeat(distances.shape[0],1).unsqueeze(-1)
        distances = distances * mask

    if threshold > 0:
        distances = torch.where(distances < threshold, torch.tensor([0.0]).to(distances.device), distances)

    if is_joint:
        ### Take the mean for all points on all rays, hypothesis is chosen per image
        quantile_mean = torch.mean(distances, axis=1) ## mean for each quantile, averaged across all rays
        samples_min = torch.min(quantile_mean, axis=0)[0]
        loss =  torch.mean(samples_min, axis=-1)

    else:
        ### Each ray selects a hypothesis
        best_hyp = torch.min(distances, dim=0)[0]   ## for each sample pick a hypothesis
        ray_mean = torch.mean(best_hyp, dim=-1) ## average across samples
        loss = torch.mean(ray_mean)  

    return loss


# def test(model, device, dataset, save_name, args, resume_step):
#     testloader = get_loader(dataset, args.dataset, mode="test")
#     print("testloader:", testloader)

#     frames = {}
#     for frame, batch in enumerate(testloader):
#         depth = get_papr_depth(model, device, batch)
#         print(depth)
#         exit()


# # main taken from test
# def main(args, save_name, mode, resume_step=0):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = get_model(args, device)
#     dataset = get_dataset(args.dataset, mode=mode)

#     if args.test.load_path:
#         try:
#             model_state_dict = torch.load(args.test.load_path)
#             print("args.test.load_path")
#             print(model_state_dict)
#             for step, state_dict in model_state_dict.items():
#                 resume_step = int(step)
#                 model.load_my_state_dict(state_dict)
#         except:
#             model_state_dict = torch.load(os.path.join(args.save_dir, args.test.load_path, "model.pth"))
#             for step, state_dict in model_state_dict.items():
#                 resume_step = step
#                 model.load_my_state_dict(state_dict)
#         print("!!!!! Loaded model from %s at step %s" % (args.test.load_path, resume_step))
#     else:
#         try:
#             model_state_dict = torch.load(os.path.join(args.save_dir, args.index, "model.pth"))
#             for step, state_dict in model_state_dict.items():
#                 resume_step = int(step)
#                 model.load_my_state_dict(state_dict)
#         except:
#             model.load_my_state_dict(torch.load(os.path.join(args.save_dir, args.index, f"model_{resume_step}.pth")))
#         print("!!!!! Loaded model from %s at step %s" % (os.path.join(args.save_dir, args.index), resume_step))

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     test(model, device, dataset, save_name, args, resume_step)


# if __name__ == '__main__':

#     args = parse_args()
#     with open(args.opt, 'r') as f:
#         config = yaml.safe_load(f)

#     resume_step = args.resume

#     log_dir = os.path.join(config["save_dir"], config['index'])
#     os.makedirs(log_dir, exist_ok=True)

#     sys.stdout = Logger(os.path.join(log_dir, 'test.log'), sys.stdout)
#     sys.stderr = Logger(os.path.join(log_dir, 'test_error.log'), sys.stderr)

#     shutil.copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))
#     shutil.copyfile(args.opt, os.path.join(log_dir, os.path.basename(args.opt)))

#     setup_seed(config['seed'])

#     for i, dataset in enumerate(config['test']['datasets']):
#         name = dataset['name']
#         mode = dataset['mode']
#         print(name, dataset)
#         config['dataset'].update(dataset)
#         args = DictAsMember(config)
#         main(args, name, mode, resume_step)
