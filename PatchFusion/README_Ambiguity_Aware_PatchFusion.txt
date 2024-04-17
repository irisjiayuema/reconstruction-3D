# Running

Create and activate environmenta as instructed in the offcial README.
Download the weights from the following link. 
https://drive.google.com/drive/folders/11CsVVOF4zmv4F3Ldod1thSNJU8mItzhb?usp=drive_link
- Put midas_model_dict.pt in PatchFusion directory.
- Put ZoeDepthv1.pt in the work_dir (for training)
Run the below lines for Testing, produces 20 different depths.
export PYTHONPATH="${PYTHONPATH}:/home/rsp8/projects/def-keli/rsp8/reconstruction-3D/PatchFusion"
export PYTHONPATH="${PYTHONPATH}:/home/rsp8/projects/def-keli/rsp8/reconstruction-3D/PatchFusion/external"
python ./tools/test.py ./configs/patchfusion_zoedepth/zoedepth_general.py --ckp-path Zhyever/patchfusion_zoedepth --cai-mode r128 --cfg-option general_dataloader.dataset.rgb_image_dir='./examples/' --save --work-dir ./work_dir/predictions --test-type general

#Training

export PYTHONPATH="${PYTHONPATH}:/home/rsp8/projects/def-keli/rsp8/reconstruction-3D/PatchFusion"
export PYTHONPATH="${PYTHONPATH}:/home/rsp8/projects/def-keli/rsp8/reconstruction-3D/PatchFusion/external"
python ./tools/train.py configs/patchfusion_zoedepth/zoedepth_fine_pretrain_u4k.py --work-dir /home/rsp8/scratch/work_dir/patchfusion_zoedepth_fine_pretrain_u4k


