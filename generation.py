#!/usr/bin/env python
# coding: utf-8


# In[2]:


import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import webdataset as wds
import PIL
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)

import utils
from models_paper4 import BrainNetwork, BrainDiffusionPrior, BrainDiffusionPriorOld, VersatileDiffusionPriorNetwork


seed=42
utils.seed_everything(seed=seed)

torch.cuda.set_device(0)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# # Configurations

# In[2]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    # Example use
    jupyter_args = "--data_path=/fsx/proj-medarc/fmri/natural-scenes-dataset \
                    --subj=1 \
                    --model_name=prior_257_final_subj01_bimixco_softclip_byol"
    
    jupyter_args = jupyter_args.split()
    print(jupyter_args)


# In[3]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="prior_257_final_subj01_bimixco_softclip_byol",
    help="name of trained model",
)
parser.add_argument(
    "--autoencoder_name", type=str, default="autoencoder_subj01_4x_locont_no_reconst",
    help="name of trained autoencoder model",
)
parser.add_argument(
    "--data_path", type=str, default="/data1/zhaoyuxiao/naturalscenesdataset/",
    help="Path to where NSD data is stored (see README)",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,5,7],
)
parser.add_argument(
    "--img2img_strength",type=float, default=.85,
    help="How much img2img (1=no img2img; 0=outputting the low-level image itself)",
)
parser.add_argument(
    "--recons_per_sample", type=int, default=1,
    help="How many recons to output, to then automatically pick the best one (MindEye uses 16)",
)
parser.add_argument(
    "--vd_cache_dir", type=str, default='/data/ZhaoYuXiao/MindEye/models--shi-labs--versatile-diffusion/',
    help="Where is cached Versatile Diffusion model; if not cached will download to this path",
)

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
if autoencoder_name=="None":
    autoencoder_name = None


# In[4]:


if subj == 1:
    num_voxels = 15724
elif subj == 2:
    num_voxels = 14278
elif subj == 3:
    num_voxels = 15226
elif subj == 4:
    num_voxels = 13153
elif subj == 5:
    num_voxels = 13039
elif subj == 6:
    num_voxels = 17907
elif subj == 7:
    num_voxels = 12682
elif subj == 8:
    num_voxels = 14386
print("subj", subj, "num_voxels", num_voxels)


# In[5]:


val_url = f"{data_path}/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
meta_url = f"{data_path}/webdataset_avg_split/metadata_subj0{subj}.json"
num_train = 8559 + 300
num_val = 982
batch_size = val_batch_size = 1
voxels_key = 'nsdgeneral.npy' # 1d inputs

val_data = wds.WebDataset(val_url, resampled=False)\
    .decode("torch")\
    .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
    .to_tuple("voxels", "images", "coco")\
    .batched(val_batch_size, partial=False)

val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, shuffle=False)

# check that your data loader is working
for val_i, (voxel, img_input, coco) in enumerate(val_dl):
    print("idx",val_i)
    print("voxel.shape",voxel.shape)
    print("img_input.shape",img_input.shape)
    break


# ## Load autoencoder

# In[6]:



# # Load VD pipe

# In[12]:


print('Creating versatile diffusion reconstruction pipeline...')
from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel


vd_cache_dir="/data1/zhaoyuxiao/models/versatile-diffusion/"

# try:
vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_cache_dir).to(device).to(torch.float16)
# except:
#     print("Downloading Versatile Diffusion to", vd_cache_dir)
#     vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(
#             "shi-labs/versatile-diffusion",
#             cache_dir = vd_cache_dir).to(device).to(torch.float16)
vd_pipe.image_unet.eval()
vd_pipe.vae.eval()
vd_pipe.image_unet.requires_grad_(False)
vd_pipe.vae.requires_grad_(False)

vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(vd_cache_dir, subfolder="scheduler")
num_inference_steps = 20

# Set weighting of Dual-Guidance 
text_image_ratio = .0 # .5 means equally weight text and image, 0 means use only image
for name, module in vd_pipe.image_unet.named_modules():
    if isinstance(module, DualTransformer2DModel):
        module.mix_ratio = text_image_ratio
        for i, type in enumerate(("text", "image")):
            if type == "text":
                module.condition_lengths[i] = 77
                module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
            else:
                module.condition_lengths[i] = 257
                module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

unet = vd_pipe.image_unet
vae = vd_pipe.vae
noise_scheduler = vd_pipe.scheduler


# ## Load Versatile Diffusion model

# In[8]:


img_variations = False

out_dim = 576*1024
voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim,clip_size=1024)
voxel2clip = BrainNetwork(**voxel2clip_kwargs)
voxel2clip.requires_grad_(False)
voxel2clip.eval()

out_dim = 1024
depth = 6
dim_head = 64
heads = 1024//64 # heads * dim_head = 12 * 64 = 768
timesteps = 100 #100

prior_network = VersatileDiffusionPriorNetwork(
        dim=1024,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = 576,
        learned_query_mode="pos_emb"
    )

diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
    voxel2clip=voxel2clip,
)

# outdir = f'/data/zhaoyuxiao/MindEye_janus_paper4_1111111111111/train/'
# 
outdir = f'/data/zhaoyuxiao/MindEye_janus_paper4_384_384/train/'
ckpt_path = os.path.join(outdir, f'best.pth')
# ckpt_path = os.path.join(outdir, f'8_last.pth')

print("ckpt_path",ckpt_path)
checkpoint = torch.load(ckpt_path, map_location="cpu")
state_dict = checkpoint['model_state_dict']
print("EPOCH: ",checkpoint['epoch'])
diffusion_prior.load_state_dict(state_dict,strict=False)
diffusion_prior.eval().to(device)
pass


# In[13]:


print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

retrieve = False
plotting = False
saving = True
verbose = False
imsize = 512

    
ind_include = np.arange(num_val)
all_brain_recons = None

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
# specify the path to the model
#model_path = "deepseek-ai/Janus-Pro-7B"
model_path = "/data/ZhaoYuXiao/deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
save_dir = "/data/zhaoyuxiao/Q4-Q9/Q4/"
os.makedirs(save_dir, exist_ok=True)

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
question = "Provide a general description of the perceived scene."
# ###Q1: Provide a general description of the perceived scene.
# Q2: Specify the number and spatial arrangement of key objects.
# Q3: What potential activities could be happening based on the scene?
image = "/home/zyx/Janus-main/image_6.png" 
conversation = [
    {
        "role": "<|User|>",
        "content": f"<image_placeholder>\n{question}",
        "images": [image],
    },
    {"role": "<|Assistant|>", "content": ""},
]

pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)
    
for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl,total=len(ind_include))):
    if val_i<np.min(ind_include):
        continue
    voxel = torch.mean(voxel,axis=1).to(device)
   
    brain_clip_embeddings0, proj_embeddings = diffusion_prior.voxel2clip(voxel.to(device).float())
    brain_clip_embeddings0 =  brain_clip_embeddings0.view(len(voxel),-1,1024)

    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    timesteps_prior = 100

    try:
        brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                cond_scale = 1., timesteps = timesteps_prior,
                                generator=generator) 
    except:
        brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
                                text_cond = dict(text_embed = brain_clip_embeddings0), 
                                                cond_scale = 1., timesteps = timestepss_prior)
        

    # attention_mask = torch.ones((1, 630), dtype=torch.int, device=device)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(brain_clip_embeddings.to(torch.bfloat16), input_ids=prepare_inputs.input_ids,
                                                 images_seq_mask=prepare_inputs.images_seq_mask,
                                                 images_emb_mask=prepare_inputs.images_emb_mask) #1*630*4096

    # # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,   #全是1，  630维度
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
       # 立即保存
    filename = f"answer_{val_i}.txt"
    file_path = os.path.join("/data/zhaoyuxiao/Q4-Q9/Q1/", filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(answer)
  
    
    # utils.torch_to_Image(img).save(f"/data/zhaoyuxiao/MindEye_janus_paper4_no_duibi/data/{val_i}.png")
    print(answer)
            




