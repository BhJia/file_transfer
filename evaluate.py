import os
import json
import torch
import imageio
import math
import numpy as np
import torchvision
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, List
from einops import rearrange
from PIL import Image, ImageDraw
import random as rnd
from LLaVA.llava.model import *
from LLaVA.llava.train.train import load_video, load_mask
from LLaVA.llava.model.video_diffusion.unet import VideoInpaintingModel

seed=0
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    # os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, duration= 1.0/fps, loop=0)

print("Start loading model")
weight_dtype = torch.float16
model = LlavaLlamaForCausalLM.from_pretrained('/data/jbh/LLaVA-Lightning-7B-delta-v1-1')
model.unet = VideoInpaintingModel.from_pretrained('/data/jbh/stable-diffusion-2-inpainting/', subfolder='unet_finetuned')
model = model.to('cuda:0')
model.text_encoder.to(weight_dtype)
model.vae.to(weight_dtype)
model.unet.to(weight_dtype)
print("Model loaded successfully")

def load_mask(video_path, indices, mask_id, convert_to_box=False):
    WIDTH = 512
    HEIGHT = 320
    
    # print(video_path)
    frame_files = list(sorted(os.listdir(video_path)))
    frame_files = [x for x in frame_files if not x.startswith('.')]  # Excludes files like .DS_Store
    selected_frames = [frame_files[i] for i in indices]
    frames = []
    
    for frame_name in selected_frames:
        image = Image.open(os.path.join(video_path, frame_name))
        all_mask = np.array(image)
        # mask = (all_mask == int(mask_id)).astype(np.uint8) * 255
        mask = all_mask.astype(np.uint8) * 255
        
        if convert_to_box:
            box_image = Image.new("L", image.size, 255)
            draw = ImageDraw.Draw(box_image)
            # Find the bounding box of the mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            # box = (xmin, ymin, xmax, ymax)
            if rows.any() and cols.any():  # Only proceed if there is at least one non-zero value
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                draw.rectangle([xmin , ymin, xmax, ymax], fill=0)
        
            box_image = box_image.resize((WIDTH, HEIGHT), resample=Image.BILINEAR)
            box_np = np.array(box_image)
            box_tensor = torch.from_numpy(box_np).float().div(255).unsqueeze(0)  # Add channel dimension
            frames.append(box_tensor)
        # Stack all tensors to create a batch
            
        else:
            image = Image.fromarray(mask)
            image = image.resize((WIDTH, HEIGHT), resample=Image.BILINEAR)
            frames.append(image)
    
    if not convert_to_box:
        # Stack images and convert to a tensor
        frames = np.stack(frames, axis=2)
        frames = torch.from_numpy(frames).permute(2, 0, 1).contiguous().unsqueeze(1)
        frames = torch.where(frames > 0, torch.tensor(0.0), torch.tensor(1.0))
    else:
        frames = torch.stack(frames, dim=0)
        frames = torch.where(frames > 0, torch.tensor(1.0), torch.tensor(0.0))
    
    return frames
    
def load_video(video_path, sample_num=16, sample_type='uniform', given_index=None):
    WIDTH = 512
    HEIGHT = 320
    
    
    frame_files = list(sorted(os.listdir(video_path)))
    # exclude .DS_Store
    frame_files = [x for x in frame_files if x[0]!='.']
    # print(frame_files)
    vlen = len(frame_files)

    n_frms = min(sample_num, vlen)
    start, end = 0, vlen

    if given_index is None:
        intervals = np.linspace(start=start, stop=end, num=n_frms + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1]))
    
        if sample_type == 'random':
            indices = []
            for x in ranges:
                if x[0] == x[1]:
                    indices.append(x[0])
                else:
                    indices.append(rnd.choice(range(x[0], x[1])))
        elif sample_type == 'uniform':
            indices = [(x[0] + x[1]) // 2 for x in ranges]
        
        selected_frames = [frame_files[i] for i in indices]
        if len(selected_frames) < sample_num:
            selected_frames += [frame_files[-1]] * (sample_num - len(selected_frames))
            indices += [indices[-1]] * (sample_num - len(indices))
    else:
        selected_frames = [frame_files[i] for i in given_index]
        indices = given_index
    
    # [:max_num_frames]
    frames = []
    # print(len(selected_frames))
    for frame_name in selected_frames:
        image = Image.open(os.path.join(video_path, frame_name)).convert("RGB")
        image = image.resize((WIDTH, HEIGHT), resample=Image.BILINEAR)
        frames.append(image)

    frames = np.stack(frames, axis=2)
    frames = torch.from_numpy(frames).permute(2, 3, 0, 1).contiguous() #.unsqueeze(0)
    frames = frames.float().div(255).clamp(0, 1).half() * 2.0 - 1.0
    return frames, indices


class EvalDataset(Dataset):
    def __init__(self,frame_num=16):
        super(EvalDataset, self).__init__()
        self.data = json.load(open('/data/jbh/RACCooN/VPLM/gt_test.json'))
        self.video_base_path = '/data/jbh/rovi/data/JPEGImages/'
        self.mask_base_path = '/data/jbh/rovi/data/Annotations/'
        self.inpainted_base_path = '/data/jbh/rovi/data/InpaintImages/'
        self.frame_num = frame_num
        # self.tokenizer, self.multimodal_cfg = tokenizer, multimodal_cfgs
        print('--num data: %d--'%(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        anno = self.data[i]
        vid = anno['vid']
        task = anno['task']
        mask_id = anno['mask_id']
        
        
        if task == 'removal':
            text = 'inpainted background' #anno['prompt'] #'inpainted background'
            target, index = load_video(os.path.join(self.inpainted_base_path, vid, mask_id), sample_num=self.frame_num)
            condition, _ = load_video(os.path.join(self.video_base_path, vid), sample_num=self.frame_num, given_index=index)
            mask = load_mask(os.path.join(self.mask_base_path, vid), index, mask_id, convert_to_box=False)
            
        elif task == 'adding':
            text = anno['description'] # anno['prompt']
            print(text)
            target, index = load_video(os.path.join(self.video_base_path, vid), sample_num=self.frame_num)
            condition, _ = load_video(os.path.join(self.inpainted_base_path, vid, mask_id), sample_num=self.frame_num, given_index=index)
            mask = load_mask(os.path.join(self.mask_base_path, vid), index, mask_id, convert_to_box=True)
        
        elif task == 'editing':
            text = anno['prompt']
            print(text)
            target, index = load_video(os.path.join(self.video_base_path, vid), sample_num=self.frame_num)
            condition, _ = load_video(os.path.join(self.video_base_path, vid), sample_num=self.frame_num, given_index=index)
            mask = load_mask(os.path.join(self.mask_base_path, vid), index, mask_id, convert_to_box=False)
            
        data_dict = {}
        data_dict['task'] = task
        data_dict['target'] = target # [1, 8, 3, 320, 512]
        data_dict['condition'] = condition # [1, 8, 3, 320, 512]
        data_dict['mask'] = mask    # [1, 8, 3, 320, 512]
        data_dict['text_prompt'] = text
        return data_dict
    
print("Start Inference")
eval_dataset = EvalDataset()
for i in range(len(eval_dataset)):
    input_dict = eval_dataset[i]
    task = [input_dict['task']]
    video = input_dict['condition'].to(weight_dtype).unsqueeze(0).to('cuda:0') # [16, 3, 320, 512]
    mask = input_dict['mask'].to(weight_dtype).unsqueeze(0).to('cuda:0') # [16, 1, 320, 512]
    text = [input_dict['text_prompt']]           
    # text = [short_dict[i]]
    inpainted = model.inpaint(
        video=video, # input video condition
        mask=mask,
        prompt=text,
        task = task)

    if os.path.exists("result/{}/".format(input_dict['task'])) == False:
        os.makedirs("result/{}/".format(input_dict['task']))
    output_path = "result/{}/{}.gif".format(input_dict['task'], str(i))
    save_videos_grid(inpainted, output_path)

