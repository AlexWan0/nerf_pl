import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from .ray_utils import *

class TUMDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(640, 480), load_limit=100):
        self.root_dir = root_dir # dataset rootdir
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()
        self.load_limit = load_limit

        self.read_meta()
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"meta.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        # scale focal length to match new image wh
        self.focal = self.meta['fx'] # make get_ray_directions use seperate fx and fy?

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for i, frame in tqdm(enumerate(self.meta['frames'])):
                if i % (len(self.meta['frames']) // self.load_limit) == 0:
                    pose = np.array(frame['transform_matrix'])[:3, :4]
                    self.poses += [pose]
                    c2w = torch.FloatTensor(pose)

                    image_path = os.path.join(self.root_dir, f"{frame['rgb_path']}")
                    self.image_paths += [image_path]
                    img = Image.open(image_path)
                    img = img.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img) # (3, h, w)
                    img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA
                    self.all_rgbs += [img]
                    
                    rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                    self.all_rays += [torch.cat([rays_o, rays_d, 
                                                 self.near*torch.ones_like(rays_o[:, :1]),
                                                 self.far*torch.ones_like(rays_o[:, :1])],
                                                 1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['rgb_path']}"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (3, H, W)

            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area

            img = img.view(3, -1).permute(1, 0) # (H*W, 3) RGBA

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample