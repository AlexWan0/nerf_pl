import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from bisect import bisect

from .ray_utils import *

class TUMDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(640, 480), load_limit=100, focal=517, bounds=(2.0, 6.0)):
        self.root_dir = root_dir # dataset rootdir
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()
        self.load_limit = load_limit

        self.focal = focal

        self.near, self.far = bounds

        self.traj_path = os.path.join(root_dir, "groundtruth.txt")
        self.rgb_path = os.path.join(root_dir, "rgb.txt")
        self.depth_path = os.path.join(root_dir, "depth.txt")

        self.read_meta()
        self.white_back = True

    def parse_frames(self, path):
        files = []
        with open(path) as file:
            for line in file:
                if not line[0] == "#":
                    spl = line.split()
                    files.append([float(spl[0]), spl[1]])
        return files

    def read_meta(self):
        # read files
        trajectory = []
        with open(self.traj_path) as traj_file:
            for line in traj_file:
                if not line[0] == "#":
                    trajectory.append([float(n) for n in line.split()])

        trajectory = np.array(trajectory)

        rgb_files = self.parse_frames(self.rgb_path)
        depth_files = self.parse_frames(self.depth_path)

        assert len(rgb_files) == len(depth_files)

        # create meta
        self.meta = {'frames': []}

        for i in range(len(rgb_files)):
            frame_data = {}
            
            frame_timestamp = rgb_files[i][0]
            
            frame_data['rgb_path'] = rgb_files[i][1]
            frame_data['depth_path'] = depth_files[i][1]
            
            traj_point = trajectory[bisect([t[0] for t in trajectory], frame_timestamp)] # gets datapoint in trajectory with closest timestamp
            traj_timestamp, tx, ty, tz, qx, qy, qz, qw = traj_point
            
            #print(frame_timestamp, traj_timestamp)
            
            c2w = np.zeros((4,4))
            
            r = Rotation.from_quat([qx, qy, qz, qw])
            c2w[:3, :3] = r.as_matrix()
            
            c2w[0:3, 3] = np.array([tx, ty, tz])
            
            c2w[3, 3] = 1
            
            frame_data['rotation_vector'] = r.as_rotvec().tolist()
            
            frame_data['transform_matrix'] = c2w.tolist()
            
            self.meta['frames'].append(frame_data)

        w, h = self.img_wh
        
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