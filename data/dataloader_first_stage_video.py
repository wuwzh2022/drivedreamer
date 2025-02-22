import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
import torch
import numpy as np
from torch.utils import data
import glob
import pickle
import os
import torch.utils
import torch.utils.data
from omegaconf import DictConfig
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud,Box
from nuscenes.utils.geometry_utils import view_points,box_in_image
from nuscenes.map_expansion.map_api import NuScenesMap,NuScenesMapExplorer
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from torch.utils import data
from utils.tools import get_this_scene_info
from ldm.util import instantiate_from_config
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from einops import repeat
import omegaconf
from PIL import Image
import time

def disabled_train(self,mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class dataloader(data.Dataset):
    def __init__(self,cfg,num_boxes,cond_stage_config,movie_len,split_name='train',device=None):
        self.split_name = split_name
        self.cfg = cfg
        self.nusc = NuScenes(version=cfg['version'],dataroot=cfg['dataroot'],verbose=True)
        self.movie_len = movie_len
        if device is not None:
            self.device = f'cuda:{device}'
        self.num_boxes = num_boxes
        self.instantiate_cond_stage(cond_stage_config)
        self.nusc_maps = {
            'boston-seaport': NuScenesMap(dataroot='.', map_name='boston-seaport'),
            'singapore-hollandvillage': NuScenesMap(dataroot='.', map_name='singapore-hollandvillage'),
            'singapore-onenorth': NuScenesMap(dataroot='.', map_name='singapore-onenorth'),
            'singapore-queenstown': NuScenesMap(dataroot='.', map_name='singapore-queenstown'),
        }
        self.clip = self.clip.to(self.device)
        self.load_data_infos()

    def load_data_infos(self):
        data_info_path = os.path.join(self.cfg['dataroot'],f"nuScenes_advanced_infos_{self.split_name}.pkl")
        with open(data_info_path,'rb') as f:
            data_infos = pickle.load(f)
        data_infos = data_infos['infos']
        print(f"len:{len(data_infos)}")
        pic_infos = {}
        video_infos = {}
        start_time = time.time()
        
        for id in range(len(data_infos)):
            sample_token = data_infos[id]['token']
            scene_token = self.nusc.get('sample',sample_token)['scene_token']
            scene = self.nusc.get('scene',scene_token)
            if not scene['name'] in pic_infos.keys():
                pic_infos[scene['name']] = [data_infos[id]]
            else:
                pic_infos[scene['name']].append(data_infos[id])
        idx = 0        
        for key,value in pic_infos.items():
            value = list(sorted(value,key=lambda e:e['timestamp']))
            frames = torch.arange(len(value)).to(torch.long)
            chunks = frames.unfold(dimension=0,size=self.movie_len,step=1)
            for ch_id,ch in enumerate(chunks):
                video_infos[idx] = [value[id] for id in ch]
                idx += 1
            #print(f"{key}:{len(value)}")
        print(video_infos.keys())
        self.video_infos = video_infos
        end_time = time.time()
        print(f"cost time:{end_time - start_time}")

    def __len__(self):
        return len(self.video_infos)
    
    def instantiate_cond_stage(self,config):
        model = instantiate_from_config(config)
        self.clip = model.eval()
        for param in self.clip.parameters():
            param.requires_grad = False

    def get_data_info(self,idx):
        #TODO:fix idx
        video_info = self.video_infos[idx]
        print(video_info)
        out = {}
        for i in range(self.movie_len):
            sample_token = video_info[i]['token']
            scene_token = self.nusc.get('sample',sample_token)['scene_token']
            scene = self.nusc.get('scene',scene_token)
            text = scene['description']
            log_token = scene['log_token']
            log = self.nusc.get('log',log_token)
            nusc_map = self.nusc_maps[log['location']]

            if self.cfg['img_size'] is not None:
                ref_img,boxes,hdmap,category,yaw,translation = get_this_scene_info(self.cfg['dataroot'],self.nusc,nusc_map,sample_token,tuple(self.cfg['img_size']))
            else:
                ref_img,boxes,hdmap,category,yaw,translation = get_this_scene_info(self.cfg['dataroot'],self.nusc,nusc_map,sample_token)
            boxes = np.array(boxes).astype(np.float32)
            ref_img = torch.from_numpy(ref_img / 255. * 2 - 1.0).to(torch.float32)
            hdmap = torch.from_numpy(hdmap / 255. * 2 - 1.0).to(torch.float32)

            if not 'reference_image' in out.keys():
                out['reference_image'] = ref_img.unsqueeze(0)
            else:
                out['reference_image'] = torch.cat([out['reference_image'],ref_img.unsqueeze(0)],dim=0)
            
            if not 'HDmap' in out.keys():
                out['HDmap'] = hdmap[:,:,:3].unsqueeze(0)
            else:
                out['HDmap'] = torch.cat([out['HDmap'],hdmap[:,:,:3].unsqueeze(0)],dim=0)
            
            if not 'text' in out.keys():
                out['text'] = self.clip(text).cpu().to(torch.float32)
            else:
                out['text'] = torch.cat([out['text'],self.clip(text).to(torch.float32)],dim=0)
            
            if boxes.shape[0] == 0:
                boxes = torch.from_numpy(np.zeros(self.num_boxes,16))
                category = torch.from_numpy(np.zeros((self.num_boxes,out['text'].shape[1])))
            elif boxes.shape[0] < self.num_boxes:
                boxes_zero = np.zeros((self.num_boxes - boxes.shape[0],16))
                boxes = torch.from_numpy(np.concatenate((boxes,boxes_zero),axis=0))
                category_embed = self.clip(category).cpu().to(torch.float32)
                category_zero = torch.zeros([self.num_boxes-category_embed.shape[0],category_embed.shape[1]])
                category = torch.cat([category_embed,category_zero],dim=0)
            else:
                boxes = torch.from_numpy(boxes[:self.num_boxes])
                category_embed = self.clip(category).cpu().to(torch.float32)
                category = category_embed[:self.num_boxes]
            
            if not '3Dbox' in out.keys():
                out['3Dbox'] = boxes.unsqueeze(0).to(torch.float32)
                out['category'] = category.unsqueeze(0).to(torch.float32)
            else:
                out['3Dbox'] = torch.cat([out['3Dbox'],boxes.unsquueze(0)],dim=0)
                out['category'] = torch.cat((out['category'],category.unsquueze(0).to(torch.float32)),dim=0)
        return out
    def __getitem__(self,idx):
        return self.get_data_info(idx)
            

if __name__ == '__main__':
    cfg ={
        'dataroot': '/storage/group/4dvlab/datasets/nuScenes',
        'version': 'advanced_12Hz_trainval'
    }

    data_loader = dataloader(cfg,70,None,3,'train')
    input_data = data_loader.__getitem__(0)
    print(input_data)




        

        