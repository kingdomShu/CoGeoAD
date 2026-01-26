import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os
import re
def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec3d':
        obj_list = ["bagel", "cable_gland", "carrot", "cookie", "dowel","foam", "peach", "potato", "rope", "tire",]
  
    elif dataset_name == 'eyecandies':
        obj_list = [
       'CandyCane',
        'ChocolateCookie',
        'ChocolatePraline',
        'Confetto',
        'GummyBear',
        'HazelnutTruffle',
        'LicoriceSandwich',
        'Lollipop',
        'Marshmallow',
        'PeppermintCandy']



    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id


def generate_is_seen_for_point_in_each_view(d2_3d_cor_list, non_zero_index_arr):
    # nv, l, 3
    d2_3d_cor = torch.stack(d2_3d_cor_list)
    # nv, l
    is_seen = d2_3d_cor[..., 2]
    nonzero_index = np.nonzero(np.asarray(non_zero_index_arr).reshape(-1,))[0]
    max_nonzero_index, min_nonzero_index = nonzero_index[-1], nonzero_index[0]
    mask = np.asarray(is_seen.bool() | (~non_zero_index_arr.permute(1, 0).bool()))
    # Few points have no projections in any views, we fill the d2_3d_cor of its neighbors as its projection to avoid the nan value when back-projecting 2d presentations to 3d
    is_seen = np.nonzero(np.all(mask == 0, axis = 0))[0]
    
    if len(is_seen):
        for i in is_seen:
            success_find = 0
            up_select_idx = 0
            down_select_idx = 0
            for j in range(i+1, max_nonzero_index + 1):
                up_select_idx = j
                if np.any(d2_3d_cor[..., j, 2].numpy()):
                    success_find = 1
                    dis = np.abs(up_select_idx - i)
                    select_index = up_select_idx
                    break
            if not success_find:
                for j in range(min_nonzero_index, i):
                    up_select_idx = j
                    if np.any(d2_3d_cor[..., j, 2].numpy()):
                        success_find = 1
                        dis = np.abs(max_nonzero_index - i + up_select_idx - min_nonzero_index)
                        select_index = up_select_idx
                        break
            if not success_find:
                raise NotImplementedError("bug")
            for j in reversed(range(min_nonzero_index, i)):
                down_select_idx = j
                if dis <= np.abs(down_select_idx - i):
                    select_index = up_select_idx
                    break
                else:
                    if np.any(d2_3d_cor[..., j, 2].numpy()):
                        success_find = 1
                        select_index = down_select_idx
                        break
            if not success_find:
                for j in reversed(range(i+1, max_nonzero_index + 1)):
                    down_select_idx = j
                    if dis <= np.abs(i - min_nonzero_index + max_nonzero_index - down_select_idx):
                        select_index = up_select_idx
                        break
                    else:
                        if np.any(d2_3d_cor[..., j, 2].numpy()):
                            success_find = 1
                            select_index = down_select_idx
                            break
            if not success_find:
                raise NotImplementedError("bug")
            d2_3d_cor[..., i, 2]=d2_3d_cor[..., select_index, 2]
    is_seen = d2_3d_cor[..., 2]
    mask = np.asarray(is_seen.bool() | (~non_zero_index_arr.permute(1, 0).bool()))
    is_seen = np.nonzero(np.all(mask == 0, axis = 0))[0]
    return is_seen, d2_3d_cor


import open3d
class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, target_transform_pc, dataset_name, train_dataset_name = None, 
                 point_size = 336, is_all = False, mode='test',selected_views=[0,1,2,3,4,5,6,7,8]):
        self.root = root
        
        self.transform = transform
        self.target_transform = target_transform
        self.target_transform_pc = target_transform_pc
        self.point_size = point_size
        self.dataset_name = dataset_name
        self.data_all = []
        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
        
        meta_info = json.load(open(f'{self.root}/all_meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)
        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
        self.selected_views = selected_views

        print(f"Dataset mode: {mode}")
        print(f"Number of classes: {len(self.cls_names)}")
        print(f"Classes: {self.cls_names}")
        print(f"Total samples: {self.length}")
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        cls_name, specie_name, anomaly = data['cls_name'], data['specie_name'], data['anomaly']

        img_path = os.path.join(self.root, data['d2_img_path'])
        mask_path = os.path.join(self.root, data['d2_mask_path'])
            
        d2_render_img_path = os.path.join(self.root, data['d2_render_img_path'])
        d2_render_gt_path = os.path.join(self.root, data['d2_render_gt_path'])
        d2_corrdinate_path = os.path.join(self.root, data['d2_corrdinate'])


        # load 2d rendering images
        rgb_img_path_list = []
        uni_img_path_list = []
        rgb_view_files = sorted([f for f in os.listdir(d2_render_img_path) if f.startswith('rgb_view_')], 
                   key=lambda x: int(re.findall(r'rgb_view_(\d+)\.png', x)[0]))
        uni_view_files = sorted([f for f in os.listdir(d2_render_img_path) if f.startswith('uni_view_')], 
                   key=lambda x: int(re.findall(r'uni_view_(\d+)\.png', x)[0]))
        # view_files = sorted([f for f in os.listdir(d2_render_img_path) if f.startswith('view_')], 
        #            key=lambda x: int(re.findall(r'view_(\d+)\.png', x)[0]))

        if self.selected_views is not None:
            rgb_view_files = [rgb_view_files[i] for i in self.selected_views if i < len(rgb_view_files)]
            uni_view_files = [uni_view_files[i] for i in self.selected_views if i < len(uni_view_files)]
        for filename in rgb_view_files:
            img = Image.open(os.path.join(d2_render_img_path, filename)).convert("RGB")
            img = self.transform(img) if self.transform is not None else img
            rgb_img_path_list.append(img)
        for filename in uni_view_files:
            img = Image.open(os.path.join(d2_render_img_path, filename)).convert("RGB")
            img = self.transform(img) if self.transform is not None else img
            uni_img_path_list.append(img)
        rgb_img = torch.stack(rgb_img_path_list)
        uni_img = torch.stack(uni_img_path_list)

        # load 2d rendering groundtruth
        d2_render_gt_path_list = []
        rendering_anomaly_list = []
        gt_files = sorted([f for f in os.listdir(d2_render_gt_path)], 
                   key=lambda x: int(re.findall(r'view_(\d+)_gt\.png', x)[0]))
        if self.selected_views is not None:
            gt_files = [gt_files[i] for i in self.selected_views if i < len(gt_files)]
        for filename in gt_files:
            img_mask = Image.open((os.path.join(d2_render_gt_path, filename))).convert('L')
            img_mask = self.target_transform(img_mask)
            img_mask[img_mask>0.5] = 1.0
            img_mask[img_mask<=0.5] = 0.0
            rendering_anomaly = 0.0 if torch.all(img_mask == 0) else 1.0
            d2_render_gt_path_list.append(img_mask)
            rendering_anomaly_list.append(rendering_anomaly)
        d2_render_gt = torch.stack(d2_render_gt_path_list)
        rendering_anomaly = torch.tensor(rendering_anomaly_list)


        # load the correspondence between points and pixels in each view
        d2_3d_cor_list = []
        non_zero_index_list = []
        # for organized point ckoud
        
        cor_files = sorted([f for f in os.listdir(d2_corrdinate_path) if f.startswith('view_')], 
                   key=lambda x: int(re.findall(r'view_(\d+)_cor\.npy', x)[0]))
        #print(cor_files)
        non_zero_filename = os.path.join(d2_corrdinate_path,'nonzero_indices.npy')
        template_non_zero_index = torch.zeros(self.point_size * self.point_size, dtype = torch.long)
        non_zero_index = np.load((os.path.join(d2_corrdinate_path, non_zero_filename)))
        non_zero_index = torch.from_numpy(non_zero_index)
        template_non_zero_index[non_zero_index] = 1
        non_zero_index_arr = template_non_zero_index.reshape(-1, 1)
        if self.selected_views is not None:
            cor_files = [cor_files[i] for i in self.selected_views if i < len(cor_files)]
        #print(cor_files)
        if self.dataset_name == 'mvtec3d' or self.dataset_name == 'eyecandies':
            for filename in cor_files:
                
                template_d2_corrdinate = torch.zeros(self.point_size * self.point_size, 3, dtype = torch.long)
                d2_corrdinate = np.load((os.path.join(d2_corrdinate_path, filename)))

                d2_corrdinate = torch.from_numpy(d2_corrdinate).long()
                template_d2_corrdinate[non_zero_index] = d2_corrdinate
                d2_3d_cor_list.append(template_d2_corrdinate)
        # for unorganized point cloud


        # remove the hidden points in each view
        is_seen, d2_3d_cor = generate_is_seen_for_point_in_each_view(d2_3d_cor_list, non_zero_index_arr)
        
        if self.dataset_name == 'mvtec3d' or self.dataset_name == 'eyecandies':
            img = Image.open(os.path.join(self.root, img_path))
            if anomaly == 0:
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                if os.path.isdir(os.path.join(self.root, mask_path)):
                    img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
                else:
                    img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                    img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            # transforms
            img = self.transform(img) if self.transform is not None else img
            img_mask = self.target_transform(   
                img_mask) if self.target_transform is not None and img_mask is not None else img_mask
            img_mask = [] if img_mask is None else img_mask


            # print("=" * 50)
            # print(f"Sample Index: {index}")
            # print(f"Class Name: {cls_name}, Anomaly: {anomaly}")
            # print(f"Image Path: {os.path.join(self.root, img_path)}")
            # print(f"Img Shape: {img.size if hasattr(img, 'size') else 'N/A'}")
            # print(f"Img Mask Type: {type(img_mask)}, Shape: {getattr(img_mask, 'shape', 'N/A')}")
            # print(f"D2 Render Img Shape: {d2_render_img.shape}")
            # print(f"D2 Render Anomaly Shape: {rendering_anomaly.shape}, Values: {rendering_anomaly}")
            # print(f"D2 Render GT Shape: {d2_render_gt.shape}")
            # print(f"D2 3D Cor Shape: {d2_3d_cor.shape}")
            # print(f"Non Zero Index Shape: {non_zero_index_arr.shape}")
            # print(f"Selected Views Count: {len(view_files)}")
            # print("=" * 50)


            return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'rgb_render_img': rgb_img, 'uni_render_img': uni_img,'d2_render_anomaly': rendering_anomaly, 'd2_render_gt': d2_render_gt, 'd2_3d_cor': d2_3d_cor,
                    'img_path': os.path.join(self.root, img_path), "cls_id":self.class_name_map_class_id[cls_name], "d2_render_img_path": d2_render_img_path, "non_zero_index": non_zero_index_arr, "index":torch.LongTensor([index])}
        
