import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform

from model import ViewAttention, DeLinearLayer, ViewAttention_linear, ViewAttention_para, WeightedSum


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


def back_to_3d(d2_similarity_map, d2_3d_cor, non_zero_index, view_weights=None, ori_resolution=336):
    h = torch.sqrt(torch.tensor(d2_3d_cor.shape[2])).int()
    w = h
    b, nv, num_points, _ = d2_3d_cor.shape
    
    xx = d2_3d_cor[:, :, :, 0].reshape(-1).long()
    yy = d2_3d_cor[:, :, :, 1].reshape(-1).long()
    nbatch = torch.repeat_interleave(torch.arange(0, b*nv)[:,None], num_points).reshape(-1, ).cuda().long()
    
    d2_similarity_map = d2_similarity_map.permute(0, 3, 1, 2)
    point_logits_raw = d2_similarity_map[nbatch, :, yy, xx]
    point_logits_raw = point_logits_raw.reshape(b, nv, num_points, 2)
    
    if view_weights is not None:
        vweights = view_weights.reshape(b, nv, 1, 1) # [B, NV, 1, 1]
    else:
        vweights = torch.ones((b, nv, 1, 1)).to(point_logits_raw.device)
    
    is_seen = d2_3d_cor[:, :, :, 2].reshape(b, nv, num_points, 1)
    valid_mask = is_seen.bool() & non_zero_index.bool() 
    
    masked_weights = vweights * valid_mask.float()
    sum_weights = masked_weights.sum(dim=1)
    
    weighted_logits = point_logits_raw * masked_weights
    sum_weighted_logits = weighted_logits.sum(dim=1) # [Batch, Num_Points, 2]

    eps = 1e-8
    final_point_logits = sum_weighted_logits / (sum_weights + eps)
    
    final_point_logits = torch.nan_to_num(final_point_logits, nan=0.0, posinf=0.0, neginf=0.0)
    final_point_logits = final_point_logits.reshape(b, h, w, 2)
    
    return final_point_logits

def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform, target_transform_pc = get_transform(args)
    test_data = Dataset(root=dataset_dir, dataset_name = args.dataset, transform=preprocess, target_transform=target_transform, target_transform_pc = target_transform_pc, mode='test', is_all = True, point_size = args.point_size,selected_views=list(range(args.num_views)))
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list
    results = {}

    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['color_pr_sp'] = []
        results[obj]['integrate_pr_sp'] = []

        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        results[obj]['color_anomaly_maps'] = []
        results[obj]['integrate_anomaly_maps'] = []

    obj_list = [c for c in obj_list]
    
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)

    if args.view_attention_type == 'ViewAttention':
        view_attention = ViewAttention()
    elif args.view_attention_type == 'ViewAttention_linear':
        view_attention = ViewAttention_linear(input_dim=3*args.image_size*args.image_size)
    elif args.view_attention_type == 'ViewAttention_para':
        view_attention = ViewAttention_para(args.num_views)
    else:
        raise ValueError(f"Unknown view attention type: {args.view_attention_type}")

    view_attention.load_state_dict(checkpoint["view_attention"])
    view_attention.to(device)
    
    # Feature Fusion Layer
    use_delinear = len(args.features_list) >= 4
    feature_fusion_layer = None

    if use_delinear:
        feature_fusion_layer = DeLinearLayer(768, len(args.features_list), "ViT-L/14@336px")
        if "feature_fusion_layer" in checkpoint:
            feature_fusion_layer.load_state_dict(checkpoint["feature_fusion_layer"])
        else:
            logger.warning("Warning: feature_fusion_layer expected but not found in checkpoint.")
        feature_fusion_layer.to(device)
        num_fusion_weights = 4
    else:
        num_fusion_weights = len(args.features_list)
    weighted_sum_model = None
    if args.use_weighted_sum:
        weighted_sum_model = WeightedSum(dataset = args.dataset).to(device)
        if "weighted_sum_model" in checkpoint:
            weighted_sum_model.load_state_dict(checkpoint["weighted_sum_model"])
        else:
            logger.warning("Warning: weighted_sum_model expected but not found in checkpoint. Using default initialization.")
        weighted_sum_model.eval()
    
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)
    

    model.to(device)
    for idx, items in enumerate(tqdm(test_dataloader)):
        color_image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        
        rgb_render_image = items['rgb_render_img'].to(device)
        uni_render_image = items['uni_render_img'].to(device)
        
        b, nv, c, h, w = rgb_render_image.shape
        rgb_render_image = rgb_render_image.reshape(-1, c, h, w)
        uni_render_image = uni_render_image.reshape(-1, c, h, w)
        d2_render_anomaly = items['d2_render_anomaly'].to(device)
        d2_3d_cor = items['d2_3d_cor'].to(device)
        non_zero_index = items['non_zero_index'].to(device)
        non_zero_index = non_zero_index.unsqueeze(1).repeat(1, nv, 1, 1)

############################################################

        with torch.no_grad():
            if args.feature_type in ['rgb', 'both']:
                rgb_image_features, rgb_patch_features = model.encode_image(rgb_render_image, features_list, DPAM_layer = 20)
                rgb_image_features = rgb_image_features / rgb_image_features.norm(dim=-1, keepdim=True)
            
            if args.feature_type in ['uni', 'both']:
                uni_image_features, uni_patch_features = model.encode_image(uni_render_image, features_list, DPAM_layer = 20)
                uni_image_features = uni_image_features / uni_image_features.norm(dim=-1, keepdim=True)
                
            color_image_features,color_patch_features = model.encode_image(color_image, features_list, DPAM_layer = 20)
            
####################################################

            if args.feature_type in ['rgb', 'both']:
                if use_delinear:
                    rgb_fused_features_groups = feature_fusion_layer(rgb_patch_features)
                else:
                    rgb_fused_features_groups = rgb_patch_features

            if args.feature_type in ['uni', 'both']:
                if use_delinear:
                    uni_fused_features_groups = feature_fusion_layer(uni_patch_features)
                else:
                    uni_fused_features_groups = uni_patch_features
            
            if use_delinear:
                color_fused_features_groups = feature_fusion_layer(color_patch_features)
            else:
                color_fused_features_groups = color_patch_features
            
            similarity_maps_rgb = []
            similarity_maps_uni = []
            similarity_maps_color = []
            

            if args.feature_type in ['rgb', 'both']:
                for i in range(len(rgb_fused_features_groups)):
                    rgb_patch_feature = rgb_fused_features_groups[i]
                    rgb_patch_feature = rgb_patch_feature / rgb_patch_feature.norm(dim=-1, keepdim=True)
                    rgb_similarity, _ = AnomalyCLIP_lib.compute_similarity(rgb_patch_feature, text_features[0])
                    rgb_similarity_map = AnomalyCLIP_lib.get_similarity_map(rgb_similarity[:, 1:, :], args.image_size)
                    similarity_maps_rgb.append(rgb_similarity_map)
            
            if args.feature_type in ['uni', 'both']:
                for i in range(len(uni_fused_features_groups)):
                    uni_patch_feature = uni_fused_features_groups[i]
                    uni_patch_feature = uni_patch_feature / uni_patch_feature.norm(dim=-1, keepdim=True)
                    uni_similarity, _ = AnomalyCLIP_lib.compute_similarity(uni_patch_feature, text_features[0])
                    uni_similarity_map = AnomalyCLIP_lib.get_similarity_map(uni_similarity[:, 1:, :], args.image_size)
                    similarity_maps_uni.append(uni_similarity_map)
                

            for i in range(len(color_fused_features_groups)):
                color_patch_feature = color_fused_features_groups[i]
                color_patch_feature = color_patch_feature / color_patch_feature.norm(dim=-1, keepdim=True)
                color_similarity, _ = AnomalyCLIP_lib.compute_similarity(color_patch_feature, text_features[0])
                color_similarity_map = AnomalyCLIP_lib.get_similarity_map(color_similarity[:, 1:, :], args.image_size)
                similarity_maps_color.append(color_similarity_map)
            
            if args.feature_type == 'rgb':
                if args.use_weighted_sum:
                    rgb_similarity_map, _ = weighted_sum_model(similarity_maps_rgb)
                else:
                    rgb_similarity_map = torch.stack(similarity_maps_rgb, dim=0).mean(dim=0)

            elif args.feature_type == 'uni':
                if args.use_weighted_sum:
                    uni_similarity_map, _ = weighted_sum_model(similarity_maps_uni)
                else:
                    uni_similarity_map = torch.stack(similarity_maps_uni, dim=0).mean(dim=0)

            elif args.feature_type == 'both':
                if args.use_weighted_sum:
                    rgb_similarity_map, _ = weighted_sum_model(similarity_maps_rgb)
                    uni_similarity_map, _ = weighted_sum_model(similarity_maps_uni)
                else:
                    rgb_similarity_map = torch.stack(similarity_maps_rgb, dim=0).mean(dim=0)
                    uni_similarity_map = torch.stack(similarity_maps_uni, dim=0).mean(dim=0)
                
            if args.use_weighted_sum:
                color_similarity_map, _ = weighted_sum_model(similarity_maps_color)
            else:
                color_similarity_map = torch.stack(similarity_maps_color, dim=0).mean(dim=0)
            
####################################################

            if args.feature_type == 'rgb':
                similarity_map = rgb_similarity_map
            elif args.feature_type == 'uni':
                similarity_map = uni_similarity_map
            elif args.feature_type == 'both':
                similarity_map = torch.max(rgb_similarity_map, uni_similarity_map)
            
####################################################
            similarity_map_nv = torch.chunk(similarity_map, nv, dim = 0)
            similarity_map_nv = torch.stack(similarity_map_nv, dim = 1)
            anomaly_map_nv = (similarity_map_nv[...,1] + 1 - similarity_map_nv[...,0])/2.0

            #########################################
            
            if args.feature_type in ['rgb', 'both']:
                rgb_text_probs = rgb_image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
                rgb_text_probs = rgb_text_probs[:, 0, ...]/0.07
            if args.feature_type in ['uni', 'both']:
                uni_text_probs = uni_image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
                uni_text_probs = uni_text_probs[:, 0, ...]/0.07
            
            rgb_images = rgb_render_image.reshape(b, nv, c, h, w)
            
            if args.use_view_attention:
                view_weights = view_attention([rgb_images[:, i] for i in range(nv)])
                d3_similarity_map = back_to_3d(similarity_map, d2_3d_cor, non_zero_index, view_weights)
            else:
                d3_similarity_map = back_to_3d(similarity_map, d2_3d_cor, non_zero_index)
            # ####################################################
            anomaly_map = d3_similarity_map[...,1]
            color_map = (color_similarity_map[...,1] + 1 - color_similarity_map[...,0])/2.0
            # ####################################################
            
            
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0)
            color_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in color_map.detach().cpu()], dim = 0)
            integrate_anomaly_map = 0.2*color_anomaly_map + 0.8*anomaly_map
            integrate_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in integrate_anomaly_map.detach().cpu()], dim = 0)
#####################################################################

            results[cls_name[0]]['pr_sp'].append(anomaly_map.max().item())
            results[cls_name[0]]['color_pr_sp'].append(color_anomaly_map.max().item())
            results[cls_name[0]]['integrate_pr_sp'].append(integrate_anomaly_map.max().item())
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
            results[cls_name[0]]['color_anomaly_maps'].append(color_anomaly_map)
            results[cls_name[0]]['integrate_anomaly_maps'].append(integrate_anomaly_map)


    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []

    integrate_table_ls = []
    integrate_image_auroc_list = []
    integrate_image_ap_list = []
    integrate_pixel_auroc_list = []
    integrate_pixel_aupro_list = []

    obj_list = [c for c in test_data.obj_list]
    for obj in obj_list:
        table = []
        integrate_table = []
        table.append(obj)
        integrate_table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        results[obj]['color_anomaly_maps'] = torch.cat(results[obj]['color_anomaly_maps']).detach().cpu().numpy()
        results[obj]['integrate_anomaly_maps'] = torch.cat(results[obj]['integrate_anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            integrate_image_auroc = image_level_metrics(results, obj, "image-auroc", modality = 'integrate_pr_sp')
            integrate_image_ap = image_level_metrics(results, obj, "image-ap", modality = 'integrate_pr_sp')
            integrate_table.append(str(np.round(integrate_image_auroc * 100, decimals=1)))
            integrate_table.append(str(np.round(integrate_image_ap * 100, decimals=1)))
            integrate_image_auroc_list.append(integrate_image_auroc)
            integrate_image_ap_list.append(integrate_image_ap)  
        elif args.metrics == 'pixel-level':
            integrate_pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc", modality = 'integrate_anomaly_maps')
            integrate_pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro", modality = 'integrate_anomaly_maps')
            integrate_table.append(str(np.round(integrate_pixel_auroc * 100, decimals=1)))
            integrate_table.append(str(np.round(integrate_pixel_aupro * 100, decimals=1)))
            integrate_pixel_auroc_list.append(integrate_pixel_auroc)
            integrate_pixel_aupro_list.append(integrate_pixel_aupro)
        elif args.metrics == 'image-pixel-level':
            integrate_image_auroc = image_level_metrics(results, obj, "image-auroc", modality = 'integrate_pr_sp')
            integrate_image_ap = image_level_metrics(results, obj, "image-ap", modality = 'integrate_pr_sp')
            integrate_pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc", modality = 'integrate_anomaly_maps')
            integrate_pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro", modality = 'integrate_anomaly_maps')
            integrate_table.append(str(np.round(integrate_pixel_auroc * 100, decimals=1)))
            integrate_table.append(str(np.round(integrate_pixel_aupro * 100, decimals=1)))
            integrate_table.append(str(np.round(integrate_image_auroc * 100, decimals=1)))
            integrate_table.append(str(np.round(integrate_image_ap * 100, decimals=1)))
            integrate_image_auroc_list.append(integrate_image_auroc)
            integrate_image_ap_list.append(integrate_image_ap) 
            integrate_pixel_auroc_list.append(integrate_pixel_auroc)
            integrate_pixel_aupro_list.append(integrate_pixel_aupro)
        integrate_table_ls.append(integrate_table)
    if args.metrics == 'image-level':
        integrate_table_ls.append(['mean', 
                        str(np.round(np.mean(integrate_image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(integrate_image_ap_list) * 100, decimals=1))])
        integrate_results = tabulate(integrate_table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    
    elif args.metrics == 'pixel-level':
        integrate_table_ls.append(['mean', str(np.round(np.mean(integrate_pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(integrate_pixel_aupro_list) * 100, decimals=1))
                        ])
        integrate_results = tabulate(integrate_table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
    
    elif args.metrics == 'image-pixel-level':
        integrate_table_ls.append(['mean', str(np.round(np.mean(integrate_pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(integrate_pixel_aupro_list) * 100, decimals=1)), 
                        str(np.round(np.mean(integrate_image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(integrate_image_ap_list) * 100, decimals=1))])
        integrate_results = tabulate(integrate_table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'], tablefmt="pipe")
    
    logger.info("\n%s", integrate_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CoGeoAD", add_help=True)
    parser.add_argument("--data_path", type=str, default="./data/xxx/", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--num_views", type=int, default=9, help="views")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--point_size", type=int, default=336, help="save frequency")
    parser.add_argument("--feature_type", type=str, default='both', choices=['rgb', 'uni', 'both'], help="which features to use")
    parser.add_argument("--use_view_attention", type=int, default=1, help="whether to use view attention (1) or simple averaging (0)")
    parser.add_argument("--view_attention_type", type=str, default='ViewAttention', 
                        choices=['ViewAttention', 'ViewAttention_linear', 'ViewAttention_para'], 
                        help="type of view attention module to use")
    parser.add_argument("--use_weighted_sum", default='1', type=int, help="whether to use weighted sum for feature aggregation")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)