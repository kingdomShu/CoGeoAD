import CoGeoAD_CLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
from model import ViewAttention, DeLinearLayer, ViewAttention_linear, ViewAttention_para, WeightedSum

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def entropy_loss(weights):
    return -torch.mean(torch.sum(weights * torch.log(weights + 1e-8), dim=1))

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


def train(args):
    logger = get_logger(args.save_path)
    preprocess, target_transform, target_transform_pc = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = CoGeoAD_CLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()
    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, target_transform_pc = target_transform_pc, dataset_name = args.dataset, point_size = args.point_size,selected_views=list(range(args.num_views)))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
  
    ##########################################################################################
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)
    
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)
    
    # ViewAttention
    if args.view_attention_type == 'ViewAttention':
        view_attention = ViewAttention().to(device)
    elif args.view_attention_type == 'ViewAttention_linear':
        view_attention = ViewAttention_linear(input_dim=3*args.image_size*args.image_size).to(device)
    elif args.view_attention_type == 'ViewAttention_para':
        view_attention = ViewAttention_para(args.num_views).to(device)
    else:
        raise ValueError(f"Unknown view attention type: {args.view_attention_type}")

    use_delinear = len(args.features_list) >= 4

    train_para = []
    train_para.extend(prompt_learner.parameters())
    train_para.extend(view_attention.parameters())

    feature_fusion_layer = None
    
 
    if use_delinear:
        feature_fusion_layer = DeLinearLayer(768, len(args.features_list) , "ViT-L/14@336px").to(device)
        feature_fusion_layer.train()
        train_para.extend(feature_fusion_layer.parameters())
        num_fusion_weights = 4 
    else:
        logger.info("Features list length < 4, skipping DeLinearLayer1 initialization.")
        num_fusion_weights = len(args.features_list) 

    # [MODIFIED] Added conditional initialization for WeightedSum
    weighted_sum_model = None
    if args.use_weighted_sum:
        weighted_sum_model = WeightedSum(dataset = args.dataset).to(device)
        weighted_sum_model.train()
        train_para.extend(weighted_sum_model.parameters())
    else:
        logger.info("WeightedSum disabled, using simple average.")

    ##########################################################################################
    optimizer = torch.optim.Adam(
        train_para,
        lr=args.learning_rate, 
        betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    model.eval()
    prompt_learner.train()
    view_attention.train()
    
    for epoch in tqdm(range(args.epoch)):
        
        d2_pixel_loss_list = []
        d3_global_loss_list = []
        d3_point_loss_list = []
        d2_image_loss_list = []
        for items in tqdm(train_dataloader):
            
            label =  items['anomaly']
            rgb_render_image = items['rgb_render_img'].to(device)
            uni_render_image = items['uni_render_img'].to(device)
            b, nv, c, h, w = rgb_render_image.shape
            rgb_render_image = rgb_render_image.reshape(-1, c, h, w)
            uni_render_image = uni_render_image.reshape(-1, c, h, w)
            d2_render_anomaly = items['d2_render_anomaly'].to(device)
            d2_3d_cor = items['d2_3d_cor'].to(device)
            non_zero_index = items['non_zero_index'].to(device)
            non_zero_index = non_zero_index.unsqueeze(1).repeat(1, nv, 1, 1)

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0


            render_gt_mask = items['d2_render_gt'].to(device)
            render_gt_mask[render_gt_mask > 0.5], render_gt_mask[render_gt_mask <= 0.5] = 1, 0

            render_gt_mask = render_gt_mask.reshape(-1, 1, h, w)
    
            with torch.no_grad():

                if args.feature_type in ['rgb', 'both']:
                    
                    rgb_image_features, rgb_patch_features = model.encode_image(rgb_render_image, args.features_list, DPAM_layer = 20)
                    rgb_image_features = rgb_image_features / rgb_image_features.norm(dim=-1, keepdim=True)
                
                if args.feature_type in ['uni', 'both']:
                    
                    uni_image_features, uni_patch_features = model.encode_image(uni_render_image, args.features_list, DPAM_layer = 20)
                    uni_image_features = uni_image_features / uni_image_features.norm(dim=-1, keepdim=True)
           ####################################
            rgb_prompts, rgb_tokenized_prompts, rgb_compound_prompts_text = prompt_learner(cls_id = None)
            uni_prompts, uni_tokenized_prompts, uni_compound_prompts_text = prompt_learner(cls_id = None)
            
            if args.feature_type in ['rgb', 'both']:
                rgb_text_features = model.encode_text_learn(rgb_prompts, rgb_tokenized_prompts, rgb_compound_prompts_text).float()
                rgb_text_features = torch.stack(torch.chunk(rgb_text_features, dim = 0, chunks = 2), dim = 1)
                rgb_text_features = rgb_text_features/rgb_text_features.norm(dim=-1, keepdim=True)
            
            if args.feature_type in ['uni', 'both']:
                uni_text_features = model.encode_text_learn(uni_prompts, uni_tokenized_prompts, uni_compound_prompts_text).float()
                uni_text_features = torch.stack(torch.chunk(uni_text_features, dim = 0, chunks = 2), dim = 1)
                uni_text_features = uni_text_features/uni_text_features.norm(dim=-1, keepdim=True)
            
            # Apply DPAM surgery
            if args.feature_type in ['rgb', 'both']:
                rgb_text_probs = rgb_image_features.unsqueeze(1) @ rgb_text_features.permute(0, 2, 1)
                rgb_text_probs = rgb_text_probs[:, 0, ...]/0.07
            if args.feature_type in ['uni', 'both']:
                uni_text_probs = uni_image_features.unsqueeze(1) @ uni_text_features.permute(0, 2, 1)
                uni_text_probs = uni_text_probs[:, 0, ...]/0.07
            
            d2_render_anomaly = d2_render_anomaly.reshape(-1,)
            
            d2_image_loss = 0
            if args.feature_type in ['rgb', 'both']:
                d2_image_loss += F.cross_entropy(rgb_text_probs.squeeze(), d2_render_anomaly.long().cuda())
            if args.feature_type in ['uni', 'both']:
                d2_image_loss += F.cross_entropy(uni_text_probs, d2_render_anomaly.long().cuda())
            
            #########################################################################
            if args.feature_type in ['uni', 'both']:
                uni_text_probs = torch.chunk(uni_text_probs,  nv, dim = 0)
            if args.feature_type in ['rgb', 'both']:
                rgb_text_probs = torch.chunk(rgb_text_probs,  nv, dim = 0)
            
            #########################################################################
            if args.use_view_attention :
                uni_images = rgb_render_image.reshape(b, nv, c, h, w)
                view_weights = view_attention([uni_images[:, i] for i in range(nv)])
                view_weights_expanded = view_weights.unsqueeze(-1)
                loss_entropy = -0.1 * entropy_loss(view_weights) 
                d3_global_loss = 0
                if args.feature_type in ['rgb', 'both']:
                    rgb_text_probs_stack = torch.stack(rgb_text_probs, dim=1) 
                    rgb_text_probs_weighted = (rgb_text_probs_stack * view_weights_expanded).sum(dim=1)
                    d3_global_loss += F.cross_entropy(rgb_text_probs_weighted, label.long().cuda())
                    
                if args.feature_type in ['uni', 'both']:
                    uni_text_probs_stack = torch.stack(uni_text_probs, dim=1)
                    uni_text_probs_weighted = (uni_text_probs_stack * view_weights_expanded).sum(dim=1)
                    d3_global_loss += F.cross_entropy(uni_text_probs_weighted, label.long().cuda())
            else:
                loss_entropy = 0
                d3_global_loss = 0
                
                if args.feature_type in ['rgb', 'both']:
                    
                    rgb_probs = torch.stack(rgb_text_probs, dim = 1).mean(1)
                    d3_global_loss += F.cross_entropy(rgb_probs, label.long().cuda())
                if args.feature_type in ['uni', 'both']:
                    uni_probs = torch.stack(uni_text_probs, dim = 1).mean(1)
                    d3_global_loss += F.cross_entropy(uni_probs, label.long().cuda())
            
            # fusion layer
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
            
            similarity_maps_rgb = []
            similarity_maps_uni = []
            
            if args.feature_type in ['rgb', 'both']:
                for i in range(len(rgb_fused_features_groups)):
                    rgb_patch_feature = rgb_fused_features_groups[i]
                    rgb_patch_feature = rgb_patch_feature / rgb_patch_feature.norm(dim=-1, keepdim=True)
                    rgb_similarity, _ = CoGeoAD_CLIP_lib.compute_similarity(rgb_patch_feature, rgb_text_features[0])
                    rgb_similarity_map = CoGeoAD_CLIP_lib.get_similarity_map(rgb_similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_maps_rgb.append(rgb_similarity_map)
            
            if args.feature_type in ['uni', 'both']:
                for i in range(len(uni_fused_features_groups)):
                    uni_patch_feature = uni_fused_features_groups[i]
                    uni_patch_feature = uni_patch_feature / uni_patch_feature.norm(dim=-1, keepdim=True)
                    uni_similarity, _ = CoGeoAD_CLIP_lib.compute_similarity(uni_patch_feature, uni_text_features[0])
                    uni_similarity_map = CoGeoAD_CLIP_lib.get_similarity_map(uni_similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_maps_uni.append(uni_similarity_map)
            
            if args.feature_type in ['rgb', 'both']:
                if args.use_weighted_sum:
                    rgb_similarity_map, _ = weighted_sum_model(similarity_maps_rgb)
                else:
                    rgb_similarity_map = torch.stack(similarity_maps_rgb, dim=0).mean(dim=0)
            
            if args.feature_type in ['uni', 'both']:
                if args.use_weighted_sum:
                    uni_similarity_map, _ = weighted_sum_model(similarity_maps_uni)
                else:
                    uni_similarity_map = torch.stack(similarity_maps_uni, dim=0).mean(dim=0)

            
####################################################################
            if args.feature_type == 'rgb':
                if args.use_view_attention :
                    d3_similarity_map = back_to_3d(rgb_similarity_map.permute(0, 2, 3, 1), d2_3d_cor, non_zero_index, view_weights).permute(0, 3, 1, 2)
                else:
                    d3_similarity_map = back_to_3d(rgb_similarity_map.permute(0, 2, 3, 1), d2_3d_cor, non_zero_index).permute(0, 3, 1, 2)
            elif args.feature_type == 'uni':
                if args.use_view_attention :
                    d3_similarity_map = back_to_3d(uni_similarity_map.permute(0, 2, 3, 1), d2_3d_cor, non_zero_index, view_weights).permute(0, 3, 1, 2)
                else:
                    d3_similarity_map = back_to_3d(uni_similarity_map.permute(0, 2, 3, 1), d2_3d_cor, non_zero_index).permute(0, 3, 1, 2)
            elif args.feature_type == 'both':
                if args.use_view_attention :
                    d3_rgb_similarity_map = back_to_3d(rgb_similarity_map.permute(0, 2, 3, 1), d2_3d_cor, non_zero_index, view_weights).permute(0, 3, 1, 2)
                    d3_uni_similarity_map = back_to_3d(uni_similarity_map.permute(0, 2, 3, 1), d2_3d_cor, non_zero_index, view_weights).permute(0, 3, 1, 2)
                else:
                    d3_rgb_similarity_map = back_to_3d(rgb_similarity_map.permute(0, 2, 3, 1), d2_3d_cor, non_zero_index).permute(0, 3, 1, 2)
                    d3_uni_similarity_map = back_to_3d(uni_similarity_map.permute(0, 2, 3, 1), d2_3d_cor, non_zero_index).permute(0, 3, 1, 2)
                d3_similarity_map = torch.max(d3_rgb_similarity_map, d3_uni_similarity_map)
####################################################################

            d3_point_loss = 0
            d3_point_loss += loss_dice(d3_similarity_map[:, 1, :, :], gt)
            d3_point_loss += loss_dice(d3_similarity_map[:, 0, :, :], 1-gt)

            d2_pixel_loss = 0
            if args.feature_type in ['rgb', 'both']:
                d2_pixel_loss += loss_focal(rgb_similarity_map, render_gt_mask)
                d2_pixel_loss += loss_dice(rgb_similarity_map[:, 1, :, :], render_gt_mask)
                d2_pixel_loss += loss_dice(rgb_similarity_map[:, 0, :, :], 1-render_gt_mask)
            if args.feature_type in ['uni', 'both']:
                d2_pixel_loss += loss_focal(uni_similarity_map, render_gt_mask)
                d2_pixel_loss += loss_dice(uni_similarity_map[:, 1, :, :], render_gt_mask)
                d2_pixel_loss += loss_dice(uni_similarity_map[:, 0, :, :], 1-render_gt_mask)
            
            optimizer.zero_grad()
            (d3_point_loss + d2_pixel_loss + d3_global_loss + d2_image_loss + loss_entropy).backward()
            optimizer.step()
            d2_pixel_loss_list.append(d2_pixel_loss.item())
            d3_point_loss_list.append(d3_point_loss.item())
            d3_global_loss_list.append(d3_global_loss.item())
            d2_image_loss_list.append(d2_image_loss.item())
        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], d2_pixel_loss:{:.4f}, d3_point_loss:{:.4f}, d3_global_loss:{:.4f}, d2_image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(d2_pixel_loss_list), np.mean(d3_point_loss_list), np.mean(d3_global_loss_list), np.mean(d2_image_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            save_dict = {
                "prompt_learner": prompt_learner.state_dict(),
                "view_attention": view_attention.state_dict(),
            }
            if use_delinear and feature_fusion_layer is not None:
                save_dict["feature_fusion_layer"] = feature_fusion_layer.state_dict()

            if weighted_sum_model is not None:
                save_dict["weighted_sum_model"] = weighted_sum_model.state_dict()
            
            torch.save(save_dict, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CoGeoAD", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/xxx/", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--num_views", type=int, default=9, help="views")
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--point_size", type=int, default=336, help="save frequency")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=3407, help="random seed")
    parser.add_argument("--feature_type", type=str, default='both', choices=['rgb', 'uni', 'both'], help="which features to use")
    parser.add_argument("--view_attention_type", type=str, default='ViewAttention', 
                        choices=['ViewAttention', 'ViewAttention_linear', 'ViewAttention_para'], 
                        help="type of view attention module to use")
    parser.add_argument("--use_view_attention", default='1',type=int, help="whether to use view attention or simple averaging")
    parser.add_argument("--use_weighted_sum", default='1', type=int, help="whether to use weighted sum for feature aggregation")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)