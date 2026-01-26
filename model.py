from torch import Tensor, nn
import torch
from torch.nn import functional as F
import numpy as np

class WeightedSum(nn.Module):

    def __init__(self, dataset):
        if dataset == "mvtec3d":
            init_values=[0.02, 0.02, 0.02, 0.94]
        elif dataset == "eyecandies":
            init_values=[0.25, 0.25, 0.25, 0.25]
        super(WeightedSum, self).__init__()
        weights_tensor = torch.tensor(init_values, dtype=torch.float32)
        self.raw_weights = nn.Parameter(torch.log(weights_tensor + 1e-8))

    def forward(self, input_data):

        normalized_weights = F.softmax(self.raw_weights, dim=0)
        weighted_sum = 0
        for i, tensor in enumerate(input_data):
           
            weighted_sum += normalized_weights[i] * tensor
            
        return weighted_sum, normalized_weights

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        # 1. 线性投影层
        self.rgb_proj = nn.Linear(dim, dim)
        self.uni_proj = nn.Linear(dim, dim)

        self.attention = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm_rgb = nn.LayerNorm(dim)
        self.norm_uni = nn.LayerNorm(dim)
        

        
    def forward(self, rgb_features, uni_features):
        """
        Args:
            rgb_features: [Batch, Seq_Len, Dim]  (注意：这里通常是 B, L, C)
            uni_features: [Batch, Seq_Len, Dim]
        """
        
        rgb_residual = rgb_features
        uni_residual = uni_features

        rgb_enhanced, _ = self.attention(query=rgb_features, key=uni_features, value=uni_features)

        uni_enhanced, _ = self.attention(query=uni_features, key=rgb_features, value=rgb_features)

        rgb_final = self.norm_rgb(rgb_residual + rgb_enhanced)
        uni_final = self.norm_uni(uni_residual + uni_enhanced)
        
        return rgb_final, uni_final
    
class DynamicFeatureSelector(nn.Module):
    def __init__(self, feature_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.attention_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim // 8),
            nn.ReLU(),
            nn.Linear(feature_dim // 8, num_layers),

        )
        

        bias_init = torch.zeros(num_layers)
        bias_init[-1] = 5.0  
        self.layer_bias = nn.Parameter(bias_init)

        self.importance_evaluator = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])
        
    def forward(self, features_list):
        # features_list: List of tensors with shape [B, L, C]
        batch_size = features_list[0].shape[0]
        
        reference_feature = features_list[-1].mean(dim=1)  # [B, C]

        raw_logits = self.attention_layer(reference_feature)  # [B, num_layers]

        biased_logits = raw_logits + self.layer_bias

        attention_weights = F.softmax(biased_logits, dim=-1) # [B, num_layers]

        weighted_features = []
        for i, feature in enumerate(features_list):
            weight = attention_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            weighted_feature = feature * weight  # [B, L, C]
            weighted_features.append(weighted_feature)
        
        fused_features = torch.stack(weighted_features, dim=-1).sum(dim=-1)  # [B, L, C]
        
        return fused_features, attention_weights

class DeLinearLayer(nn.Module):
    def __init__(self,  output_dim, num_layers, model_name):
        super().__init__()
        self.num_layers = num_layers
        # 将num_layers分为4组
        self.groups = 4
        self.layers_per_group = num_layers // self.groups
        remainder = num_layers % self.groups
        
        # 计算每组的层数，确保总数等于num_layers
        self.layers_in_group = [self.layers_per_group + (1 if i < remainder else 0) for i in range(self.groups)]
        
        # 为每组特征创建独立的DynamicFeatureSelector
        self.dynamic_selectors = nn.ModuleList([
            DynamicFeatureSelector(output_dim, self.layers_in_group[i]) 
            for i in range(self.groups)
        ])
        
    def forward(self, features_list):

        group_start_idx = 0
        fused_features_groups = []
        
        for group_idx in range(self.groups):
            # 获取当前组的特征
            group_end_idx = group_start_idx + self.layers_in_group[group_idx]
            group_features = features_list[group_start_idx:group_end_idx]
            
            # 动态选择和融合当前组的特征
            fused_features, attention_weights = self.dynamic_selectors[group_idx](group_features)
            fused_features_groups.append(fused_features)
            
            # 更新起始索引
            group_start_idx = group_end_idx
        
        # 返回4组特征，每组都保持独立
        return fused_features_groups
class DeLinearLayer_fusion(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, model_name):
        super().__init__()
        self.num_layers = num_layers
        self.dynamic_selector = DynamicFeatureSelector(output_dim, num_layers)
        
        # 为每层特征创建独立的投影层
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
                nn.GELU()
            ) for _ in range(num_layers)
        ])
        
    def forward(self, features_list):
        # 对每层特征进行投影
        projected_features = []
        for i, features in enumerate(features_list):
            if len(features.shape) == 3:
                features = features[:, 1:, :]
            projected = self.projections[i](features)
            projected_features.append(projected)
        
        # 动态选择和融合特征
        fused_features, attention_weights = self.dynamic_selector(projected_features)
        
        return fused_features.unsqueeze(0)
class ViewAttention_linear(nn.Module):
    def __init__(self, input_dim=3*336*336, hidden_dim=512, temperature_init=1.0):
        super(ViewAttention_linear, self).__init__()
        
        # LinearMLP
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.scorer = nn.Linear(hidden_dim // 2, 1)
        
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        
        
        self._initialize_weights()

    def _initialize_weights(self):
        # 通用初始化
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

        nn.init.constant_(self.scorer.weight, 0)
        nn.init.constant_(self.scorer.bias, 0)

    def forward(self, x_list):
        # x_list: list of tensors, each [B, C, H, W]
        if isinstance(x_list, list):
            x_stack = torch.stack(x_list, dim=1)  # [B, NV, C, H, W]
        else:
            x_stack = x_list

        B, NV, C, H, W = x_stack.shape

        x_flat = x_stack.view(B * NV, -1) # [B*NV, C*H*W]
        encoded_feats = self.encoder(x_flat) # [B*NV, hidden_dim//2]
        raw_scores = self.scorer(encoded_feats) # [B*NV, 1]
        view_scores = raw_scores.view(B, NV)    # [B, NV]
        # Softmax 
        scale = 0.1 
        final_weights = F.softmax((view_scores * scale) / self.temperature, dim=1)
        return final_weights


class ViewAttention_para(nn.Module):
    def __init__(self, num_views=10):
        super(ViewAttention_para, self).__init__()
        self.view_logits = nn.Parameter(torch.zeros(1, num_views))
        
    def forward(self, x_list):
        
        if isinstance(x_list, list):
            B = x_list[0].shape[0]
        else:
            B = x_list.shape[0]
        weights = F.softmax(self.view_logits, dim=1) # [1, NV]
        
        
        return weights.expand(B, -1)
class SimpleCNNEncoder(nn.Module):

    def __init__(self, output_dim=512):
        super(SimpleCNNEncoder, self).__init__()
        # Input: [Batch, 3, 336, 336]

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3) # -> [32, 84, 84]
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2) # -> [64, 42, 42]
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # -> [128, 21, 21]
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # -> [256, 11, 11]
        self.bn4 = nn.BatchNorm2d(256)

        self.gap = nn.AdaptiveAvgPool2d((1, 1)) 

        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        
        x = self.gap(x) # [B, 256, 1, 1]
        x = x.view(x.size(0), -1) # Flatten -> [B, 256]
        x = self.fc(x) # -> [B, output_dim]
        return x

class ViewAttention(nn.Module):
    def __init__(self, hidden_dim=512, temperature_init=1.0):
        super(ViewAttention, self).__init__()
        
        # 1. 核心特征提取 (保留你的 CNN)
        self.pixel_encoder = SimpleCNNEncoder(output_dim=hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(hidden_dim // 2, 1) # 输出标量分数
        )
        
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

        if isinstance(self.scorer[-1], nn.Linear):
            nn.init.constant_(self.scorer[-1].weight, 0)
            nn.init.constant_(self.scorer[-1].bias, 0)

    def forward(self, x_list):
        # 处理输入堆叠
        if isinstance(x_list, list):
            x_stack = torch.stack(x_list, dim=1) 
        else:
            x_stack = x_list

        B, NV, C, H, W = x_stack.shape
        x_flat = x_stack.view(B * NV, C, H, W)

        
        encoded_feats = self.pixel_encoder(x_flat) # [B*NV, hidden_dim]
        

        raw_scores = self.scorer(encoded_feats)    # [B*NV, 1]
        
      
        view_scores = raw_scores.view(B, NV)       # [B, NV]
        

        scale = 0.1 
        

        final_weights = F.softmax((view_scores * scale) / self.temperature, dim=1)
        
        return final_weights








