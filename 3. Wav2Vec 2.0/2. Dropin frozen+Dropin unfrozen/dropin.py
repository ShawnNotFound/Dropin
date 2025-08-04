import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayer
import copy
import os
class ProjectedEncoderLayer(nn.Module):
    def __init__(self, origin_layer, config, dropin_dim = 768 * 2, output_dim=768):
        super().__init__()
        self.origin_layer = origin_layer
        self.layer_addition = Wav2Vec2EncoderLayer(config)
        self.projector = nn.Linear(dropin_dim, output_dim)
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        out1 = self.origin_layer(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        ) # 原来的
        out2 = self.layer_addition(hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,)
        # last_hidden_state, optional attn weights
        hidden = torch.cat([out1[0], out2[0]], dim = 2)
        projected = self.projector(hidden)
        return (projected,)

class DropInEncoderLayer(Wav2Vec2EncoderLayer):
    """
    扩展的编码器层，支持不同的隐藏层维度
    """
    def __init__(self, config, input_dim=None, output_dim=None):
        # 如果指定了不同的输入输出维度，需要修改config
        if input_dim is not None or output_dim is not None:
            config = copy.deepcopy(config)
            if input_dim is not None:
                config.hidden_size = input_dim
            if output_dim is not None:
                # 这里需要添加投影层来处理输出维度变化
                self.needs_projection = True
                self.target_output_dim = output_dim
            else:
                self.needs_projection = False
        else:
            self.needs_projection = False
            
        super().__init__(config)
        
        # 如果需要投影，添加投影层
        if self.needs_projection:
            self.output_projection = nn.Linear(config.hidden_size, self.target_output_dim)
    
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        outputs = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        
        hidden = outputs[0]
        if self.needs_projection:
            hidden = self.output_projection(hidden)
            
        return (hidden,) + outputs[1:]

class DropInWav2Vec2(nn.Module):
    """
    支持DropIn操作的Wav2Vec2模型
    """
    def __init__(self, 
                 pretrained_model="/data/yl7622/dropin/wav2vec/wav2vec2_model", 
                 num_labels=2,
                 dropin_config=None,
                 mode='frozen'):
        """
        Args:
            pretrained_model: 预训练模型路径
            num_labels: 分类标签数量
            dropin_config: DropIn配置，格式为 {'layer_idx': new_hidden_size, ...}
            mode: 'frozen' 或 'unfrozen'
        """
        super().__init__()
        self.mode = mode
        self.dropin_config = dropin_config or {}
        
        # 加载基础配置和模型
        self.base_model = Wav2Vec2Model.from_pretrained(pretrained_model)
        
        final_hidden_size = self.base_model.config.hidden_size
        
        if mode == 'frozen':
            
            # 尝试载入 baseline 权重
            baseline_path = '/data/yl7622/dropin/wav2vec_baseline/checkpoints/best_model.pth'

            checkpoint = torch.load(baseline_path, map_location='cpu', weights_only=False)

            try:
                self.base_model.load_state_dict(checkpoint, strict=False)
                print(f"Loaded baseline weights from {baseline_path}")
            except Exception as e:
                print(f"Error loading baseline weights: {e}")

            self._freeze_base_model()
            self._add_dropin_layers()
                
            final_hidden_size = self.base_model.config.hidden_size

        else:
            # Unfrozen模式：从零开始训练
            self.base_model = Wav2Vec2Model.from_pretrained(pretrained_model)
            self._add_dropin_layers(freeze=False)
            final_hidden_size = self.base_model.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(final_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
        
    def _freeze_base_model(self):
        """冻结基础模型的所有参数"""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def _add_dropin_layers(self, freeze=True):
        """在frozen模式下添加DropIn层"""
        for layer_idx, new_hidden_size in self.dropin_config.items():
            if layer_idx >= len(self.base_model.encoder.layers):
                print(f"Warning: Layer {layer_idx} does not exist, skipping")
                continue
                
            original_layer = self.base_model.encoder.layers[layer_idx]

            # 创建新层
            new_layer = ProjectedEncoderLayer(original_layer, self.base_model.config, new_hidden_size, self.base_model.config.hidden_size)
            
            # 替换层
            self.base_model.encoder.layers[layer_idx] = new_layer
            if freeze:
                for param in new_layer.origin_layer.parameters():
                    param.requires_grad = False

    
    def _initialize_dropin_layer(self, new_layer, original_layer):
        """初始化DropIn层的参数"""
        try:
            # 获取原始层的state dict
            original_state = original_layer.state_dict()
            new_state = new_layer.state_dict()
            
            # 复制可以直接复制的参数
            for key in new_state.keys():
                if key in original_state and 'output_projection' not in key:
                    old_param = original_state[key]
                    new_param = new_state[key]
                    
                    if old_param.shape == new_param.shape:
                        # 形状相同，直接复制
                        new_state[key] = old_param.clone()
                    elif len(old_param.shape) == 2 and len(new_param.shape) == 2:
                        # 2D参数（如Linear层权重），复制到左上角
                        min_dim0 = min(old_param.shape[0], new_param.shape[0])
                        min_dim1 = min(old_param.shape[1], new_param.shape[1])
                        new_state[key][:min_dim0, :min_dim1] = old_param[:min_dim0, :min_dim1]
                        
                        # 初始化剩余部分为小随机值
                        if new_param.shape[0] > old_param.shape[0]:
                            nn.init.xavier_uniform_(new_state[key][old_param.shape[0]:, :])
                        if new_param.shape[1] > old_param.shape[1]:
                            nn.init.xavier_uniform_(new_state[key][:, old_param.shape[1]:])
                            
                    elif len(old_param.shape) == 1 and len(new_param.shape) == 1:
                        # 1D参数（如bias、LayerNorm），复制前面部分
                        min_dim = min(old_param.shape[0], new_param.shape[0])
                        new_state[key][:min_dim] = old_param[:min_dim]
                        
                        # 初始化剩余部分为0
                        if new_param.shape[0] > old_param.shape[0]:
                            new_state[key][old_param.shape[0]:] = 0.0
                else:
                    # 对于新添加的层（如output_projection），使用Xavier初始化
                    if 'output_projection' in key and 'weight' in key:
                        nn.init.xavier_uniform_(new_state[key])
                    elif 'output_projection' in key and 'bias' in key:
                        nn.init.zeros_(new_state[key])
            
            # 加载修改后的state dict
            new_layer.load_state_dict(new_state, strict=False)
            
            # 验证没有NaN值
            for name, param in new_layer.named_parameters():
                if torch.isnan(param).any():
                    print(f"Warning: NaN found in {name}, re-initializing")
                    if len(param.shape) == 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.zeros_(param)
                        
        except Exception as e:
            print(f"Warning: Could not initialize dropin layer: {e}")
    
    
    def forward(self, input_values, attention_mask=None):
        outputs = self.base_model(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        
        # 全局平均池化
        if attention_mask is not None:
            # 考虑attention mask的池化
            mask_expanded = attention_mask.unsqueeze(-1).float()
            hidden = hidden * mask_expanded
            pooled = hidden.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = hidden.mean(dim=1)
        
        return self.classifier(pooled)

# 测试代码
if __name__ == "__main__":
    # 创建测试模型
    model = DropInWav2Vec2(
        pretrained_model="/data/yl7622/dropin/wav2vec/wav2vec2_model",
        num_labels=2,
        dropin_config={5: 1024},
        mode='frozen'
    )
    
    # 测试前向传播
    batch_size = 2
    seq_len = 16000 * 4  # 4秒音频
    input_values = torch.randn(batch_size, seq_len)
    
    with torch.no_grad():
        output = model(input_values)
    
    print(f"Input shape: {input_values.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")
