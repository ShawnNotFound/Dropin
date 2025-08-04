import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayer

class ProjectedEncoderLayer(Wav2Vec2EncoderLayer):
    def __init__(self, config, output_dim=768):
        super().__init__(config)
        self.projector = nn.Linear(config.hidden_size, output_dim)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        out = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden = out[0]
        projected = self.projector(hidden)
        return (projected,)

def copy_state_manually(origin_layer, new_layer):
    old_sd = origin_layer.state_dict()
    new_sd = new_layer.state_dict()
    for k in new_sd.keys():
        if k not in old_sd:
            print(f"[skip] {k} not found in old layer")
            continue

        old_param = old_sd[k]
        new_param = new_sd[k]

        # 判断维度对不上的情况
        if old_param.shape == new_param.shape:
            new_sd[k] = old_param
        elif len(old_param.shape) == 2:
            # Linear weights: expand top-left corner
            new_sd[k][:old_param.shape[0], :old_param.shape[1]] = old_param
        elif len(old_param.shape) == 1:
            # Bias / LayerNorm
            new_sd[k][:old_param.shape[0]] = old_param
        else:
            print(f"[warn] Shape mismatch for {k}: old {old_param.shape}, new {new_param.shape}")

    new_layer.load_state_dict(new_sd, strict=False)


class Wav2Vec2WithCustomLayer(nn.Module):
    def __init__(self, pretrained_model="/data/yl7622/dropin/wav2vec/wav2vec2_model", num_labels=2):
        super().__init__()
        config = Wav2Vec2Config.from_pretrained(pretrained_model)
        base_model = Wav2Vec2Model(config)

        # 加载预训练参数
        state_dict = Wav2Vec2Model.from_pretrained(pretrained_model).state_dict()
        base_model.load_state_dict(state_dict)

        # 替换第6层（第7层）为自定义层，第5层feedforward
        layer5_dropin = ProjectedEncoderLayer(config, output_dim=1024)
        config_dropin=Wav2Vec2Config(
                hidden_size=1024,
                num_attention_heads=16,
                intermediate_size=4096,
                hidden_dropout=0.1,
                attention_dropout=0.1,
            )
        layer6_dropin = ProjectedEncoderLayer(config_dropin, output_dim=768)
        origin_layer5 = base_model.encoder.layers[5]
        copy_state_manually(origin_layer5, layer5_dropin)
        base_model.encoder.layers[5] = layer5_dropin  # 注意是索引5

        origin_layer6 = base_model.encoder.layers[6]
        copy_state_manually(origin_layer6, layer6_dropin)
        base_model.encoder.layers[6] = layer6_dropin  # 注意是索引6

        self.wav2vec2 = base_model
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.classifier(pooled)

# if __name__ == "__main__":
flac_path = "/data/yl7622/dropin/dataset/LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac"
processor = Wav2Vec2Processor.from_pretrained("/data/yl7622/dropin/wav2vec/wav2vec2_model/wav2vec-base")

# 读取音频，返回 waveform (channel, samples) 和 sample_rate
waveform, sample_rate = torchaudio.load(flac_path)
input_audio = waveform.squeeze()
inputs = processor(input_audio, sampling_rate=sample_rate, return_tensors="pt")

model = Wav2Vec2WithCustomLayer()
model.eval()

with torch.no_grad():
    logits = model(**inputs)

print("Logits shape:", logits.shape)  # 应该是 [2, 2]
print("Logits:", logits)