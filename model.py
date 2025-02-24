import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class EnhancedMoE(nn.Module):
    def __init__(self, input_dim, num_experts=12, expert_dim=1024, dropout_rate=0.1):
        super(EnhancedMoE, self).__init__()
        self.num_experts = num_experts
        # More sophisticated experts with two layers
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(expert_dim, expert_dim)
            ) for _ in range(num_experts)
        ])
        # Improved gating with attention-like mechanism
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, num_experts)
        )
        self.layer_norm = nn.LayerNorm(expert_dim)

    def forward(self, x):
        gating_scores = F.softmax(self.gating_network(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(gating_scores.unsqueeze(-1) * expert_outputs, dim=1)
        return self.layer_norm(output)

class UltraSmarterModel(nn.Module):
    def __init__(
        self,
        text_model_name="bert-base-uncased",
        image_dim=2048,
        audio_dim=512,
        num_classes=None,
        hidden_dim=1024
    ):
        super(UltraSmarterModel, self).__init__()
        
        # Text processing
        self.text_config = AutoConfig.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # Enhanced modality experts
        self.image_expert = EnhancedMoE(image_dim, expert_dim=hidden_dim)
        self.audio_expert = EnhancedMoE(audio_dim, expert_dim=hidden_dim)
        
        # Cross-attention between modalities
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Fusion and output
        fused_dim = hidden_dim * 3  # Text + Image + Audio
        self.fusion_layer = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Flexible output layer (classification or regression)
        self.output_dim = num_classes if num_classes else hidden_dim
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)
        
        # Additional improvements
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_input, image_input, audio_input):
        # Text features from CLS token
        text_features = self.text_encoder(**text_input).last_hidden_state[:, 0, :]
        text_features = self.dropout(F.relu(text_features))
        
        # Process image and audio through enhanced MoE
        image_features = self.image_expert(image_input)
        audio_features = self.audio_expert(audio_input)
        
        # Reshape for cross-attention (batch_size, seq_len=1, embed_dim)
        text_features = text_features.unsqueeze(1)
        image_features = image_features.unsqueeze(1)
        audio_features = audio_features.unsqueeze(1)
        
        # Cross-attention between modalities
        modality_features = torch.cat([text_features, image_features, audio_features], dim=1)
        attn_output, _ = self.cross_attention(
            modality_features, modality_features, modality_features
        )
        
        # Fuse features
        fused_features = attn_output.reshape(attn_output.size(0), -1)
        fused_features = self.fusion_layer(fused_features)
        fused_features = self.layer_norm(fused_features)
        
        # Final output
        output = self.output_layer(fused_features)
        
        # Apply softmax/sigmoid if classification
        if self.output_dim > 1:
            return F.softmax(output, dim=-1)
        return output

# Example usage
if __name__ == "__main__":
    # Sample inputs
    batch_size = 4
    model = UltraSmarterModel(num_classes=10)  # For 10-class classification
    
    text_input = {
        "input_ids": torch.randint(0, 1000, (batch_size, 128)),
        "attention_mask": torch.ones(batch_size, 128)
    }
    image_input = torch.randn(batch_size, 2048)
    audio_input = torch.randn(batch_size, 512)
    
    # Forward pass
    output = model(text_input, image_input, audio_input)
    print(f"Output shape: {output.shape}")  # Should be [batch_size, 10]
