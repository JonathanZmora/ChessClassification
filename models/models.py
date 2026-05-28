import torch
import torch.nn as nn
from torchvision import models


model_names = {
    "convnext_transformer",
    "convnext_fine_tuned_final_stage"
}


class ConvTransformer(nn.Module):
    """
    Hybrid CNN-Transformer model.
    Extracts features using a frozen pre-trained CNN backbone and models 
    global context with a Transformer Encoder.

    Args:
        saved_model_path (str): Path to the pre-trained CNN model object.
        num_classes (int, optional): Number of output classes. Defaults to 13.
        d_square (int, optional): Input image crop dimension size. Defaults to 112.
        d_model (int, optional): Transformer embedding dimension. Defaults to 768.
        nhead (int, optional): Number of attention heads. Defaults to 8.
        num_layers (int, optional): Number of Transformer Encoder layers. Defaults to 4.
    """
    def __init__(self, saved_model_path, num_classes=13, d_square=112, d_model=768, nhead=8, num_layers=4):
        super().__init__()

        self.d_square = d_square
        
        self.backbone = torch.load(saved_model_path, weights_only=False)
        
        # Remove classification head
        self.backbone.classifier[2] = nn.Identity()
        
        # Freeze the backone
        self.backbone.eval() 
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 64, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            batch_first=True,
            dropout=0.1
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # New classification head
        self.classifier = nn.Linear(d_model, num_classes)

    
    def forward(self, x):
        if x.dim() == 5:
            B = x.size(0)
            x = x.view(B * 64, 3, self.d_square, self.d_square)
        elif x.dim() == 4:
            B = x.size(0) // 64
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Saved model forward pass
        with torch.no_grad():
            features = self.backbone(x)
            
        # Reshape for the Transformer Encoder
        features = features.view(B, 64, 768)
        
        # Add positional encoding
        features = features + self.pos_encoder
        
        # Pass through Transformer Encoder
        transformer_out = self.transformer(features)
        
        # Classification
        logits = self.classifier(transformer_out)
        
        return logits.view(B * 64, -1)


class DinoTransformer(nn.Module):
    """
    Vision Transformer model.
    Extracts features using a frozen pre-trained DINOv2 (ViT-Big) backbone 
    and models global context via a Transformer encoder.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 13.
        d_square (int, optional): Input image crop dimension size. Defaults to 112.
        nhead (int, optional): Number of attention heads. Defaults to 8.
        num_layers (int, optional): Number of Transformer Encoder layers. Defaults to 4.
    """
    def __init__(self, num_classes=13, d_square=112, nhead=8, num_layers=4):
        super().__init__()

        self.d_square = d_square
        
        # Load DINOv2 Big as the backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
        
        # Freeze the backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 64, 768))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768, 
            nhead=nhead, 
            dim_feedforward=768 * 4, 
            batch_first=True, 
            dropout=0.1
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        if x.dim() == 5:
            B = x.size(0)
            x = x.view(B * 64, 3, self.d_square, self.d_square)
        elif x.dim() == 4:
            B = x.size(0) // 64
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
            
        # Backbone forward pass
        with torch.no_grad():
            features = self.backbone(x)
            
        # Reshape for Transformer Encoder
        features = features.view(B, 64, 768)

        # Add positional encoding
        features = features + self.pos_encoder
        
        # Pass through Transformer Encoder
        transformer_out = self.transformer(features)

        # Classification
        logits = self.classifier(transformer_out)
        
        return logits.view(B * 64, -1)


def init_model(model_name, saved_model_path):
    if model_name not in model_names:
        raise ValueError("model name not in allowed names list")

    if "transformer" in model_name:
        model = ConvTransformer(saved_model_path)

    else:
        model = torch.load(saved_model_path, weights_only=False)
    
        for param in model.parameters():
            param.requires_grad = False
    
        # Final three CNBlocks
        for param in model.features[-1].parameters():
            param.requires_grad = True
    
        # Final LayerNorm
        for param in model.classifier[0].parameters():
            param.requires_grad = True
    
        # Final Linear layer
        for param in model.classifier[2].parameters():
            param.requires_grad = True
        
    return model
        