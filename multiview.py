import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(dropout) if dropout else nn.Identity(),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(dropout) if dropout else nn.Identity()
                                  )
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                          nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        h = self.conv(x)
        return h + self.shortcut(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d, d_k, num_head):
        super().__init__()
        assert d % num_head == 0
        self.d = d
        self.d_k = d_k
        self.num_head = num_head
        self.head_dim = d_k // num_head

        self.qkv_proj = nn.Linear(d, 3*d_k)
        self.o_proj = nn.Linear(d_k, d)
                        
    def forward(self, x):
        """
        x: (batch_size, d)
        """
        B, d = x.size()
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(self.num_head, B, 3*self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        
        attn_logits  = torch.einsum("ijk,ilk->ijl", q, k) / math.sqrt(self.head_dim)
        attention    = F.softmax(attn_logits, dim=-1)
        
        values = torch.einsum("ijk,ikl->ijl", attention, v)
        
        values = values.permute(1, 0, 2).reshape(B, d)
        
        out = self.o_proj(values)
        return out
        
class TransformerBlock(nn.Module):
    def __init__(self, d, d_k, dim_out, num_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.mha = MultiHeadAttention(d, d_k, num_head)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Linear(d, dim_out)
        
    def forward(self, x):
        h = self.ln1(x)
        h = self.mha(h) + x
        x = h
        h = self.ln2(x)
        h = self.mlp(h)
        return h+x
    
class MultiView(nn.Module):
    def __init__(self, in_channels, out_channels, num_views, d, num_classes, image_size=28, dropout=0.2):
        super().__init__()
        self.num_views = num_views
        flatten_dim = out_channels * (image_size//8) * (image_size//8)  
        
        self.conv_backbone = nn.Sequential(
            ResidualBlock(in_channels, out_channels, dropout),
            ResidualBlock(out_channels, out_channels, dropout),
            ResidualBlock(out_channels, out_channels, dropout),
            ResidualBlock(out_channels, out_channels, dropout),
            nn.AdaptiveAvgPool2d((image_size//8, image_size//8)),
            nn.Flatten()
        )
        
        self.embedding = nn.Sequential(
            nn.Linear(flatten_dim, d),
            nn.LayerNorm(d),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        
        self.transformer = nn.Sequential(
            TransformerBlock(d, d, d, num_head=4),
            TransformerBlock(d, d, d, num_head=4),
            TransformerBlock(d, d, d, num_head=4)
        )
        
        self.pool = nn.MaxPool1d(num_views)
        self.classify  = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes)
        )
            
    def forward(self, x):
        """
        x: (batch_size, num_views, in_channels, image_size, image_size)
        Returns: (batch_size, num_classes)
        """
        b, v, c, h, w = x.size()
        
        x = x.reshape(b*v, c, h, w)
        x = self.conv_backbone(x)

        x = self.embedding(x)
        x = self.transformer(x)

        x = x.reshape(b, v, -1) 
        x = x.permute(0, 2, 1)
        x = self.pool(x) 
        x = x.squeeze(-1)
        x = self.classify(x)      

        return x