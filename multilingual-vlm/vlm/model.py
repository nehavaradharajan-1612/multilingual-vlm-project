import torch
from torch import nn
from torch.nn import functional as F

class VisionLanguageClassifier(nn.Module):
    """Trainable CNN image encoder, frozen multilingual semantic class prototypes."""
    def __init__(self,prototypes):
        super().__init__()
        if prototypes.ndim!=3 or prototypes.shape[:2]!=(3,15):
            raise ValueError('Expected 3 languages × 15 classes × embedding dimension')
        self.register_buffer('prototypes',F.normalize(prototypes.float(),dim=-1))
        self.vision=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.GroupNorm(4,32),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.GroupNorm(8,128),nn.GELU(),nn.MaxPool2d(2),
            nn.Flatten(),nn.Linear(128*8*8,256),nn.GELU(),nn.Dropout(.2),
            nn.Linear(256,prototypes.shape[-1]))
    def forward(self,x):
        z=F.normalize(self.vision(x),dim=-1)
        return 20*torch.einsum('bd,lcd->blc',z,self.prototypes)
