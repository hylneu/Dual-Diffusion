# flowtok.py
import torch
import torch.nn as nn

class ViTPatchTokenizer(nn.Module):
    """

    """
    def __init__(self, patch_size=16, in_ch=3, hidden_size=1024):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, hidden_size,
                              kernel_size=patch_size,
                              stride=patch_size,
                              bias=False)

    def forward(self, x):                                    # x: (B, 3, H, W)
        x = self.proj(x)                                     # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)                     # (B, N_i, D);  N_i = H/P * W/P
        return x
