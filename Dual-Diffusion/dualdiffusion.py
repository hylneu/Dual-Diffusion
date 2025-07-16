import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q_feat, kv_feat):
        q, k, v = self.q(q_feat), self.k(kv_feat), self.v(kv_feat)
        attn = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / q.size(-1) ** 0.5)
        return torch.matmul(attn, v)


# ---------- HIA-Refiner ���� -------------------------------------------------
class HIA_Refiner(nn.Module):
    """
    Hierarchical Identity Attention
   Image_adapter
    """

    def __init__(self, hidden_size: int = 1024, num_layers: int = 2):
        super().__init__()

        self.self_blocks = nn.ModuleList()
        self.text_ca = CrossAttention(hidden_size)  # prompt / tokenflow
        self.id_ca = CrossAttention(hidden_size)    # ID-token

        self.router = nn.Linear(hidden_size, hidden_size, bias=False)

        self.alpha = nn.Parameter(torch.zeros(1))

        # Adapter MLP
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )


        self._cached_text = None
        self._cached_id = None


    @torch.no_grad()
    def cache_cond(self, text_tokens: torch.Tensor, id_tokens: torch.Tensor):

        self._cached_text = text_tokens
        self._cached_id = id_tokens


    def forward(
        self,
        tokens: torch.Tensor,
        text_tokens: torch.Tensor = None,
        id_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        # cache_cond
        text_tokens = text_tokens if text_tokens is not None else self._cached_text
        id_tokens = id_tokens if id_tokens is not None else self._cached_id
        assert (
            text_tokens is not None and id_tokens is not None
        ), "HIA_Refiner: text_tokens �� id_tokens ����������"

        # 1)
        for blk in self.self_blocks:
            tokens = tokens + blk(tokens)  # residual

        # 2) prompt / tokenflow
        text_out = self.text_ca(tokens, text_tokens)

        # 3)
        w = F.softmax(
            torch.matmul(self.router(tokens), id_tokens.transpose(-2, -1))
            / tokens.size(-1) ** 0.5,
            dim=-1,
        )  # (B,Nq,Nid)
        routed_kv = torch.matmul(w, id_tokens)  # (B,Nq,D)
        id_out = self.id_ca(tokens, routed_kv)

        # 4) �������� + ����
        tokens = tokens + text_out + torch.sigmoid(self.alpha) * id_out

        # 5) Adapter MLP
        return self.adapter(tokens)

