import math
import torch.nn as nn


class LoRA(nn.Module):

    def __init__(self, embed_dim, r):
        super().__init__()

        self.w_A_q = nn.Linear(embed_dim, r, bias=False)
        self.w_B_q = nn.Linear(r, embed_dim, bias=False)

        self.w_A_v = nn.Linear(embed_dim, r, bias=False)
        self.w_B_v = nn.Linear(r, embed_dim, bias=False)

        self.reset_parameters()

    def forward(self, q, k, v, mha_q, mha_k, mha_v):
        # tgt_len, bsz, embed_dim = q.shape
        # src_len, _, _ = k.shape

        q = self.w_B_q(self.w_A_q(q))
        v = self.w_B_v(self.w_A_v(v))

        q = q + mha_q
        k = mha_k
        v = v + mha_v

        return q, k, v

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_A_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_B_q.weight)

        nn.init.kaiming_uniform_(self.w_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_B_v.weight)
