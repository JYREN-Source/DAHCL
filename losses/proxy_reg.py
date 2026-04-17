# losses/proxy_reg.py
import torch
import torch.nn.functional as F


def proxy_separation_loss(proxy_weight: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """
    proxy 分离正则：
      - 约束不同类 proxy 之间的余弦相似度不要太大
      - 对 cos(w_i, w_j) > margin 的部分进行二次惩罚

    参数
    ----
    proxy_weight : (K, d) 的张量，对应分类器最后一层的权重
    margin       : 允许的最大 cos(w_i, w_j)（i≠j），> margin 的部分会被惩罚

    返回
    ----
    标量 loss
    """
    # 归一化到单位球面
    w = F.normalize(proxy_weight, dim=1)         # (K, d)
    # 余弦相似度矩阵
    sim = torch.matmul(w, w.t())                 # (K, K)

    K = sim.size(0)
    eye = torch.eye(K, device=sim.device, dtype=sim.dtype)
    # 去掉对角，自身与自身不算
    sim_offdiag = sim * (1 - eye)

    # 惩罚 cos > margin 的项
    penalty = F.relu(sim_offdiag - margin)
    loss = (penalty ** 2).mean()
    return loss