import torch
import torch.nn as nn
import math
import sys


def mask_correlated_samples(N):
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(N // 2):
        mask[i, N // 2 + i] = 0
        mask[N // 2 + i, i] = 0
    mask = mask.bool()
    return mask

def teacher_infoNCE(s_i, s_j, batch_size, temperature_s):
    N = 2 * batch_size
    h = torch.cat((s_i, s_j), dim=0)

    sim = torch.matmul(h, h.T) / temperature_s
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N)
    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = criterion(logits, labels)
    loss /= N
    return loss

def student_infoNCE(t_i, t_j, class_num, temperature_t):
    p_i = t_i.sum(0).view(-1)
    p_i /= p_i.sum()
    ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
    p_j = t_j.sum(0).view(-1)
    p_j /= p_j.sum()
    ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
    entropy = ne_i + ne_j

    t_i = t_i.t()
    t_j = t_j.t()
    N = 2 * class_num
    q = torch.cat((t_i, t_j), dim=0)
    similarity = nn.CosineSimilarity(dim=2)
    sim = similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_t
    sim_i_j = torch.diag(sim, class_num)
    sim_j_i = torch.diag(sim, -class_num)

    positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N)
    negative_clusters = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_clusters.device).long()
    logits = torch.cat((positive_clusters, negative_clusters), dim=1)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = criterion(logits, labels)
    loss /= N
    return loss + entropy

def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j
def crossview_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):  #仅仅是一个中间数比较绝对误差。
    """Contrastive loss for maximizng the consistency"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()     #碾成一条又扩展回来？这是为啥？
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()     #碾成一行又扩展回来？这是为啥？

    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()

    return loss