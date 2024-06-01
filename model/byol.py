import torch.nn as nn
import copy
import math
class BYOL(nn.Module):
    def __init__(self, feature_dim, high_feature_dim, class_num):
        super().__init__()
        self.student = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),                               #Cas数据集就必须注释掉
            nn.Linear(feature_dim, high_feature_dim)
        )
        self.cls = nn.Sequential(
            nn.Linear(high_feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.teacher = copy.deepcopy(self.student)

    def target_ema(self, k, K, base_tau=0.4):
        return 1 - (1 - base_tau) * (math.cos(math.pi * k / K) + 1) / 2

    def update_moving_average(self, global_step, max_steps):
        tau = self.target_ema(global_step, max_steps)
        for online, target in zip(self.student.parameters(), self.teacher.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data