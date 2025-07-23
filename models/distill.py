import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillKL(nn.Module):
    def __init__(self, initial_T):
        super(DistillKL, self).__init__()
        self.T = nn.Parameter(torch.tensor(initial_T, dtype=torch.float32))

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss

    def get_temperature(self):
        return self.T.item()

    def set_temperature(self, new_T):
        self.T.data = torch.tensor(new_T, dtype=torch.float32)

class HintLoss(nn.Module):
    def __init__(self, initial_T):
        super(HintLoss, self).__init__()
        self.T = nn.Parameter(torch.tensor(initial_T, dtype=torch.float32))
        self.crit = nn.MSELoss()

    def forward(self, fs, ft):
        return self.crit(fs, ft)

    def get_temperature(self):
        return self.T.item()

    def set_temperature(self, new_T):
        self.T.data = torch.tensor(new_T, dtype=torch.float32)

class FocalDistillKL(nn.Module):
    def __init__(self, initial_T, gamma=2):
        super(FocalDistillKL, self).__init__()
        self.T = nn.Parameter(torch.tensor(initial_T, dtype=torch.float32))
        self.gamma = gamma
    
    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        
        # 计算focal weight
        focal_weight = (1 - p_t) ** self.gamma
        loss = (focal_weight * F.kl_div(p_s, p_t, reduction='none')).sum(dim=1).mean()
        loss = loss * (self.T ** 2)
        return loss
    
    def get_temperature(self):
        return self.T.item()
    
    def set_temperature(self, new_T):
        # 确保新温度在正确的设备上
        self.T.data = torch.tensor(new_T, dtype=torch.float32).to(self.T.device)

class ContrastiveDistillKL(nn.Module):
    def __init__(self, initial_T, temperature=0.07):
        super(ContrastiveDistillKL, self).__init__()
        self.T = nn.Parameter(torch.tensor(initial_T, dtype=torch.float32))
        self.temperature = temperature
    
    def forward(self, f_s, f_t, y_s, y_t):
        # 确保输入是二维的
        if f_s.dim() == 1:
            f_s = f_s.unsqueeze(0)
        if f_t.dim() == 1:
            f_t = f_t.unsqueeze(0)
        if y_s.dim() == 1:
            y_s = y_s.unsqueeze(0)
        if y_t.dim() == 1:
            y_t = y_t.unsqueeze(0)
        
        # KL divergence loss
        p_s = F.log_softmax(y_s/self.T, dim=-1)
        p_t = F.softmax(y_t/self.T, dim=-1)
        kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        
        # Contrastive loss
        f_s = F.normalize(f_s, dim=-1)
        f_t = F.normalize(f_t, dim=-1)
        logits = torch.matmul(f_s, f_t.t()) / self.temperature
        labels = torch.arange(f_s.shape[0]).to(f_s.device)
        
        contrastive_loss = F.cross_entropy(logits, labels)
        
        # Combine losses
        total_loss = kl_loss + contrastive_loss
        return total_loss
    
    def get_temperature(self):
        return self.T.item()
    
    def set_temperature(self, new_T):
        self.T.data = torch.tensor(new_T, dtype=torch.float32)
        
class DualContrastiveDistillKL(nn.Module):
    def __init__(self, initial_T, temperature=0.07):
        super(DualContrastiveDistillKL, self).__init__()
        self.T = nn.Parameter(torch.tensor(initial_T, dtype=torch.float32))
        self.temperature = temperature
    
    def forward(self, f_s, f_t, y_s, y_t, f_s_dual, f_t_dual):
        # 确保输入是二维的
        if f_s.dim() == 1:
            f_s = f_s.unsqueeze(0)
        if f_t.dim() == 1:
            f_t = f_t.unsqueeze(0)
        if y_s.dim() == 1:
            y_s = y_s.unsqueeze(0)
        if y_t.dim() == 1:
            y_t = y_t.unsqueeze(0)
        if f_s_dual.dim() == 1:
            f_s_dual = f_s_dual.unsqueeze(0)
        if f_t_dual.dim() == 1:
            f_t_dual = f_t_dual.unsqueeze(0)

        # KL divergence loss
        p_s = F.log_softmax(y_s / self.T, dim=-1)
        p_t = F.softmax(y_t / self.T, dim=-1)
        kl_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)

        # Contrastive loss for original features
        f_s = F.normalize(f_s, dim=-1)
        f_t = F.normalize(f_t, dim=-1)
        logits = torch.matmul(f_s, f_t.t()) / self.temperature
        labels = torch.arange(f_s.shape[0]).to(f_s.device)
        contrastive_loss = F.cross_entropy(logits, labels)

        # Contrastive loss for dual features
        f_s_dual = F.normalize(f_s_dual, dim=-1)
        f_t_dual = F.normalize(f_t_dual, dim=-1)
        dual_logits = torch.matmul(f_s_dual, f_t_dual.t()) / self.temperature
        dual_labels = torch.arange(f_s_dual.shape[0]).to(f_s_dual.device)
        dual_contrastive_loss = F.cross_entropy(dual_logits, dual_labels)

        # Combine losses
        total_loss = kl_loss + contrastive_loss + dual_contrastive_loss
        return total_loss
    
    def get_temperature(self):
        return self.T.item()
    
    def set_temperature(self, new_T):
        self.T.data = torch.tensor(new_T, dtype=torch.float32)
