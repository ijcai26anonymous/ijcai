# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
import torch.nn.functional as F


class FeatureAdapter(nn.Module):
    """Adapter network: converts SNN conv0 output to features suitable for ANN input"""
    def __init__(self, in_channels, out_channels):
        super(FeatureAdapter, self).__init__()
        mid_channels = in_channels  # Middle layer channels
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.adapter(x)



class ReadOut(nn.Module):
    def __init__(self, model='avg'):
        super(ReadOut, self).__init__()

    def forward(self, spike):
        if self.step_mode == 's':
            return spike
        else:
            output = spike.reshape(self.time_step, -1, spike.shape[1])
            avg_fr = output.mean(dim=0)
            return avg_fr


class LIFLayer(neuron.LIFNode):

    def __init__(self, **cell_args):
        super(LIFLayer, self).__init__()
        tau = 1.0 / (1.0 - torch.sigmoid(cell_args['decay'])).item()
        super().__init__(tau=tau, decay_input=False, v_threshold=cell_args['thresh'], v_reset=cell_args['v_reset'],
                         detach_reset=cell_args['detach_reset'], step_mode='s')
        self.register_memory('elig', 0.)
        self.register_memory('elig_factor', 1.0)
        self.register_memory('out_spikes_mean', 0.)
        # self.register_memory('curr_time_step', 0)

    @staticmethod
    # @torch.jit.script
    def calcu_sg_and_elig(current_t: int, v: torch.Tensor, elig: torch.Tensor, elig_factor: float, v_threshold: float,
                          sigmoid_alpha: float = 4.0):
        sgax = ((v - v_threshold) * sigmoid_alpha).sigmoid()
        sg = (1. - sgax) * sgax * sigmoid_alpha
        elig = 1. / (current_t + 1) * (current_t * elig + elig_factor * sg)
        return sg, elig

    def calcu_elig_factor(self, elig_factor, lam, sg, spike):
        if self.v_reset is not None:  # hard-reset
            elig_factor = self.calcu_elig_factor_hard_reset(elig_factor, lam, spike, self.v, sg)
        else:  # soft-reset
            if not self.detach_reset:  # soft-reset w/ reset_detach==False
                elig_factor = self.calcu_elig_factor_soft_reset_not_detach_reset(elig_factor, lam, sg)
            else:  # soft-reset w/ reset_detach==True
                elig_factor = self.calcu_elig_factor_soft_reset_detach_reset(elig_factor, lam)
        return elig_factor

    @staticmethod
    # @torch.jit.script
    def calcu_elig_factor_hard_reset(elig_factor: torch.Tensor, lam: float, spike: torch.Tensor, v: torch.Tensor,
                                     sg: torch.Tensor):
        elig_factor = 1. + elig_factor * (lam * (1. - spike) - lam * v * sg)
        return elig_factor

    @staticmethod
    # @torch.jit.script
    def calcu_elig_factor_soft_reset_not_detach_reset(elig_factor: torch.Tensor, lam: float, sg: torch.Tensor):
        elig_factor = 1. + elig_factor * (lam - lam * sg)
        return elig_factor

    @staticmethod
    # @torch.jit.script
    def calcu_elig_factor_soft_reset_detach_reset(elig_factor: float, lam: float):
        elig_factor = 1. + elig_factor * lam
        return elig_factor

    def elig_init(self, x: torch.Tensor):
        self.elig = torch.zeros_like(x.data)
        self.elig_factor = 1.0

    def reset_state(self):
        self.reset()
        self.curr_time_step = 0

    def forward(self, x, **kwargs):
        if self.step_mode == 's':
            self.v_float_to_tensor(x)
            self.neuronal_charge(x)
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            return spike
        else:
            assert len(x.shape) in (2, 4)
            # Support arbitrary T by determining it from the input shape
            # We assume step_mode='m' and input is [T*B, ...]
            # To handle this robustly, we use self.time_step as a hint or just follow the input
            if hasattr(self, 'time_step') and self.time_step > 0:
                T = self.time_step
                B = x.shape[0] // T
            else:
                # Default to treat whole input as one time step if not set
                T = 1
                B = x.shape[0]
            
            x = x.view(T, B, *x.shape[1:])

            self.reset()
            # self.v = torch.zeros_like(x[0])
            spikes = []
            for t in range(self.time_step):
                self.v_float_to_tensor(x[t])
                self.neuronal_charge(x[t])
                spike = self.neuronal_fire()
                spikes.append(spike)
                self.neuronal_reset(spike)

            # out = torch.stack(spikes, dim=0) if not self.train_mode_multi else torch.cat(spikes, dim=0)
            out = torch.cat(spikes, dim=0)

            return out


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2

    return loss_kd


def direction_loss(incorrect_avg, correct_avg):
    """
    Make the direction of logits in incorrect time steps close to that of correct time steps.
    Focus on direction, not magnitude or specific distribution.
    
    Args:
        incorrect_avg: [B, C] average logits of incorrect time steps
        correct_avg:   [B, C] average logits of correct time steps
    Returns:
        loss: mean of (1 - cosine_similarity)
    """
    cos_sim = F.cosine_similarity(incorrect_avg, correct_avg, dim=1)
    return (1 - cos_sim).mean()


def binary_entropy_loss(output, labels, margin=0.05):
    """
    Calculate binary entropy between correct class and predicted incorrect class for incorrect time steps.
    Target: make binary entropy of incorrect time steps as close to maximum as possible (ln(2) approx 0.693).
    Use hinge loss to ensure entropy reaches a certain threshold.
    
    Based on formula:
        diff = z_pos - z_neg
        p_pos = sigmoid(diff) = 1/(1+e^(-diff))
        H(diff) = -p_pos log p_pos - p_neg log p_neg
    
    Target: H(diff) -> ln(2) approx 0.693 (maximum binary entropy)
    
    Args:
        output: [T, B, C] logits per time step
        labels: [B] ground truth labels
        margin: entropy target threshold (default 0.05 means target entropy >= ln(2) - 0.05)
    Returns:
        loss: max(0, target_entropy - actual_entropy)
    """
    T, B, C = output.shape
    
    # Binary entropy max value
    max_binary_entropy = torch.log(torch.tensor(2.0, device=output.device))  # ln(2) approx 0.693
    target_entropy = max_binary_entropy - margin  # target entropy threshold
    
    # Get predictions
    predictions = output.argmax(dim=2)  # [T, B]
    incorrect_mask = (predictions != labels.unsqueeze(0))  # [T, B] focus on incorrect time steps
    
    # Return zero loss if no incorrect time steps
    if not incorrect_mask.any():
        return torch.tensor(0.0, device=output.device)
    
    # Collect binary entropy for all incorrect time steps
    entropies = []
    
    for t in range(T):
        # Find samples with incorrect predictions at this time step
        incorrect_t = incorrect_mask[t]  # [B]
        if not incorrect_t.any():
            continue
        
        pred_t = predictions[t, incorrect_t]  # [N] N = incorrect_t.sum()
        logits_t = output[t, incorrect_t]  # [N, C]
        labels_t = labels[incorrect_t]  # [N]
        
        # Get logits for correct class and predicted incorrect class
        batch_idx = torch.arange(incorrect_t.sum(), device=output.device)
        z_pos = logits_t[batch_idx, labels_t]  # [N] logit of correct class
        z_neg = logits_t[batch_idx, pred_t]  # [N] logit of predicted incorrect class
        
        # Calculate diff = z_pos - z_neg
        delta = z_pos - z_neg  # [N]
        
        # Calculate binary probability p_pos = sigmoid(delta)
        p_pos = torch.sigmoid(delta)  # [N]
        p_neg = 1 - p_pos  # [N]
        
        # Calculate binary entropy H(delta)
        entropy = -(p_pos * torch.log(p_pos + 1e-8) + p_neg * torch.log(p_neg + 1e-8))  # [N]
        entropies.append(entropy)
    
    if len(entropies) == 0:
        return torch.tensor(0.0, device=output.device)
    
    # Merge entropy for all incorrect time steps
    all_entropies = torch.cat(entropies)  # [total_incorrect]
    
    # Hinge loss: produce loss when entropy is less than target threshold
    # loss = max(0, target_entropy - entropy)
    loss = F.relu(target_entropy - all_entropies).mean()
    
    return loss


def soft_ce_loss(avg_fr, correct_avg):
    """
    Use soft CE to make avg_fr approach the distribution of correct_avg
    
    Args:
        avg_fr:      [B, C] average logits of all time steps
        correct_avg: [B, C] average logits of correct time steps
    Returns:
        loss: soft cross entropy
    """
    soft_target = F.softmax(correct_avg.detach(), dim=1)
    log_pred = F.log_softmax(avg_fr, dim=1)
    loss = -(soft_target * log_pred).sum(dim=1).mean()
    return loss


class LearnableMargin(nn.Module):
    """Learnable margin parameter"""
    def __init__(self, init_value=0.2):
        super().__init__()
        self.margin = nn.Parameter(torch.tensor(init_value))
    
    def forward(self):
        # Use sigmoid to limit in (0, 1) range
        return torch.sigmoid(self.margin)


def kl_time_loss(output, labels):
    """
    Calculate KL divergence between average of correct time steps and average of incorrect time steps.
    
    Args:
        output: [T, B, C] logits per time step
        labels: [B] labels
    Returns:
        loss: KL divergence
    """
    T, B, C = output.shape
    
    # Get predictions
    predictions = output.argmax(dim=2)  # [T, B]
    correct_mask = (predictions == labels.unsqueeze(0))  # [T, B]
    incorrect_mask = ~correct_mask  # [T, B]
    
    # Calculate average for correct and incorrect time steps
    epsilon = 1e-8
    
    # Average of correct time steps
    correct_count = correct_mask.sum(dim=0, keepdim=True).unsqueeze(2) + epsilon  # [1, B, 1]
    correct_avg = (output * correct_mask.unsqueeze(2)).sum(dim=0) / correct_count.squeeze(0)  # [B, C]
    
    # Average of incorrect time steps
    incorrect_count = incorrect_mask.sum(dim=0, keepdim=True).unsqueeze(2) + epsilon
    incorrect_avg = (output * incorrect_mask.unsqueeze(2)).sum(dim=0) / incorrect_count.squeeze(0)  # [B, C]
    
    # Only calculate for samples that have both correct and incorrect steps
    has_correct = correct_mask.sum(0) > 0  # [B]
    has_incorrect = incorrect_mask.sum(0) > 0  # [B]
    has_both = has_correct & has_incorrect
    
    if not has_both.any():
        return (output * 0.0).sum()
    
    # KL divergence: KL(correct || incorrect)
    # Make distribution of incorrect time steps close to that of correct time steps
    correct_prob = F.softmax(correct_avg[has_both], dim=1)
    log_incorrect_prob = F.log_softmax(incorrect_avg[has_both], dim=1)
    
    kl_loss = F.kl_div(log_incorrect_prob, correct_prob, reduction='batchmean')
    
    return kl_loss


def infonce_time_loss(output, avg_fr, correct_mask, margin, tau=0.07):
    """
    InfoNCE loss: make correct time steps close to average, and incorrect time steps away from average
    
    Args:
        output: [T, B, C] logits per time step
        avg_fr: [B, C] average logits of all time steps
        correct_mask: [T, B] mask of correct time steps
        margin: learnable margin added to positive sample similarity
        tau: temperature coefficient
    Returns:
        loss: InfoNCE loss
    """
    T, B, C = output.shape
    
    # Calculate cosine similarity between each time step and average
    sims = []
    for t in range(T):
        sim = F.cosine_similarity(output[t], avg_fr, dim=1)  # [B]
        sims.append(sim)
    sims = torch.stack(sims, dim=0)  # [T, B]
    
    # Only calculate for samples that have both correct and incorrect steps
    incorrect_mask = ~correct_mask
    has_correct = correct_mask.sum(0) > 0  # [B]
    has_incorrect = incorrect_mask.sum(0) > 0  # [B]
    has_both = has_correct & has_incorrect  # [B]
    
    if not has_both.any():
        # Return zero loss associated with output to maintain gradient flow
        return (output * 0.0).sum()
    
    # Filter valid samples
    sims = sims[:, has_both]  # [T, B']
    correct_mask = correct_mask[:, has_both]  # [T, B']
    incorrect_mask = incorrect_mask[:, has_both]  # [T, B']
    
    B_valid = sims.size(1)
    losses = []
    
    for b in range(B_valid):
        sim_b = sims[:, b]  # [T]
        correct_b = correct_mask[:, b]  # [T]
        incorrect_b = incorrect_mask[:, b]  # [T]
        
        # Positive samples: average similarity of correct steps
        sim_pos = sim_b[correct_b].mean() + margin  # Add margin to enhance positive samples
        
        # Negative samples: similarity of incorrect steps
        sim_neg = sim_b[incorrect_b]  # [num_neg]
        
        # InfoNCE: -log(exp(sim_pos/tau) / (exp(sim_pos/tau) + sum(exp(sim_neg/tau))))
        logits = torch.cat([sim_pos.unsqueeze(0), sim_neg]) / tau  # [1 + num_neg]
        labels = torch.tensor(0, device=output.device)  # positive sample at index 0
        loss_b = F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))
        losses.append(loss_b)
    
    loss = torch.stack(losses).mean()
    return loss


def align_teacher_student_logits(logits_student, logits_teacher, target, method='ela'):
    """
    Error-aware Logits Alignment (ELA)
    
    Args:
        logits_student: [B, C] student logits
        logits_teacher: [B, C] teacher logits
        target: [B] ground truth labels
        method: alignment method (only 'ela' supported)
    
    Returns:
        s, t: aligned logits
    """
    s = logits_student.clone()
    t = logits_teacher.clone()

    # ELA: Check if Student predicted incorrectly
    pred_s = s.argmax(dim=1)        # [B]
    wrong_mask = (pred_s != target) # [B]
    
    if not wrong_mask.any():
        return s, t
    
    idx = wrong_mask.nonzero(as_tuple=True)[0]
    wrong_cls = pred_s[idx]     # incorrect class predicted by student
    gt_cls = target[idx]        # correct class
    
    # align Student + shrink Teacher
    # Student: incorrect class logit = correct class logit
    s_correct_logits = s[idx, gt_cls]
    s[idx, wrong_cls] = s_correct_logits
    
    # Teacher: both logits shrink to minimum
    t_y = t[idx, gt_cls]
    t_w = t[idx, wrong_cls]
    min_logit = torch.min(t_y, t_w)
    t[idx, gt_cls] = min_logit
    t[idx, wrong_cls] = min_logit

    return s, t


def cal_loss(outputs, labels, criterion):
    T = outputs.size(0)
    Loss_es = 0
    Loss_mmd = 0
    for t in range(T):
        Loss_es += criterion(outputs[t, :, ...], labels)
    Loss_es = Loss_es / T
    return Loss_es


def make_teacher_student(avg_fr, labels):
    """
    Calculate average of correct time steps and incorrect time steps.
    
    Args:
        avg_fr: SNN output [T, B, C]
        labels: Ground truth labels [B]
    
    Returns:
        correct_avg: Average output of correct time steps [B, C]
        incorrect_avg: Average output of incorrect time steps [B, C]
        correct_mask: Mask of correct time steps [T, B]
        has_incorrect: Flags indicating which samples have incorrect time steps [B]
    """
    predictions = avg_fr.argmax(dim=2)  # [T, B]

    correct_mask = (predictions == labels.unsqueeze(0))  # [T, B]
    incorrect_mask = ~correct_mask  # [T, B]

    epsilon = 1e-8

    # Average of correct time steps
    correct_avg_fr = avg_fr * correct_mask.unsqueeze(2)  # [T, B, C]
    correct_count = correct_mask.sum(dim=0).unsqueeze(1)  # [B, 1]
    correct_avg = correct_avg_fr.sum(dim=0) / (correct_count + epsilon)  # [B, C]

    # Average of incorrect time steps
    incorrect_avg_fr = avg_fr * incorrect_mask.unsqueeze(2)  # [T, B, C]
    incorrect_count = incorrect_mask.sum(dim=0).unsqueeze(1)  # [B, 1]
    incorrect_avg = incorrect_avg_fr.sum(dim=0) / (incorrect_count + epsilon)  # [B, C]

    # Flags indicating which samples have incorrect time steps
    has_incorrect = incorrect_count.squeeze(1) > 0  # [B]

    return correct_avg, incorrect_avg, correct_mask, has_incorrect


def contrastive_avg_loss(logits, labels, tau=0.1):
    """
    Make average logits closer to the center of correct time step logits, 
    and away from the center of incorrect time step logits.
    
    Args:
        logits: [T, B, C] logits per time step
        labels: [B] Ground truth labels
        tau:    Temperature coefficient
    Returns:
        loss3: contrastive loss
    """
    T, B, C = logits.shape
    predictions = logits.argmax(dim=2)  # [T, B]
    
    correct_mask = (predictions == labels.unsqueeze(0))  # [T, B]
    incorrect_mask = ~correct_mask
    
    z_avg = logits.mean(dim=0)  # average across all time steps
    
    # Correct time step center
    correct_count = correct_mask.sum(dim=0, keepdim=True).unsqueeze(2) + 1e-8  # [1, B, 1]
    z_pos = (logits * correct_mask.unsqueeze(2)).sum(dim=0) / correct_count.squeeze(0)  # [B, C]
    
    # Incorrect time step center
    incorrect_count = incorrect_mask.sum(dim=0, keepdim=True).unsqueeze(2) + 1e-8
    z_err = (logits * incorrect_mask.unsqueeze(2)).sum(dim=0) / incorrect_count.squeeze(0)  # [B, C]
    
    # Only calculate for samples that have both correct and incorrect steps
    has_both = (correct_mask.sum(0) > 0) & (incorrect_mask.sum(0) > 0)  # [B]
    
    if not has_both.any():
        return torch.tensor(0.0, device=logits.device)
    
    z_avg = z_avg[has_both]
    z_pos = z_pos[has_both]  # do not detach, gradient flows through z_pos
    z_err = z_err[has_both]  # do not detach, gradient flows through z_err
    
    # Cosine similarity
    s_pos = F.cosine_similarity(z_avg, z_pos, dim=1) / tau  # [B']
    s_err = F.cosine_similarity(z_avg, z_err, dim=1) / tau  # [B']
    
    # InfoNCE: -log(exp(s_pos) / (exp(s_pos) + exp(s_err)))
    loss = -F.log_softmax(torch.stack([s_pos, s_err], dim=1), dim=1)[:, 0].mean()
    
    return loss


def time_step_kd_loss(output, temperature=3, method='sta'):
    """
    Each time step learns from other time steps (Similarity-aware Temporal Alignment - STA)
    """
    T, B, C = output.shape
    if T <= 1:
        return torch.tensor(0.0, device=output.device)

    # STA (Similarity-aware Temporal Alignment)
    # 1. Confidence
    probs = F.softmax(output, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(C, dtype=output.dtype, device=output.device))
    confidence = (max_entropy - entropy) / max_entropy
    
    # 2. Similarity
    logits_norm = F.normalize(output, p=2, dim=-1)
    logits_norm_t = logits_norm.transpose(0, 1)
    cos_sim_matrix = torch.bmm(logits_norm_t, logits_norm_t.transpose(1, 2))
    cos_sim_matrix = cos_sim_matrix.transpose(0, 1).transpose(1, 2) # [T, T, B]
    
    # Use full similarity range
    sim_metric = cos_sim_matrix
        
    # 3. Weight scores: confidence x similarity
    weight_scores = confidence.unsqueeze(0) * sim_metric
        
    # 4. Mask & Softmax
    mask = ~torch.eye(T, dtype=torch.bool, device=output.device).unsqueeze(-1)
    weight_scores = weight_scores.masked_fill(~mask, float('-inf'))
    weight_matrix = F.softmax(weight_scores, dim=1)
    
    # 5. Weighting
    weight_matrix_bt = weight_matrix.permute(2, 0, 1)
    logits_bt = output.transpose(0, 1)
    weighted_logits_bt = torch.bmm(weight_matrix_bt, logits_bt)
    weighted_logits = weighted_logits_bt.transpose(0, 1)

    # --- Compute vectorized KD Loss ---
    s_logits = output.reshape(-1, C)
    t_logits = weighted_logits.reshape(-1, C).detach()
    
    log_pred_s = F.log_softmax(s_logits / temperature, dim=1)
    pred_t = F.softmax(t_logits / temperature, dim=1)
    
    loss_time = F.kl_div(log_pred_s, pred_t, reduction='batchmean') * (temperature ** 2)
    
    return loss_time

