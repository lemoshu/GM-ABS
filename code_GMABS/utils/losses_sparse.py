import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import contextlib
import math
from math import exp

def binary_dice_loss_mask(score,target,mask):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target* mask)
    y_sum = torch.sum(target * target* mask)
    z_sum = torch.sum(score * score* mask)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def CE_Mask_loss(logits, target, weight_mask, batch_size, H, W, D):
    # Calculate log probabilities
    logp = F.log_softmax(logits,dim=1)
    # Gather log probabilities with respect to target
    logp = logp.gather(1, target.view(batch_size, 1, H, W, D))
    # Multiply with weights
    weighted_logp = (logp * weight_mask).view(batch_size, -1)
    # Rescale so that loss is in approx. same interval
    #weighted_loss = weighted_logp.sum(1) / weights.view(batch_size, -1).sum(1)
    weighted_loss = (weighted_logp.sum(1) - 0.00001) / (weight_mask.view(batch_size, -1).sum(1) + 0.00001)
    # Average over mini-batch
    weighted_CE_loss = -1.0 * weighted_loss.mean()
    return weighted_CE_loss


def CE_Mask_loss_multiclass(logits, target, weight_mask=None):
    """
    Multi-class Cross-Entropy loss with pixel-wise weighting
    
    Args:
        logits: (Tensor) raw model output [N, C, H, W, D]
        target: (Tensor) ground truth indices [N, H, W, D] or [N, 1, H, W, D]
        weight_mask: (Tensor) per-pixel weights [N, H, W, D] or None
    Returns:
        weighted_CE_loss: (Tensor) scalar loss value
    """
    # Validate dimensions
    assert logits.dim() == 5, "Logits must be 5D: (N, C, H, W, D)"
    assert target.dim() in [4, 5], "Target must be 4D/5D: (N, H, W, D) or (N, 1, H, W, D)"    
    # Remove singleton channel if present
    if target.dim() == 5:
        target = target.squeeze(1)    
    # Get dimensions from tensors
    N, C, H, W, D = logits.shape
    spatial_dims = (H, W, D)   
    # Calculate log probabilities
    logp = F.log_softmax(logits, dim=1)    
    # Gather log probabilities for true classes
    logp = logp.gather(1, target.view(N, 1, *spatial_dims))  # [N, 1, H, W, D]    
    # Default to uniform weights if None
    if weight_mask is None:
        weight_mask = torch.ones(N, *spatial_dims, device=logits.device)    
    # Ensure weight mask matches dimensions
    weight_mask = weight_mask.view(N, 1, *spatial_dims)  # [N, 1, H, W, D]    
    # Weighted log probabilities
    weighted_logp = logp * weight_mask  # [N, 1, H, W, D]    
    # Flatten spatial dimensions
    weighted_logp = weighted_logp.view(N, -1)  # [N, H*W*D]
    weight_sum = weight_mask.view(N, -1).sum(1)  # [N]   
    # Stable normalization
    eps = 1e-8
    weighted_loss = weighted_logp.sum(1) / (weight_sum + eps)  # [N]   
    # Final loss
    weighted_CE_loss = -weighted_loss.mean()    
    return weighted_CE_loss


class NoiseRobustDiceLoss(nn.Module):
    def __init__(self, gamma=1.5, n_classes=2):
        super(NoiseRobustDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.gamma = gamma
    def _one_hot_encoder(self,input_tensor,n_classes):
        tensor_list = []
        for i in range(n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    def reshape_tensor_to_2D(self,x):
        tensor_dim = len(x.size())
        num_class  = list(x.size())[1]
        if(tensor_dim == 5):
            x_perm  = x.permute(0, 2, 3, 4, 1)
        elif(tensor_dim == 4):
            x_perm  = x.permute(0, 2, 3, 1)
        else:
            raise ValueError("{0:}D tensor not supported".format(tensor_dim))
        y = torch.reshape(x_perm, (-1, num_class)) 
        return y
    def forward(self, predict, soft_y):
        # soft_y = soft_y.unsqueeze(1)
        soft_y = self._one_hot_encoder(soft_y,self.n_classes)
        predict = self.reshape_tensor_to_2D(predict)
        soft_y  = self.reshape_tensor_to_2D(soft_y) 
        numerator = torch.abs(predict - soft_y)
        numerator = torch.pow(numerator, self.gamma)
        denominator = predict + soft_y 
        numer_sum = torch.sum(numerator,  dim = 0)
        denom_sum = torch.sum(denominator,  dim = 0)
        loss_vector = numer_sum / (denom_sum + 1e-5)
        loss = torch.mean(loss_vector)   
        return loss


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def soft_cross_entropy_loss(logits, soft_targets):
    """
    Calculate the cross-entropy loss given logits and soft targets.
    
    Args:
    logits (torch.Tensor): Logits from the model (before softmax).
    soft_targets (torch.Tensor): Target probabilities for each class.
    
    Returns:
    torch.Tensor: The cross-entropy loss.
    """
    log_softmax = F.log_softmax(logits, dim=1)
    return torch.mean(torch.sum(-soft_targets * log_softmax, dim=1))


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

# for CLB (MICCAI22)
class Block_DiceLoss(nn.Module):
    def __init__(self, n_classes, block_num):
        super(Block_DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.block_num = block_num
        self.dice_loss = DiceLoss(self.n_classes)
    def forward(self, inputs, target, weight=None, softmax=False):
        shape = inputs.shape
        img_size = shape[-1]
        div_num = self.block_num
        block_size = math.ceil(img_size / self.block_num)
        if target is not None:
            loss = []
            for i in range(div_num):
                for j in range(div_num):
                    block_features = inputs[:, :, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                    block_labels = target[:, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                    tmp_loss = self.dice_loss(block_features, block_labels.unsqueeze(1))
                    loss.append(tmp_loss)
            loss = torch.stack(loss).mean()
        return loss

def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


#https://github.com/lemoshu/ContrastiveSSL-LA/blob/main/code/utils/losses.py
def scc_loss(cos_sim,tau,lb_center_12_bg,lb_center_12_la, un_center_12_bg, un_center_12_la):
    loss_intra_bg = torch.exp((cos_sim(lb_center_12_bg, un_center_12_bg))/tau)
    loss_intra_la = torch.exp((cos_sim(lb_center_12_la, un_center_12_la))/tau)
    loss_inter_bg_la = torch.exp((cos_sim(lb_center_12_bg, un_center_12_la))/tau)
    loss_inter_la_bg = torch.exp((cos_sim(lb_center_12_la, un_center_12_bg))/tau)
    loss_contrast_bg = -torch.log(loss_intra_bg)+torch.log(loss_inter_bg_la)
    loss_contrast_la = -torch.log(loss_intra_la)+torch.log(loss_inter_la_bg)
    loss_contrast = torch.mean(loss_contrast_bg+loss_contrast_la)
    return loss_contrast


# MICCAI22
def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss
    
class softDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(softDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice
        return loss / self.n_classes
        
@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d):
    # pdb.set_trace()
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8  ###2-p length of vector
    return d

class VAT3d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT3d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = Binary_dice_loss
        
    def forward(self, model, x):
        with torch.no_grad():
            pred= F.softmax(model(x)[0], dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device) ### initialize a random tensor between [-0.5, 0.5]
        d = _l2_normalize(d) ### an unit vector
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)[0]
                p_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(p_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
            pred_hat = model(x + self.epi * d)[0]
            p_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(p_hat, pred)
        return lds
