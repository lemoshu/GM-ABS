"""
[TMI Submission] GM-ABS for 3D LA segmentation

python train_final_GMABS_LA_public.py --labeled_num 4 --budget 16 --active_type uncerper_div --gpu 0 --label_strategy majority --exp LA_GMABS_HERD_release --add_point 2
"""

import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.la_sparse_SAM import (LA_heart_sparse_SAM, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses_sparse, metrics, ramps
from val_3D import test_all_case, test_all_case_AL
from val_3D_mobileSAM import test_all_case_AL_SAM_ProtoPredict, test_all_case_AL_SAM_ProtoPredict_iterative
import torchio as tio

# AL tools
import cleanlab
from sklearn.cluster import KMeans

# SAM
import cv2
from PIL import Image
# from huggingface_hub import hf_hub_download
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/LA_cross_labeling_h5', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA_GMABS_mobilesam', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet_AL', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--lr_decay', type=int, default=1,
                    help='if using learning rate decay or not?')
parser.add_argument('--patch_size', type=list,  default=[112, 112, 80], 
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu id')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=4,
                    help='labeled data')
parser.add_argument('--total_sample', type=int, default=80,
                    help='total samples')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
# Semi-supervised learning
parser.add_argument('--semi_type', type=str,
                    default='MT', help='semi_type')

# Active Learning
parser.add_argument('--active_type', type=str,
                    default='uncerper_div', help='active_type')
parser.add_argument('--topk', type=int, default=2,
                    help='top K cases to select')
parser.add_argument('--budget', type=int, default=20,
                    help='budget for active learning')
parser.add_argument('--each_round_interval', type=int, default=200,
                    help='each round interval for active learning')
parser.add_argument('--warmup_iter', type=int, default=600,
                    help='warmup iterations for active learning')
parser.add_argument('--load_pretrain_model', type=int, default=0,
                    help='load pretrain model or not?')
parser.add_argument('--pretrain_path', type=str,
                    default='', help='pretrain model path')

# SAM
parser.add_argument('--sam_model', type=str,
                    default='vit_t', help='Example model type: vit_b, vit_h, Mobilesam: vit_t')
parser.add_argument('--sam_checkpoint_path', type=str,
                    default='./sam_weights/mobile_sam.pt', help='sam_checkpoint_path') # ./sam_weights/mobile_sam.pt, ./sam_vit_b_01ec64
parser.add_argument('--each_samupdate_interval', type=int, default=600,
                    help='each update interval for SAM')
parser.add_argument('--label_strategy', type=str,
                    default='majority', help='label_strategy') # 'all', 'majority', 'soft'
parser.add_argument('--bbox_strategy', type=str,
                    default='2D', help='bbox_strategy') # '3D', '2D' [3D performs worse]
parser.add_argument('--SAM_mode', type=str,
                    default='3view', help='SAM_mode')
parser.add_argument('--logitclip', type=int, default=1,
                    help='logitclip for robust noisy label learning')
parser.add_argument('--add_point', type=int, default=2,
                    help='add point for SAM') # 0 (None), 1 (random), 2 (prototype similarity)
parser.add_argument('--sam_stage', type=int, default=1,
                    help='SAM stage') # 1 (single stage), 2 (two-stage)

args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

# AL Tools
def entropy_map_AL(p, C=2):
    # p C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=0) / \
        torch.tensor(np.log(C)).cuda()
    return y1

def diversity_kmeans_sampling(features, n_clusters=2):
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(cluster_centers, features) # function: find the closest point to the cluster center
    return closest


def sharpening(P):
    T = 1/0.1
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

# Aug
def adjust_contrast_3d_image(image):
    random_contrast = tio.transforms.RandomGamma(log_gamma=(-0.3, 0.3))
    for i in range(image.shape[0]):
        image[i] = random_contrast(image[i])
    return image

def adjust_biasfield_3d_image(image):
    device = image.device
    # to np array
    image = image.cpu().numpy()
    random_biasfield = tio.transforms.RandomBiasField(coefficients=(0.2, 0.4))
    for i in range(image.shape[0]):
        single_image = image[i, ...]
        image[i] = random_biasfield(single_image)
    return torch.from_numpy(image).to(device)


def MTCL_label_denoise(pred_prob_tensor, noisy_label_tensor, class_num=2, patch_size=[112, 112, 80]):
    # Step 1: convert the noisy label and the third-party prediction to npy
    pred_prob_np = pred_prob_tensor.cpu().detach().numpy()
    noisy_label_np = noisy_label_tensor.cpu().detach().numpy()
    # Step 2: identify the noise map
    pred_prob_np_accumulated_0 = np.swapaxes(pred_prob_np, 1, 2)
    pred_prob_np_accumulated_1 = np.swapaxes(pred_prob_np_accumulated_0, 2, 3)
    pred_prob_np_accumulated_2 = np.swapaxes(pred_prob_np_accumulated_1, 3, 4)
    pred_prob_np_accumulated_3 = pred_prob_np_accumulated_2.reshape(-1, class_num)
    pred_prob_np_accumulated = np.ascontiguousarray(pred_prob_np_accumulated_3)
    noisy_label_np_accumulated = noisy_label_np.reshape(-1).astype(np.uint8)
    assert noisy_label_np_accumulated.shape[0] == pred_prob_np_accumulated.shape[0]
    noise = cleanlab.filter.find_label_issues(noisy_label_np_accumulated, pred_prob_np_accumulated, filter_by='both', n_jobs=1)
    noise_map = noise.reshape(-1, patch_size[0], patch_size[1], patch_size[2]).astype(np.uint8)
    corrected_sam_label_np = noisy_label_np + noise_map * np.power(-1, noisy_label_np)
    corrected_sam_label_tensor = torch.from_numpy(corrected_sam_label_np).cuda(pred_prob_tensor.device)
    print('successfully denoising')
    noise_map_tensor = torch.from_numpy(noise_map).cuda(pred_prob_tensor.device)
    return corrected_sam_label_tensor, noise_map_tensor


def LogitClipping(logits, temp=1.0):
    delta = 1/temp
    # Compute the L2 norm of the logits for each instance, add a small epsilon to avoid division by zero
    norms = torch.norm(logits, 2, dim=1, keepdim=True) + 1e-7
    # Scale the logits by their norms, and then by delta
    logits_norm = torch.div(logits, norms) * delta
    # Create a mask where each element is True if the norm of the logits is greater than the temperature
    clip = (norms > temp).expand_as(logits)
    # Apply the mask: use scaled logits where the norm exceeds the temperature, original logits otherwise
    logits_final = torch.where(clip, logits_norm, logits)
    return logits_final


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    patch_size = args.patch_size
    num_classes = 2

    def create_model(ema=False):
        # Network definition
        net = net_factory_3d(net_type=args.model,
                             in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    if args.load_pretrain_model == 1:
        # only load the pretrained encoder
        pretrained_dict = torch.load(args.pretrain_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'encoder' in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('load pretrained encoder')

    db_train = LA_heart_sparse_SAM(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.total_sample))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, 
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses_sparse.DiceLoss(2)
    soft_dice_loss = losses_sparse.softDiceLoss(num_classes)
    noiserobustdice_loss = losses_sparse.NoiseRobustDiceLoss(1.5, 2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    selected_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            start_time_1 = time.time()
            volume_batch, label_batch, cross_label, indicator, sam_label = sampled_batch['image'], sampled_batch['label'], sampled_batch['cross_label'], sampled_batch['indicator'], sampled_batch['pseudo_label']
            volume_batch, label_batch, cross_label, indicator, sam_label = volume_batch.cuda(), label_batch.cuda(), cross_label.cuda(), indicator.cuda(), sam_label.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            ema_inputs_2 = adjust_contrast_3d_image(unlabeled_volume_batch)

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            fts = model.featuremap

            with torch.no_grad():
                ema_output_all_clean = ema_model(volume_batch)
                ema_output = ema_model(ema_inputs)
                ema_output_2 = ema_model(ema_inputs_2)

            # Supervised loss for labeled slices
            loss_ce = losses_sparse.CE_Mask_loss(outputs[:args.labeled_bs], cross_label[:args.labeled_bs][:], indicator[:args.labeled_bs][:], 
                                                 batch_size=args.labeled_bs, H=patch_size[0], W=patch_size[1], D=patch_size[2])
            # loss_ce = losses_sparse.CE_Mask_loss_multiclass(outputs[:args.labeled_bs], cross_label[:args.labeled_bs][:], indicator[:args.labeled_bs][:])
            supervised_loss = loss_ce


################################## SAM-based loss ##################################
            if sam_label.sum() == 0:
                loss_sam = 0
            else:
                # Noise-robust dice loss
                if args.label_strategy == 'all' or args.label_strategy == 'majority':
                    sam_label_corrected, noise_map_tensor = MTCL_label_denoise(outputs_soft, sam_label, class_num=num_classes, patch_size=patch_size)
                    if args.logitclip == 1:
                        outputs_clip = LogitClipping(outputs)
                        print('outputs_clip shape: ', outputs_clip.shape)
                        print('sam_label_corrected shape: ', sam_label_corrected.shape)
                        loss_sam = ce_loss(outputs_clip, sam_label_corrected)
                    else:              
                        loss_ce_sam = ce_loss(outputs, sam_label_corrected)
                        loss_sam = loss_ce_sam

################################### Semi-supervised loss ###################################
            # MAE-based consistency loss
            if args.consistency_type == 'mae':
                consistency_dist = torch.mean(torch.abs(outputs[args.labeled_bs:] - ema_output))  
                consistency_dist_2 = torch.mean(torch.abs(outputs[args.labeled_bs:] - ema_output_2))  
                consistency_loss = 0.5*consistency_dist + 0.5*consistency_dist_2

            # MSE-based consistency loss
            elif args.consistency_type == 'mse':
                consistency_dist = losses_sparse.softmax_mse_loss(outputs[args.labeled_bs:], ema_output)  
                consistency_dist_2 = losses_sparse.softmax_mse_loss(outputs[args.labeled_bs:], ema_output_2) 
                consistency_loss = 0.5*(torch.mean(consistency_dist) + torch.mean(consistency_dist_2))

            consistency_weight = get_current_consistency_weight(iter_num//150)

            # Total loss
            # sam-based consistency loss weight should be decreased as training goes
            w_sam = 1 - 0.9*ramps.sigmoid_rampup(iter_num,max_iterations)
            loss = supervised_loss + 0.5 * w_sam * loss_sam + consistency_weight * (consistency_loss) 


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            end_time_1 = time.time()
            elapsed_time_iter = end_time_1 - start_time_1
            print('Elapsed time for iteration {}: {:.2f} seconds'.format(iter_num, elapsed_time_iter))
                        
            if args.lr_decay == 1:
                lr_ = base_lr
                if iter_num % 2500 == 0 and iter_num >= 5000:
                    lr_ = base_lr * 0.1 ** (iter_num // 2500)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            else:
                lr_ = base_lr

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice_labeled_slice, iter_num)
            writer.add_scalar('info/loss_dice_sam', loss_sam, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

################################## SAM Pseudo Label Update ##################################
            if iter_num >= args.warmup_iter and iter_num % args.each_samupdate_interval == 0 and iter_num < 0.8*args.max_iterations:
                print('Adding and Updating SAM-generated pseudo labels in the dataset')
                ema_model.eval()
                if args.SAM_mode == '3view':
                    print('single stage SAM pseudo label generation')
                    prob_train_list, name_list, label_pred_list, sam_label_list = test_all_case_AL_SAM_ProtoPredict(
                        ema_model, args.root_path, test_list="train.txt", num_classes=2, patch_size=args.patch_size,
                        stride_xy=64, stride_z=64, model_type=args.sam_model, checkpoint_path=args.sam_checkpoint_path, 
                        strategy=args.label_strategy, add_point=args.add_point)
                    db_train.update_pseudo_labels(name_list, sam_label_list)
                    print('UPDATE THE SAM PSEUDO LABELS IN THE DATASET (3V)')
                    # clean to save memory
                    del prob_train_list, name_list, label_pred_list, sam_label_list
                    ema_model.train()
                else:
                    break

################################# Active Learning: sample selection #################################
            each_round_interval = args.each_round_interval
            warmup_iter = args.warmup_iter        
            if iter_num % each_round_interval == 0 and iter_num > warmup_iter and selected_num < args.budget:
                print('Active Learning process - iter_num: ', iter_num)
                selected_num += args.topk                
                print('we have selected {} volumes'.format(selected_num))
                model.eval()                            
                prob_train_list, name_list, label_pred_list, image_np_list = test_all_case_AL(
                    model, args.root_path, test_list="train.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64)
                
                if args.active_type == 'random':
                    print('random sampling')
                    # select K random cases
                    idx = random.sample(unlabeled_idxs, args.topk)
                    # check if the selected cases are from unlabeled_idxs; otherwise, select another one
                    for i in range(len(idx)):
                        while idx[i] not in unlabeled_idxs:
                            idx[i] = random.randint(0, len(prob_train_list)-1)
                    print('random select case: ', idx)                 
                    # save the cases to txt
                    with open(os.path.join(snapshot_path, 'selection.txt'), 'a') as f:
                        for i in idx:
                            f.write(name_list[i]+'\n')                  

                elif args.active_type == 'uncertainty_percentage':
                    print('uncertainty percentage sampling')
                    # select a case with highest percentage of high uncertainty (using entropy, >0.5)
                    uncertainty_per_list = []
                    for i in range(len(prob_train_list)):
                        EM_map = entropy_map_AL(torch.tensor(prob_train_list[i]).cuda(), C=num_classes)
                        # percentage of voxels with high uncertainty (>thre)
                        threshold = 0.35 + 0.25*ramps.sigmoid_rampup(iter_num,max_iterations)
                        percent_EM = torch.sum(EM_map>threshold) / torch.tensor(EM_map.shape).prod()
                        uncertainty_per_list.append(percent_EM)
                    print('current uncertainty threshold: ', threshold)
                    uncertainty_per_list = torch.tensor(uncertainty_per_list)
                    # select the top K cases with highest percentage of high uncertainty
                    idx = torch.topk(uncertainty_per_list, args.topk).indices
                    # see if these cases are from unlabeled_idxs; otherwise, select another one
                    for i in range(len(idx)):
                        while idx[i] not in unlabeled_idxs:
                            uncertainty_per_list[int(idx[i])] = 0
                            idx[i] = torch.topk(uncertainty_per_list, args.topk).indices[i]
                    # print('uncertainty percent select case: ', idx, name_list[idx])
                    # save the cases to txt
                    with open(os.path.join(snapshot_path, 'selection.txt'), 'a') as f:
                        for i in idx:
                            f.write(name_list[i]+'\n')

                elif args.active_type == 'uncerper_div':
                    def calculate_histogram_features(image_tensor, num_bins=20):
                        # Calculate a normalized histogram of intensities for the image tensor
                        hist, _ = np.histogram(image_tensor.cpu().numpy(), bins=num_bins, range=(image_tensor.min().item(), image_tensor.max().item()))
                        hist = hist / hist.sum()  # Normalize the histogram
                        return hist
                    print('Uncertainty percentage sampling with histogram-based diversity')
                    # select a case with highest percentage of high uncertainty (using entropy, >0.5)
                    uncertainty_per_list = []
                    histogram_features_list = []
                    for i in range(len(prob_train_list)):
                        EM_map = entropy_map_AL(torch.tensor(prob_train_list[i]).cuda(), C=num_classes)
                        # percentage of voxels with high uncertainty (>thre)
                        threshold = 0.35 + 0.25*ramps.sigmoid_rampup(iter_num,max_iterations)
                        percent_EM = torch.sum(EM_map>threshold) / torch.tensor(EM_map.shape).prod()
                        uncertainty_per_list.append(percent_EM)

                        # Calculate histogram-based intensity features for diversity selection
                        image_tensor = torch.tensor(image_np_list[i]).cuda()
                        histogram_features = calculate_histogram_features(image_tensor)
                        histogram_features_list.append(histogram_features)

                    print('current uncertainty threshold: ', threshold)
                    uncertainty_per_list = torch.tensor(uncertainty_per_list)
                    # Step 2: Select the top 10% uncertain samples
                    num_top_uncertain = max(1, int(0.1 * len(uncertainty_per_list)))  # Ensure at least one sample is selected
                    uncertain_idx = torch.topk(uncertainty_per_list, num_top_uncertain).indices

                    # see if these cases are from unlabeled_idxs; otherwise, select another one
                    for i in range(len(uncertain_idx)):
                        while uncertain_idx[i] not in unlabeled_idxs:
                            uncertainty_per_list[int(uncertain_idx[i])] = 0
                            uncertain_idx[i] = torch.topk(uncertainty_per_list, num_top_uncertain).indices[i]
                                        
                    # Calculate the diversity of the selected cases
                    selected_histogram_features = np.array([histogram_features_list[i] for i in uncertain_idx])
                    # Calculate the diversity of the selected cases using KMeans clustering
                    kmeans = KMeans(n_clusters=args.topk).fit(selected_histogram_features)
                    idx = []

                    # Select one representative sample per cluster for diversity
                    for cluster_idx in range(args.topk):
                        cluster_indices = np.where(kmeans.labels_ == cluster_idx)[0]
                        if len(cluster_indices) > 1:
                            idx.append(uncertain_idx[cluster_indices[0]].item())

                    # print('uncertainty percent select case: ', idx, name_list[idx])
                    # save the cases to txt
                    with open(os.path.join(snapshot_path, 'selection.txt'), 'a') as f:
                        for i in idx:
                            f.write(name_list[i]+'\n')
                            print('uncertainty percent and diversity select case: ', i, name_list[i])                
                else:
                    raise NotImplementedError('The active learning type is not implemented yet')

                # add the selected case to the labeled data; Remove the selected samples from the unlabeled_idxs
                labeled_idxs.extend(idx)
                unlabeled_idxs = [i for i in unlabeled_idxs if i not in idx]
                # check the remaining unlabeled_idxs
                print('current unlabeled_idxs: ', len(unlabeled_idxs))    

                del prob_train_list, name_list, label_pred_list            
                                          
                # update the batch sampler
                batch_sampler = TwoStreamBatchSampler(
                    labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
                trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                                            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
                model.train()


            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_sam: %f, consistency_loss: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice_labeled_slice, loss_sam, consistency_loss))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num > 400 and iter_num % 200 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="test.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=18, stride_z=4)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model_active/{}_{}_{}/{}".format(
        args.exp, args.labeled_num, args.budget, args.model)
    # remove the selection.txt
    if os.path.exists(os.path.join(snapshot_path, 'selection.txt')):
        os.remove(os.path.join(snapshot_path, 'selection.txt'))
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code_Active'):
        shutil.rmtree(snapshot_path + '/code_Active')
    shutil.copytree('.', snapshot_path + '/code_Active',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
