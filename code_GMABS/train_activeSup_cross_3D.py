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
from dataloaders.la_sparse import (LA_heart_sparse, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses_sparse, metrics, ramps
from val_3D import test_all_case, test_all_case_AL

# AL tools
import cleanlab


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/LA_cross_labeling_h5', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA_SupAL_cross_random', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
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
# Active Learning
parser.add_argument('--active_type', type=str,
                    default='random', help='active_type')
parser.add_argument('--budget', type=int, default=16,
                    help='budget for active learning')

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
    closest, _ = pairwise_distances_argmin_min(cluster_centers, features) # Find the closest point to the cluster center
    return closest

    
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

    db_train = LA_heart_sparse(base_dir=train_data_path,
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

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    selected_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch, cross_label, indicator = sampled_batch['image'], sampled_batch['label'], sampled_batch['cross_label'], sampled_batch['indicator']
            volume_batch, label_batch, cross_label, indicator = volume_batch.cuda(), label_batch.cuda(), cross_label.cuda(), indicator.cuda()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            # Supervised loss for labeled slices                                                           
            loss_ce = losses_sparse.CE_Mask_loss(outputs[:args.labeled_bs], cross_label[:args.labeled_bs][:], indicator[:args.labeled_bs][:], 
                                                 batch_size=args.labeled_bs, H=patch_size[0], W=patch_size[1], D=patch_size[2])
            supervised_loss = loss_ce

            loss = supervised_loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
                        
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
            

            # Active Learning: sample selection
            each_round_interval = 200         
            warmup_iter = 1000         
            if iter_num % each_round_interval == 0 and iter_num >= warmup_iter and iter_num <= (args.budget*each_round_interval + warmup_iter):
                print('Active Learning process - iter_num: ', iter_num)
                selected_num += 1                
                print('we have selected {} volumes'.format(selected_num))
                model.eval()
                # inference on the unlabeled data (no dropout: model.eval())                            
                prob_train_list, name_list, label_pred_list, image_np_list = test_all_case_AL(
                    model, args.root_path, test_list="train.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64)
                
                if args.active_type == 'random':
                    print('random sampling')
                    # select a random case
                    idx = random.randint(0, len(prob_train_list)-1)
                    # see if this case is from unlabeled_idxs; otherwise, select another one
                    while idx not in unlabeled_idxs:
                        idx = random.randint(0, len(prob_train_list)-1)
                    print('random select case: ', idx, name_list[idx])
                    # save the cases to txt
                    with open(os.path.join(snapshot_path, 'selection.txt'), 'a') as f:
                        f.write(name_list[idx]+'\n')

                elif args.active_type == 'least_confidence':
                    print('least confidence sampling')
                    # select a case with lowest mean confidence of 25% lowest confidence voxels
                    confidence_list = []
                    for i in range(len(prob_train_list)):
                        # caculate the max probability for each voxel
                        confidences = torch.max(torch.tensor(prob_train_list[i]).cuda(), dim=0)
                        # find the top 25% of all voxels with the lowest confidence
                        quantile_confidence_25 = torch.quantile(confidences.values, 0.25)
                        print('quantile_confidence_25: ', quantile_confidence_25)
                        # append the mean of the top 25% of all voxels with the lowest confidence
                        confidence_list.append(torch.mean(confidences.values[confidences.values<quantile_confidence_25]))
                    confidence_list = torch.tensor(confidence_list)
                    idx = torch.argmin(confidence_list).item()
                    # see if this case is from unlabeled_idxs; otherwise, select another one
                    while idx not in unlabeled_idxs:
                        confidence_list[idx] = 1
                        idx = torch.argmin(confidence_list).item()
                    print('least confidence select case: ', idx, name_list[idx])
                    # save the cases to txt
                    with open(os.path.join(snapshot_path, 'selection.txt'), 'a') as f:
                        f.write(name_list[idx]+'\n')

                elif args.active_type == 'uncertainty':
                    print('uncertainty sampling')
                    # select a case with high uncertainty (using entropy)
                    uncertainty_list = []
                    for i in range(len(prob_train_list)):
                        EM_map = entropy_map_AL(torch.tensor(prob_train_list[i]).cuda(), C=num_classes)
                        # mean entropy for all voxels
                        mean_EM = torch.mean(EM_map)
                        uncertainty_list.append(mean_EM)
                    uncertainty_list = torch.tensor(uncertainty_list)
                    idx = torch.argmax(uncertainty_list).item()
                    # see if this case is from unlabeled_idxs; otherwise, select another one
                    while idx not in unlabeled_idxs:
                        uncertainty_list[idx] = 0
                        idx = torch.argmax(uncertainty_list).item()
                    print('uncertainty select case: ', idx, name_list[idx])
                    # save the cases to txt
                    with open(os.path.join(snapshot_path, 'selection.txt'), 'a') as f:
                        f.write(name_list[idx]+'\n')

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
                    idx = torch.argmax(uncertainty_per_list).item()
                    # see if this case is from unlabeled_idxs; otherwise, select another one
                    while idx not in unlabeled_idxs:
                        uncertainty_per_list[idx] = 0
                        idx = torch.argmax(uncertainty_per_list).item()
                    print('uncertainty percent select case: ', idx, name_list[idx])
                    # save the cases to txt
                    with open(os.path.join(snapshot_path, 'selection.txt'), 'a') as f:
                        f.write(name_list[idx]+'\n')

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
                labeled_idxs.append(idx)
                unlabeled_idxs.remove(idx)

                # update the batch sampler
                batch_sampler = TwoStreamBatchSampler(
                    labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
                trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                                            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
                model.train()


            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice_labeled_slice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num > 400 and iter_num % 200 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="test.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64)
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
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
