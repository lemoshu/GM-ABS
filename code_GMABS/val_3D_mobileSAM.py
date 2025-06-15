"""
Environment: source activate mobilesam
"""

import math
from glob import glob

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm
from networks.net_factory_3d import net_factory_3d
import math

# SAM
import cv2
from PIL import Image
import os
# from huggingface_hub import hf_hub_download
# from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import copy


# SAM modules
def get_bounding_box(ground_truth_map):
    '''
    This function creates varying bounding box coordinates based on the segmentation contours as prompt for the SAM model
    The padding is random int values between 0 and 5 pixels
    '''
    box_aug = 5
    if len(np.unique(ground_truth_map)) > 1:
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)       
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, box_aug))
        x_max = min(W, x_max + np.random.randint(0, box_aug))
        y_min = max(0, y_min - np.random.randint(0, box_aug))
        y_max = min(H, y_max + np.random.randint(0, box_aug))    
        bbox = [x_min, y_min, x_max, y_max]
        bbox_ROI = True
        return bbox, bbox_ROI
    else:
        # if there is no mask in the array, set bbox to image size
        return [0, 0, ground_truth_map.shape[0], ground_truth_map.shape[1]], False


def get_points_from_mask(mask):
    '''
    This function returns coordinates of 5 random points in a mask
    '''
    # get the coordinates of the mask
    y_indices, x_indices = np.where(mask > 0)
    # get the coordinates outside the mask
    y_indices_bg, x_indices_bg = np.where(mask == 0)
    # randomly select 5 points from the mask
    if len(y_indices) > 5:
        idx = np.random.choice(len(y_indices), 5, replace=False)
        points = np.array([x_indices[idx], y_indices[idx]]).T
        labels = np.ones(5)
    else:
        points = None
        labels = None
    return points, labels


def get_points_from_mask_add_neg_topk(mask, obj_similarity_map, topk=5):
    '''
    Note: Here the mask should be the prototype-based mask
    This function returns coordinates of 2 topk (similarity) points in a mask and 1 point (lowest confidence) outside the mask
    '''
    # get the coordinates of the mask
    y_indices, x_indices = np.where(mask > 0)
    # get the coordinates outside the mask
    y_indices_bg, x_indices_bg = np.where(mask == 0)
    # randomly select 5 points from the mask
    if len(y_indices) > topk:
        # we can only select the most similar (top-K) points from prototype-based mask
        idx = np.argsort(obj_similarity_map[y_indices, x_indices])[-topk:]
        idx_bg = np.argsort(obj_similarity_map[y_indices_bg, x_indices_bg])[:1]
        # concatenate the points
        points = np.concatenate((np.array([x_indices[idx], y_indices[idx]]).T, np.array([x_indices_bg[idx_bg], y_indices_bg[idx_bg]]).T), axis=0)
        labels = np.concatenate((np.ones(topk), np.zeros(1)), axis=0)
        # points = np.array([x_indices[idx], y_indices[idx]]).T
        # labels = np.ones(5)
    else:
        points = None
        labels = None
    return points, labels


def sam_fixed_predictor(image, prediction, view=2, model_type="vit_t", checkpoint_path='./sam_weights/mobile_sam.pt'): # image shape: (H, W, D), prediction shape: (H, W, D)
    # print('image shape: ', image.shape)
    sam_prediction = np.zeros(prediction.shape)
    print('SAM model type: ', model_type)
    # model_type = "vit_b"  # Example model type: vit_b, vit_h
    # checkpoint_path = './sam_vit_b_01ec64.pth' # sam_vit_b_01ec64, sam_vit_h_4b8939
    # Load the model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    for i in range(image.shape[view]):
        if view == 0:
            img = image[i,:,:]
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img = img.astype(np.uint8)
            img_rgb = np.dstack((img, img, img))
            prediction_ = prediction[i,:,:]
            bbox_prompt, havebbox = get_bounding_box(prediction_) # list
            bbox_prompt = np.array(bbox_prompt)
            point_prompt, point_label = get_points_from_mask(prediction_)            
            # print('point_prompt: ', point_prompt)
            if havebbox:
                sam_predictor.set_image(img_rgb)
                masks, scores, logits = sam_predictor.predict(
                    point_coords=point_prompt,
                    point_labels=point_label,
                    box=bbox_prompt[None, :],
                    multimask_output=True, # If True, return many masks
                )
                best_mask = masks[np.argmax(scores), :, :]
            else:
                best_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
            sam_prediction[i,:,:] = best_mask.astype(np.uint8)
        elif view == 1:
            img = image[:,i,:]
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img = img.astype(np.uint8)
            img_rgb = np.dstack((img, img, img))
            prediction_ = prediction[:,i,:]
            bbox_prompt, havebbox = get_bounding_box(prediction_)
            bbox_prompt = np.array(bbox_prompt)
            point_prompt, point_label = get_points_from_mask(prediction_) 
            if havebbox:
                sam_predictor.set_image(img_rgb)
                masks, scores, logits = sam_predictor.predict(
                    point_coords=point_prompt,
                    point_labels=point_label,
                    box=bbox_prompt[None, :],
                    multimask_output=True, # If True, return many masks
                )
                best_mask = masks[np.argmax(scores), :, :]
            else:
                best_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
            sam_prediction[:,i,:] = best_mask.astype(np.uint8)
        elif view == 2:
            img = image[:,:,i]
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img = img.astype(np.uint8)
            img_rgb = np.dstack((img, img, img))
            prediction_ = prediction[:,:,i]
            bbox_prompt, havebbox = get_bounding_box(prediction_)
            bbox_prompt = np.array(bbox_prompt)
            point_prompt, point_label = get_points_from_mask(prediction_)
            if havebbox:
                sam_predictor.set_image(img_rgb)
                masks, scores, logits = sam_predictor.predict(
                    point_coords=point_prompt,
                    point_labels=point_label,
                    box=bbox_prompt[None, :],
                    multimask_output=True, # If True, return many masks
                )
                best_mask = masks[np.argmax(scores), :, :]
            else:
                best_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
            sam_prediction[:,:,i] = best_mask.astype(np.uint8)
    # print('view_{} finished'.format(view))
    return sam_prediction



def test_single_case_AL(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    # feats = net.featuremap_center
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
                
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    print('Shape of label_map: ', label_map.shape)
    return label_map, score_map


def test_all_case_AL_SAM(net, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=18, stride_z=4, model_type="vit_t", checkpoint_path='./sam_weights/mobile_sam.pt', strategy='all', bbox_strategy='2D'):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/data/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 2))
    print("Inference begin")
    score_prediction_list = []
    image_name_list = []
    label_prediction_list = []
    sam_label_list = []
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        # label = h5f['label'][:]
        label_prediction, score_prediction = test_single_case_AL(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        ### Using SAM to predict each slice from 3 views
        sam_prediction_fused = np.zeros(label_prediction.shape)
        if bbox_strategy == '2D':
            # For view 0
            sam_prediction_view0 = sam_fixed_predictor(image, label_prediction, view=0, model_type=model_type, checkpoint_path=checkpoint_path)
            # For view 1
            sam_prediction_view1 = sam_fixed_predictor(image, label_prediction, view=1, model_type=model_type, checkpoint_path=checkpoint_path)
            # For view 2
            sam_prediction_view2 = sam_fixed_predictor(image, label_prediction, view=2, model_type=model_type, checkpoint_path=checkpoint_path)
        elif bbox_strategy == '3D':
            # For view 0
            sam_prediction_view0 = sam_fixed_predictor_3D_box(image, label_prediction, view=0, model_type=model_type, checkpoint_path=checkpoint_path)
            # For view 1
            sam_prediction_view1 = sam_fixed_predictor_3D_box(image, label_prediction, view=1, model_type=model_type, checkpoint_path=checkpoint_path)
            # For view 2
            sam_prediction_view2 = sam_fixed_predictor_3D_box(image, label_prediction, view=2, model_type=model_type, checkpoint_path=checkpoint_path)

        # Fuse the 3 views
        sam_prediction_fused = sam_prediction_view0 + sam_prediction_view1 + sam_prediction_view2
        sam_prediction_fused_prob = (sam_prediction_view0 + sam_prediction_view1 + sam_prediction_view2)/3
        # get the final label
        if strategy == 'all':
            sam_prediction_fuse = sam_prediction_fused_prob * (sam_prediction_fused_prob == 1)
        elif strategy == 'majority':
            sam_prediction_fuse = (sam_prediction_fused >= 2).astype(np.uint8)
        elif strategy == 'soft':
            # 2 channels
            sam_prediction_fuse = sam_prediction_fused_prob.astype(np.uint8)

        # append score_prediction
        score_prediction_list.append(score_prediction)
        # append image_name
        image_name_list.append(image_path.split('/')[-1].split('.')[0])
        # append label_prediction
        label_prediction_list.append(label_prediction)       
        # append sam_label
        sam_label_list.append(sam_prediction_fuse)
    print("Inference end")
    return score_prediction_list, image_name_list, label_prediction_list, sam_label_list


def test_all_case_AL_SAM_singleview(net, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=18, stride_z=4, model_type="vit_t", checkpoint_path='./sam_weights/mobile_sam.pt', view=2):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/data/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 2))
    print("Inference begin")
    score_prediction_list = []
    image_name_list = []
    label_prediction_list = []
    sam_label_list = []
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        # label = h5f['label'][:]
        label_prediction, score_prediction = test_single_case_AL(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        ### Using SAM to predict each slice from 3 views
        sam_prediction_fused = np.zeros(label_prediction.shape)
        # For view 2
        sam_prediction_view2 = sam_fixed_predictor(image, label_prediction, view=2, model_type=model_type, checkpoint_path=checkpoint_path)
        # Fuse the 3 views
        sam_prediction_fused = sam_prediction_view2
        # get the final label (all agree)
        sam_prediction_fuse = sam_prediction_fused

        # append score_prediction
        score_prediction_list.append(score_prediction)
        # append image_name
        image_name_list.append(image_path.split('/')[-1].split('.')[0])
        # append label_prediction
        label_prediction_list.append(label_prediction)       
        # append sam_label
        sam_label_list.append(sam_prediction_fuse)
    print("Inference end")
    return score_prediction_list, image_name_list, label_prediction_list, sam_label_list



"""
Update: use prototype-based pseudo labels to generate prompts for SAM
"""
def test_single_case_AL_PrototypePredict(net, image, stride_xy, stride_z, patch_size, num_classes=2, feats_channel=16):

    # Prototype-based non-parametric function
    def getPrototype(features, mask, class_confidence):
        # adjust the features H, W shape
        fts = F.interpolate(features, size=mask.shape[-3:], mode='trilinear') 
        # masked average pooling
        mask_new = mask.unsqueeze(1)  # bs x 1 x Z x H x W
        # get the masked features
        masked_features = torch.mul(fts, mask_new)  # here use a broadcast mechanism: https://blog.csdn.net/da_kao_la/article/details/87484403
        masked_fts = torch.sum(masked_features*class_confidence, dim=(2, 3, 4)) / ((mask_new*class_confidence).sum(dim=(2, 3, 4)) + 1e-5)  # bs x C
        # print(sum1.shape)
        # print('masked fts:', masked_fts.shape)
        return masked_fts


    def calDist(fts_adj_size, prototype):
        scalar = 20
        dist = F.cosine_similarity(fts_adj_size, prototype[..., None, None, None], dim=1) * scalar
        return dist

    def prototype_pred(src_prototypes, feature_tgt, mask_src, class_nums):
        # 1 extract the foreground features via masked average pooling
        feature_tgt_adj = F.interpolate(feature_tgt, size=mask_src.shape[-3:], mode='trilinear')  # 3D uses tri, 2D uses bilinear, [2, 256, 96, 96, 96]
        # print('feature_tgt_adj:', feature_tgt_adj.shape)
        for class_index in range(class_nums):
            dist = calDist(feature_tgt_adj, src_prototypes[class_index]).unsqueeze(1)
            final_dist = dist if class_index == 0 else torch.cat((final_dist, dist), 1)
        final_dist_soft = torch.softmax(final_dist, dim=1)
        return final_dist_soft
           
    w, h, d = image.shape
    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)
    # initialize the feature map (fts_channel, H, W, D)
    feats_map = np.zeros((feats_channel, ) + image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    feats = net.featuremap.detach().cpu().numpy()
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                # Get the features_map for each voxel like the score_map
                feats_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = feats_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + feats[0, :, :, :, :]
                # cnt is the count of patches that overlap with each voxel
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
                
    score_map = score_map/np.expand_dims(cnt, axis=0)
    # feats_map
    feats_map = feats_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        feats_map = feats_map[:, wl_pad:wl_pad +
                                w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    
    # get the prototypes (1, numpy to tensor; 2, get the prototype, 3, get the prototype-based pseudo labels)
    feats_map_tensor = torch.from_numpy(feats_map).cuda().unsqueeze(0) # add a batch size 1
    score_map_tensor = torch.from_numpy(score_map).cuda().unsqueeze(0)
    label_map_tensor = torch.from_numpy(label_map).cuda().unsqueeze(0)
    # get the prototypes
    feats_map_confidence = []
    feats_map_prototypes = []
    for class_index in range(num_classes):
        confidence = score_map_tensor[:, class_index, :, :, :].unsqueeze(1).detach()
        feats_map_confidence.append(confidence)
    for class_index in range(num_classes):
        feats_proto = getPrototype(feats_map_tensor, (label_map_tensor == class_index).float(), feats_map_confidence[class_index]).detach()
        feats_map_prototypes.append(feats_proto)
    # get the prototype-based pseudo labels
    pseudo_label_map_soft = prototype_pred(feats_map_prototypes, feats_map_tensor, label_map_tensor, num_classes)
    pseudo_label_map_soft_np = pseudo_label_map_soft.cpu().numpy()[0, ...]  
    print('check pseudo_label_map_soft_np shape:' , pseudo_label_map_soft_np.shape) 
    # argmax and reduce the dimension
    pseudo_label_map_np = torch.argmax(pseudo_label_map_soft, dim=1, keepdim=False).cpu().numpy()
    pseudo_label_map_np = pseudo_label_map_np[0, :, :, :]

    return label_map, score_map, pseudo_label_map_np, pseudo_label_map_soft_np


def test_all_case_AL_SAM_ProtoPredict(net, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=18, stride_z=4, model_type="vit_t", checkpoint_path='./sam_weights/mobile_sam.pt', strategy='all', add_point=2):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/data/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    print("Inference begin")
    score_prediction_list = []
    image_name_list = []
    label_prediction_list = []
    sam_label_list = []
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        # label = h5f['label'][:]
        label_prediction, score_prediction, pseudo_label_map_np, pseudo_label_map_soft_np = test_single_case_AL_PrototypePredict(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        ### Using SAM to predict each slice from 3 views
        sam_prediction_fused = np.zeros(label_prediction.shape)

        # For view 0
        pseudo_label_map_soft_np_obj = pseudo_label_map_soft_np[1, :, :, :] # H, W, D
        sam_prediction_view0 = sam_fixed_predictor_dualpredict(image, pseudo_label_map_np, label_prediction, pseudo_label_map_soft_np_obj, view=0, model_type=model_type, checkpoint_path=checkpoint_path, add_point=add_point)
        # For view 1
        sam_prediction_view1 = sam_fixed_predictor_dualpredict(image, pseudo_label_map_np, label_prediction, pseudo_label_map_soft_np_obj, view=1, model_type=model_type, checkpoint_path=checkpoint_path, add_point=add_point)
        # For view 2
        sam_prediction_view2 = sam_fixed_predictor_dualpredict(image, pseudo_label_map_np, label_prediction, pseudo_label_map_soft_np_obj, view=2, model_type=model_type, checkpoint_path=checkpoint_path, add_point=add_point)

        # Fuse the 3 views
        sam_prediction_fused = sam_prediction_view0 + sam_prediction_view1 + sam_prediction_view2
        sam_prediction_fused_prob = (sam_prediction_view0 + sam_prediction_view1 + sam_prediction_view2)/3
        # get the final label
        if strategy == 'all':
            sam_prediction_fuse = sam_prediction_fused_prob * (sam_prediction_fused_prob == 1)
        elif strategy == 'majority':
            sam_prediction_fuse = (sam_prediction_fused >= 2).astype(np.uint8)

        # append score_prediction
        score_prediction_list.append(score_prediction)
        # append image_name
        image_name_list.append(image_path.split('/')[-1].split('.')[0])
        # append label_prediction
        label_prediction_list.append(label_prediction)       
        # append sam_label
        sam_label_list.append(sam_prediction_fuse)
    print("Inference end")
    return score_prediction_list, image_name_list, label_prediction_list, sam_label_list


def sam_fixed_predictor_dualpredict(image, prediction, ori_pred, pro_similarity_map, view=2, model_type="vit_t", checkpoint_path='./sam_weights/mobile_sam.pt', add_point=2): # image shape: (H, W, D), prediction shape: (H, W, D)
    # print('image shape: ', image.shape)
    sam_prediction = np.zeros(prediction.shape)
    print('SAM model type: ', model_type)
    # model_type = "vit_b"  # Example model type: vit_b, vit_h
    # checkpoint_path = './sam_vit_b_01ec64.pth' # sam_vit_b_01ec64, sam_vit_h_4b8939
    # Load the model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    if add_point==2:
        print('prompting mode: positive topk + negative one')

    for i in range(image.shape[view]):
        if view == 0:
            img = image[i,:,:]
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img = img.astype(np.uint8)
            img_rgb = np.dstack((img, img, img))
            prediction_ = prediction[i,:,:] # prototype-based pseudo label
            ori_pred_ = ori_pred[i,:,:] # original model prediction
            pro_similarity_map_ = pro_similarity_map[i,:,:] # prototype-based similarity map
            bbox_prompt, havebbox = get_bounding_box(prediction_) # list
            bbox_prompt = np.array(bbox_prompt)
            if add_point==1:
                point_prompt, point_label = get_points_from_mask(ori_pred_)   
            elif add_point==0:
                point_prompt, point_label = None, None
            elif add_point==2:
                point_prompt, point_label = get_points_from_mask_add_neg_topk(prediction_, pro_similarity_map_, topk=5)
                           
            # print('point_prompt: ', point_prompt)
            if havebbox:
                sam_predictor.set_image(img_rgb)
                masks, scores, logits = sam_predictor.predict(
                    point_coords=point_prompt,
                    point_labels=point_label,
                    box=bbox_prompt[None, :],
                    multimask_output=True, # If True, return many masks
                )
                best_mask = masks[np.argmax(scores), :, :]
            else:
                best_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
            sam_prediction[i,:,:] = best_mask.astype(np.uint8)
        elif view == 1:
            img = image[:,i,:]
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img = img.astype(np.uint8)
            img_rgb = np.dstack((img, img, img))
            prediction_ = prediction[:,i,:]
            ori_pred_ = ori_pred[:,i,:]
            pro_similarity_map_ = pro_similarity_map[:,i,:]
            bbox_prompt, havebbox = get_bounding_box(prediction_)
            bbox_prompt = np.array(bbox_prompt)
            if add_point==1:
                point_prompt, point_label = get_points_from_mask(ori_pred_)   
            elif add_point==0:
                point_prompt, point_label = None, None
            elif add_point==2:
                point_prompt, point_label = get_points_from_mask_add_neg_topk(prediction_, pro_similarity_map_, topk=5)
            if havebbox:
                sam_predictor.set_image(img_rgb)
                masks, scores, logits = sam_predictor.predict(
                    point_coords=point_prompt,
                    point_labels=point_label,
                    box=bbox_prompt[None, :],
                    multimask_output=True, # If True, return many masks
                )
                best_mask = masks[np.argmax(scores), :, :]
            else:
                best_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
            sam_prediction[:,i,:] = best_mask.astype(np.uint8)
        elif view == 2:
            img = image[:,:,i]
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img = img.astype(np.uint8)
            img_rgb = np.dstack((img, img, img))
            prediction_ = prediction[:,:,i]
            ori_pred_ = ori_pred[:,:,i]
            pro_similarity_map_ = pro_similarity_map[:,:,i]
            bbox_prompt, havebbox = get_bounding_box(prediction_)
            bbox_prompt = np.array(bbox_prompt)
            if add_point==1:
                point_prompt, point_label = get_points_from_mask(ori_pred_)   
            elif add_point==0:
                point_prompt, point_label = None, None
            elif add_point==2:
                point_prompt, point_label = get_points_from_mask_add_neg_topk(prediction_, pro_similarity_map_, topk=5)
            if havebbox:
                sam_predictor.set_image(img_rgb)
                masks, scores, logits = sam_predictor.predict(
                    point_coords=point_prompt,
                    point_labels=point_label,
                    box=bbox_prompt[None, :],
                    multimask_output=True, # If True, return many masks
                )
                best_mask = masks[np.argmax(scores), :, :]
            else:
                best_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
            sam_prediction[:,:,i] = best_mask.astype(np.uint8)
    # print('view_{} finished'.format(view))
    return sam_prediction



"""
The following section: iterative refinement using previous SAM logits
"""
def test_all_case_AL_SAM_ProtoPredict_iterative(net, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=18, stride_z=4, model_type="vit_t", checkpoint_path='./sam_weights/mobile_sam.pt', strategy='all', add_point=2):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/data/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    print("Inference begin")
    score_prediction_list = []
    image_name_list = []
    label_prediction_list = []
    sam_label_list = []
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        # label = h5f['label'][:]
        label_prediction, score_prediction, pseudo_label_map_np, pseudo_label_map_soft_np = test_single_case_AL_PrototypePredict(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        ### Using SAM to predict each slice from 3 views
        sam_prediction_fused = np.zeros(label_prediction.shape)

        # For view 0
        pseudo_label_map_soft_np_obj = pseudo_label_map_soft_np[1, :, :, :] # H, W, D
        sam_prediction_view0 = sam_fixed_predictor_iterative_2stage(image, pseudo_label_map_np, label_prediction, pseudo_label_map_soft_np_obj, view=0, model_type=model_type, checkpoint_path=checkpoint_path, add_point=add_point)
        # For view 1
        sam_prediction_view1 = sam_fixed_predictor_iterative_2stage(image, pseudo_label_map_np, label_prediction, pseudo_label_map_soft_np_obj, view=1, model_type=model_type, checkpoint_path=checkpoint_path, add_point=add_point)
        # For view 2
        sam_prediction_view2 = sam_fixed_predictor_iterative_2stage(image, pseudo_label_map_np, label_prediction, pseudo_label_map_soft_np_obj, view=2, model_type=model_type, checkpoint_path=checkpoint_path, add_point=add_point)

        # Fuse the 3 views
        sam_prediction_fused = sam_prediction_view0 + sam_prediction_view1 + sam_prediction_view2
        sam_prediction_fused_prob = (sam_prediction_view0 + sam_prediction_view1 + sam_prediction_view2)/3
        # get the final label
        if strategy == 'all':
            sam_prediction_fuse = sam_prediction_fused_prob * (sam_prediction_fused_prob == 1)
        elif strategy == 'majority':
            sam_prediction_fuse = (sam_prediction_fused >= 2).astype(np.uint8)

        # append score_prediction
        score_prediction_list.append(score_prediction)
        # append image_name
        image_name_list.append(image_path.split('/')[-1].split('.')[0])
        # append label_prediction
        label_prediction_list.append(label_prediction)       
        # append sam_label
        sam_label_list.append(sam_prediction_fuse)
    print("Inference end")
    return score_prediction_list, image_name_list, label_prediction_list, sam_label_list


# Will use the previous SAM logits as mask_input as a new prompt
def sam_fixed_predictor_iterative_2stage(image, prediction, ori_pred, pro_similarity_map, view=2, model_type="vit_t", checkpoint_path='./sam_weights/mobile_sam.pt', add_point=2): # image shape: (H, W, D), prediction shape: (H, W, D)
    # print('image shape: ', image.shape)
    sam_prediction = np.zeros(prediction.shape)
    print('SAM model type: ', model_type)
    # model_type = "vit_b"  # Example model type: vit_b, vit_h
    # checkpoint_path = './sam_vit_b_01ec64.pth' # sam_vit_b_01ec64, sam_vit_h_4b8939
    # Load the model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    if add_point==2:
        print('prompting mode: positive topk + negative one')

    for i in range(image.shape[view]):
        if view == 0:
            img = image[i,:,:]
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img = img.astype(np.uint8)
            img_rgb = np.dstack((img, img, img))
            prediction_ = prediction[i,:,:] # prototype-based pseudo label
            ori_pred_ = ori_pred[i,:,:] # original model prediction
            pro_similarity_map_ = pro_similarity_map[i,:,:] # prototype-based similarity map
            bbox_prompt, havebbox = get_bounding_box(prediction_) # list
            bbox_prompt = np.array(bbox_prompt)
            if add_point==1:
                point_prompt, point_label = get_points_from_mask(ori_pred_)   
            elif add_point==0:
                point_prompt, point_label = None, None
            elif add_point==2:
                point_prompt, point_label = get_points_from_mask_add_neg_topk(prediction_, pro_similarity_map_, topk=5)  
            # print('point_prompt: ', point_prompt)
            if havebbox:
                sam_predictor.set_image(img_rgb)
                masks, scores, logits = sam_predictor.predict(
                    point_coords=point_prompt,
                    point_labels=point_label,
                    box=bbox_prompt[None, :],
                    multimask_output=True, # If True, return many masks
                )
                best_mask = masks[np.argmax(scores), :, :]
                # 2-stage: using best logits as the new prompt
                # print('iterative refinement: 2-stage')
                best_logits = logits[np.argmax(scores), :, :]
                bbox_prompt, havebbox = get_bounding_box(best_mask)
                bbox_prompt = np.array(bbox_prompt)
                masks, scores, logits = sam_predictor.predict(
                    point_coords=point_prompt,
                    point_labels=point_label,
                    box=bbox_prompt[None, :],
                    mask_input=best_logits[None, :, :],
                    multimask_output=True, # If True, return many masks
                )
                best_mask = masks[np.argmax(scores), :, :]
            else:
                best_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
            sam_prediction[i,:,:] = best_mask.astype(np.uint8)

        elif view == 1:
            img = image[:,i,:]
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img = img.astype(np.uint8)
            img_rgb = np.dstack((img, img, img))
            prediction_ = prediction[:,i,:]
            ori_pred_ = ori_pred[:,i,:]
            pro_similarity_map_ = pro_similarity_map[:,i,:]
            bbox_prompt, havebbox = get_bounding_box(prediction_)
            bbox_prompt = np.array(bbox_prompt)
            if add_point==1:
                point_prompt, point_label = get_points_from_mask(ori_pred_)   
            elif add_point==0:
                point_prompt, point_label = None, None
            elif add_point==2:
                point_prompt, point_label = get_points_from_mask_add_neg_topk(prediction_, pro_similarity_map_, topk=5)
            if havebbox:
                sam_predictor.set_image(img_rgb)
                masks, scores, logits = sam_predictor.predict(
                    point_coords=point_prompt,
                    point_labels=point_label,
                    box=bbox_prompt[None, :],
                    multimask_output=True, # If True, return many masks
                )
                best_mask = masks[np.argmax(scores), :, :]
                # 2-stage: using best logits as the new prompt
                # print('iterative refinement: 2-stage')
                best_logits = logits[np.argmax(scores), :, :]
                bbox_prompt, havebbox = get_bounding_box(best_mask)
                bbox_prompt = np.array(bbox_prompt)
                masks, scores, logits = sam_predictor.predict(
                    point_coords=point_prompt,
                    point_labels=point_label,
                    box=bbox_prompt[None, :],
                    mask_input=best_logits[None, :, :],
                    multimask_output=True, # If True, return many masks
                )
                best_mask = masks[np.argmax(scores), :, :]
            else:
                best_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
            sam_prediction[:,i,:] = best_mask.astype(np.uint8)
        elif view == 2:
            img = image[:,:,i]
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img = img.astype(np.uint8)
            img_rgb = np.dstack((img, img, img))
            prediction_ = prediction[:,:,i]
            ori_pred_ = ori_pred[:,:,i]
            pro_similarity_map_ = pro_similarity_map[:,:,i]
            bbox_prompt, havebbox = get_bounding_box(prediction_)
            bbox_prompt = np.array(bbox_prompt)
            if add_point==1:
                point_prompt, point_label = get_points_from_mask(ori_pred_)   
            elif add_point==0:
                point_prompt, point_label = None, None
            elif add_point==2:
                point_prompt, point_label = get_points_from_mask_add_neg_topk(prediction_, pro_similarity_map_, topk=5)
            if havebbox:
                sam_predictor.set_image(img_rgb)
                masks, scores, logits = sam_predictor.predict(
                    point_coords=point_prompt,
                    point_labels=point_label,
                    box=bbox_prompt[None, :],
                    multimask_output=True, # If True, return many masks
                )
                best_mask = masks[np.argmax(scores), :, :]
                # 2-stage: using best logits as the new prompt
                # print('iterative refinement: 2-stage')
                best_logits = logits[np.argmax(scores), :, :]
                bbox_prompt, havebbox = get_bounding_box(best_mask)
                bbox_prompt = np.array(bbox_prompt)
                masks, scores, logits = sam_predictor.predict(
                    point_coords=point_prompt,
                    point_labels=point_label,
                    box=bbox_prompt[None, :],
                    mask_input=best_logits[None, :, :],
                    multimask_output=True, # If True, return many masks
                )
                best_mask = masks[np.argmax(scores), :, :]
            else:
                best_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
            sam_prediction[:,:,i] = best_mask.astype(np.uint8)
    # print('view_{} finished'.format(view))
    return sam_prediction