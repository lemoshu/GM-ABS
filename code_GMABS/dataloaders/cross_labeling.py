import glob
import os
import re
import h5py
import numpy as np
import shutil
import scipy.ndimage
import random
import SimpleITK as sitk

def find_3d_bounding_box(arr):
    """
    Find the 3D bounding box of an object in a 3D array.
    The object is assumed to be represented by non-zero values in the array.
    
    Args:
    arr (numpy.ndarray): A 3D numpy array.

    Returns:
    tuple: Bounding box in the format (x1, y1, z1, x2, y2, z2).
    """
    # Indices where the array is non-zero
    non_zero_indices = np.argwhere(arr)
    
    # If the array is entirely zero, return an empty bounding box
    assert non_zero_indices.size > 0, 'Array does not contain any non-zero elements!'

    # Minimum and maximum indices along each dimension
    min_coords = non_zero_indices.min(axis=0)
    max_coords = non_zero_indices.max(axis=0)

    # Construct the bounding box
    bounding_box = (min_coords[0], min_coords[1], min_coords[2],
                    max_coords[0], max_coords[1], max_coords[2])

    return bounding_box

# This new function is for labeling middle slices of the ROI from 3 views (indicated by the ground truth)
def cross_labeling_ROI_middle(ori_mask, planes=['xy', 'xz', 'yz']):
    # ori_mask: (x, y, z)
    # dst_mask: (x, y, z)
    dst_mask = np.zeros(ori_mask.shape)
    # find the 3D bounding box of the ROI (label==1) in the ori_mask, and then find the middle slice of the bounding box
    # bounding box: (x1, y1, z1, x2, y2, z2)
    ori_mask = np.array(ori_mask)
    # print('ori_mask shape and max:', ori_mask.shape, max(ori_mask))
    bounding_box = find_3d_bounding_box(ori_mask)
    print('bounding box:', bounding_box)
    if 'xy' in planes:
        dst_mask[:, :, int((bounding_box[2]+bounding_box[5])/2)] = ori_mask[:, :, int((bounding_box[2]+bounding_box[5])/2)]
    if 'xz' in planes:
        dst_mask[:, int((bounding_box[1]+bounding_box[4])/2), :] = ori_mask[:, int((bounding_box[1]+bounding_box[4])/2), :]
    if 'yz' in planes:
        dst_mask[int((bounding_box[0]+bounding_box[3])/2), :, :] = ori_mask[int((bounding_box[0]+bounding_box[3])/2), :, :]

    # generate a mask to indicate the labeled slices (these slices are not empty)
    labeled_slice_indicator = np.zeros(ori_mask.shape)
    if 'xy' in planes:
        labeled_slice_indicator[:, :, int((bounding_box[2]+bounding_box[5])/2)] = 1
    if 'xz' in planes:
        labeled_slice_indicator[:, int((bounding_box[1]+bounding_box[4])/2), :] = 1
    if 'yz' in planes:
        labeled_slice_indicator[int((bounding_box[0]+bounding_box[3])/2), :, :] = 1
    # print(max(labeled_slice_indicator.flatten()))
    return dst_mask, labeled_slice_indicator

def new_ROI_cross_labeling_dataset_generation(ori_Dir, new_Dir):
    print('ROI middle slice labeling')
    num = 0
    for case_name in os.listdir(ori_Dir):
        case_name = case_name.replace('\n', '').split('.')[0]
        print(case_name)
        ori_path = os.path.join(ori_Dir, case_name+'.h5')
        new_path = os.path.join(new_Dir, case_name+'.h5')
        # read the labels, cross labeling, and save h5 to new_path
        print(ori_path)
        h5f_ori = h5py.File(ori_path, 'r')
        label_ori = h5f_ori['label']
        print('ori:', label_ori.shape)
        dst_label, labeled_slice_indicator = cross_labeling_ROI_middle(label_ori)
        print('new:', dst_label.shape)
        f = h5py.File(new_path, 'w')
        f.create_dataset('image', data=h5f_ori['image'], compression="gzip")
        f.create_dataset('label', data=h5f_ori['label'], compression="gzip")
        f.create_dataset('cross_label', data=dst_label, compression="gzip")
        f.create_dataset('labeled_slice_indicator', data=labeled_slice_indicator, compression="gzip")
        f.close()
        num += 1
    print('num:', num)


# visualize the cross labeling; save the labels to nii.gz
def VIS_cross_labeling_generation(cross_labeling_h5_Dir, cross_labeling_nii_Dir):
    if not os.path.exists(cross_labeling_nii_Dir):
        os.makedirs(cross_labeling_nii_Dir)
    h5_list = glob.glob(os.path.join(cross_labeling_h5_Dir, '*.h5'))
    # print(h5_list)
    for h5_path in h5_list:
        # case_name is the last one, seperated by '/' and removed '.h5'
        case_name = h5_path.split('/')[-1].split('.')[0]
        print(case_name)
        h5f = h5py.File(h5_path, 'r')
        cross_label = h5f['cross_label']
        print('label shape:', cross_label.shape)
        # convert to nii.gz
        cross_label_itk = sitk.GetImageFromArray(cross_label)
        sitk.WriteImage(cross_label_itk, os.path.join(cross_labeling_nii_Dir, case_name+'.nii.gz'))

    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_list_path = '../../data/train.txt'
    test_list_path = '../../data/test.txt'
    ori_Dir_data = '../../data/LA_data_h5/data/'
    new_Dir = '../../data/LA_cross_labeling_h5/'
    new_Dir_data = '../../data/LA_cross_labeling_h5/data/'
    if not os.path.exists(new_Dir):
        os.makedirs(new_Dir)
    if not os.path.exists(new_Dir_data):
        os.makedirs(new_Dir_data)
    # copy train_list and test_list to new folder
    shutil.copyfile(train_list_path, os.path.join(new_Dir, 'train.txt'))
    shutil.copyfile(test_list_path, os.path.join(new_Dir, 'test.txt'))
    # generate new dataset
    # new_cross_labeling_dataset_generation(ori_Dir_data, new_Dir_data)
    new_ROI_cross_labeling_dataset_generation(ori_Dir_data, new_Dir_data)
    # visualize the cross labeling
    cross_labeling_h5_Dir = '../../data/LA_cross_labeling_h5/data/'
    cross_labeling_nii_Dir = '../../data/LA_cross_labeling_nii/'
    VIS_cross_labeling_generation(cross_labeling_h5_Dir, cross_labeling_nii_Dir)
