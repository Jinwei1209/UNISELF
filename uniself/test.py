import os
import torch
import argparse
import numpy as np
import nibabel as nib

from torch.autograd import Variable
from .unet2d import Unet as Unet2D
from scipy.ndimage import label
from collections import OrderedDict
from tqdm import tqdm


def loadNii(file_name):

    return nib.load(file_name).get_fdata()


def convert_state_dict(state_dict):
    
    new_state_dict = OrderedDict()

    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state

    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict


def zScoreNorm(img, ignore_zero = True):

    if ignore_zero:
        img[img==0] = np.nan
        res = (img * 1.0 - np.nanmean(img)) / np.nanstd(img)
        res[np.where(np.isnan(res))] = 0
    else:
        res = (img - np.min(img)) / np.max(img)
        res = (res * 1.0 - np.mean(res)) / np.std(res)

    return res


def brainPadding(img, num_slices=224):
    pad_x = (num_slices - img.shape[0]) // 2
    pad_y = (num_slices - img.shape[1]) // 2
    pad_z = (num_slices - img.shape[2]) // 2

    # Calculate the remaining padding
    pad_x_remainder = num_slices - img.shape[0] - pad_x
    pad_y_remainder = num_slices - img.shape[1] - pad_y
    pad_z_remainder = num_slices - img.shape[2] - pad_z

    img = np.pad(img, 
                ((pad_x, pad_x_remainder), 
                (pad_y, pad_y_remainder), 
                (pad_z, pad_z_remainder)), 
                mode='constant')

    return img


def centerCropToOriSize(original_array, crop_size):

    # Calculate the crop boundaries
    crop_height = crop_size[0]
    crop_width = crop_size[1]
    crop_depth = crop_size[2]

    original_height, original_width, original_depth = original_array.shape

    # Calculate the starting indices for each dimension
    start_height = (original_height - crop_height) // 2
    start_width = (original_width - crop_width) // 2
    start_depth = (original_depth - crop_depth) // 2

    # Perform the center crop
    cropped_array = original_array[start_height:start_height + crop_height,
                                start_width:start_width + crop_width,
                                start_depth:start_depth + crop_depth]

    return cropped_array


def centerPaddingToOriSize(original_array, ori_size_before_cropping):

    pad_x = (ori_size_before_cropping[0] - original_array.shape[0]) // 2
    pad_y = (ori_size_before_cropping[1] - original_array.shape[1]) // 2
    pad_z = (ori_size_before_cropping[2] - original_array.shape[2]) // 2

    # Calculate the remaining padding
    pad_x_remainder = ori_size_before_cropping[0] - original_array.shape[0] - pad_x
    pad_y_remainder = ori_size_before_cropping[1] - original_array.shape[1] - pad_y
    pad_z_remainder = ori_size_before_cropping[2] - original_array.shape[2] - pad_z

    img = np.pad(original_array, 
                ((pad_x, pad_x_remainder), 
                (pad_y, pad_y_remainder), 
                (pad_z, pad_z_remainder)), 
                mode='constant')

    return img


def merge_overlapping_lesions(mask1, mask2):
    '''
        mask1: majority voting (no false postive between CSF)
        mask2: union (contain false postive between CSF)

        output: adding true positive lesions in union to majority voting
    '''

    # Label connected components
    labeled_mask1, num_labels1 = label(mask1)
    labeled_mask2, num_labels2 = label(mask2)

    # Iterate through unique labels in mask2 (excluding background label)
    for label2 in range(1, num_labels2 + 1):
        # Check if there's any overlap with mask1
        if np.any((labeled_mask1 > 0) & (labeled_mask2 == label2)):
            # Merge lesion regions from mask2 to mask1
            mask1[labeled_mask2 == label2] = 1

    return mask1


def singleRun(opt):

    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_id']

    # get multi-contrast input
    t1_path = opt['t1_path']
    t2_path = opt['t2_path']
    pd_path = opt['pd_path']
    t2flair_path = opt['t2flair_path']
    brain_mask_path = opt['brain_mask_path']

    img_order = []
    input_contrast_code = list('0000')
    idx_input_contrast_code = 0
    if t1_path != '':
        img_order.append('T1')
        input_contrast_code[idx_input_contrast_code] = '1'
    idx_input_contrast_code += 1

    if t2_path != '':
        img_order.append('T2')
        input_contrast_code[idx_input_contrast_code] = '1'
    idx_input_contrast_code += 1

    if pd_path != '':
        img_order.append('PD')
        input_contrast_code[idx_input_contrast_code] = '1'
    idx_input_contrast_code += 1
    
    if t2flair_path != '':
        img_order.append('FLAIR')
        input_contrast_code[idx_input_contrast_code] = '1'
    idx_input_contrast_code += 1

    mc = torch.zeros((1, 4)).cuda()
    for idx_mc, code in enumerate(input_contrast_code):
        if code == '1':
            mc[0, idx_mc] = 1

    print('Input contrasts:', img_order)
    print('Input contrast code:', input_contrast_code)
    print('Input mc:', mc)

    if opt['num_contrasts'] == 4:
        img_order = ['T1', 'T2', 'PD', 'FLAIR']
    
    mc_paths = [t1_path, t2_path, pd_path, t2flair_path]
    print(mc_paths)
    
    for idx, mc_path in enumerate(mc_paths):
        if mc_path != '':
            affine_input = nib.load(mc_path).affine
            mc_image = loadNii(mc_path)
            break

    if t1_path != '':
        t1_image = loadNii(t1_path)
    else:
        t1_image = np.zeros(mc_image.shape)

    if t2_path != '':
        t2_image = loadNii(t2_path)
    else:
        t2_image = np.zeros(mc_image.shape)

    if pd_path != '':
        pd_image = loadNii(pd_path)
    else:
        pd_image = np.zeros(mc_image.shape)

    if t2flair_path != '':
        t2flair_image = loadNii(t2flair_path)
    else:
        t2flair_image = np.zeros(mc_image.shape)

    opt['in_channels'] = (opt['num_adj_slices'] * 2 + 1) * opt['num_contrasts']
    
    if opt['inference_mode'] == 'fast':
        opt['random_flip'] = 0
        print("Fast inference mode")
    elif opt['inference_mode'] == 'standard':
        opt['random_flip'] = 1
        print("Standard inference mode")
    else:
        raise ValueError('Unkown inference mode: please select either \'fast\' or \'standard \' mode')

    # Getting components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt['model'] == 'Unet2D5':
        model = Unet2D(input_channels = opt['in_channels'], num_filters = opt['num_filters'], use_bn=opt['use_bn'])
    
    if opt['weight_path'] != '':
        weight_path = opt['weight_path']
        print('Use weight {}'.format(opt['weight_path']))
    else:
        raise ValueError('Weight path is not specified.')

    states = convert_state_dict(torch.load(weight_path))
    model.load_state_dict(states)
    model.to(device)
    model.eval()

    # padding
    if brain_mask_path == '':
        brain_mask_image = (abs(t1_image) > 0).astype(float)
    else:
        brain_mask_image = loadNii(brain_mask_path)

    t1_image *= brain_mask_image
    t2_image *= brain_mask_image
    pd_image *= brain_mask_image
    t2flair_image *= brain_mask_image

    img_x, img_y, img_z = t2flair_image.shape
    opt['ori_size_before_cropping'] = [img_x, img_y, img_z]
    flag_x_cropping, flag_y_cropping, flag_z_cropping = 0, 0, 0
    if img_x > 224:
        flag_x_cropping = 1
        t1_image = t1_image[img_x//2-112:img_x//2+112, ...]
        t2_image = t2_image[img_x//2-112:img_x//2+112, ...]
        pd_image = pd_image[img_x//2-112:img_x//2+112, ...]
        t2flair_image = t2flair_image[img_x//2-112:img_x//2+112, ...]

    if img_y > 224:
        flag_y_cropping = 1
        t1_image = t1_image[:, img_y//2-112:img_y//2+112, ...]
        t2_image = t2_image[:, img_y//2-112:img_y//2+112, ...]
        pd_image = pd_image[:, img_y//2-112:img_y//2+112, ...]
        t2flair_image = t2flair_image[:, img_y//2-112:img_y//2+112, ...]

    if img_z > 224:
        flag_z_cropping = 1
        t1_image = t1_image[:, :, img_z//2-112:img_z//2+112, ...]
        t2_image = t2_image[:, :, img_z//2-112:img_z//2+112, ...]
        pd_image = pd_image[:, :, img_z//2-112:img_z//2+112, ...]
        t2flair_image = t2flair_image[:, :, img_z//2-112:img_z//2+112, ...]

    img_x, img_y, img_z = t2flair_image.shape
    opt['ori_size'] = [img_x, img_y, img_z]

    t1_image = zScoreNorm(brainPadding(t1_image, opt['num_slices']))
    t1_image = np.pad(t1_image, 
                    ((opt['num_adj_slices'], opt['num_adj_slices']), 
                    (opt['num_adj_slices'], opt['num_adj_slices']), 
                    (opt['num_adj_slices'], opt['num_adj_slices'])), 
                    mode='constant')

    t2_image = zScoreNorm(brainPadding(t2_image, opt['num_slices']))
    t2_image = np.pad(t2_image, 
                    ((opt['num_adj_slices'], opt['num_adj_slices']), 
                    (opt['num_adj_slices'], opt['num_adj_slices']), 
                    (opt['num_adj_slices'], opt['num_adj_slices'])), 
                    mode='constant')

    pd_image = zScoreNorm(brainPadding(pd_image, opt['num_slices']))
    pd_image = np.pad(pd_image, 
                    ((opt['num_adj_slices'], opt['num_adj_slices']), 
                    (opt['num_adj_slices'], opt['num_adj_slices']), 
                    (opt['num_adj_slices'], opt['num_adj_slices'])), 
                    mode='constant')

    t2flair_image = zScoreNorm(brainPadding(t2flair_image, opt['num_slices']))
    t2flair_image = np.pad(t2flair_image, 
                    ((opt['num_adj_slices'], opt['num_adj_slices']), 
                    (opt['num_adj_slices'], opt['num_adj_slices']), 
                    (opt['num_adj_slices'], opt['num_adj_slices'])), 
                    mode='constant')

    imgs = {}
    imgs['T1'] = t1_image
    imgs['T2'] = t2_image
    imgs['PD'] = pd_image
    imgs['FLAIR'] = t2flair_image

    permutes = [[0, 1, 2], [1, 0, 2], [2, 0, 1], [0, 2, 1], [1, 2, 0], [2, 1, 0]]
    mem_imgs = []
    for permute_id in range(len(permutes)):

        permute = permutes[permute_id]

        img_slices = []
        for key in img_order:
            tmp = np.transpose(imgs[key], permute)
            img_slices.append(tmp)
        
        im_size = img_slices[0].shape
        for i in range(opt['num_adj_slices'], len(img_slices[0])-opt['num_adj_slices']):
            imgs_tmp = {}
            for idx_key, key in enumerate(img_order):
                imgs_tmp[key] = img_slices[idx_key][i-opt['num_adj_slices']:i+1+opt['num_adj_slices'], 
                    opt['num_adj_slices']:im_size[1]-opt['num_adj_slices'], opt['num_adj_slices']:im_size[2]-opt['num_adj_slices']]
            mem_imgs.append(imgs_tmp)

    num_samples = ((len(img_slices[0]) - 2*opt['num_adj_slices']) * len(permutes))

    print("Three plane slices has been stored in memory with %d slices" % num_samples)

    if opt['random_flip'] == 0:
        num_flips = 1
        folder_indices = [4]
        folder_indices2 = [2]
    else:
        num_flips = 4
        folder_indices = [16]
        folder_indices2 = [8]
    # for permutes back
    permutes = [[0, 1, 2], [1, 0, 2], [1, 2, 0], [0, 2, 1], [2, 0, 1], [2, 1, 0]]

    # prediction
    if 1:
        save_pred_slices = [[] for i in range(num_flips)]
        save_prob_slices = [[] for i in range(num_flips)]
        num_slices = 0
        for idx in tqdm(range(num_samples)):

            imgs = mem_imgs[idx]
            tmp = imgs['T1'].shape
            img = np.empty([0, tmp[0], tmp[1], tmp[2]])

            for key in img_order:
                tImg = imgs[key]
                tImg = np.reshape(tImg, (1, *tImg.shape))
                img = np.append(img, tImg, axis = 0)

            img = torch.from_numpy(img).float()
            img.transpose_(3, 2).transpose_(2, 1)
            img = torch.cat([img[:, :, i, ...] for i in range(img.shape[2])], dim=0)

            for flip_id in range(num_flips):

                with torch.no_grad():

                    num_slices += 1

                    img_input = Variable(img[None, ...].cuda()).to(device)
                    if flip_id == 1:
                        img_input = torch.flip(img_input, dims=[2])
                    elif flip_id == 2:
                        img_input = torch.flip(img_input, dims=[3])
                    elif flip_id == 3:
                        img_input = torch.flip(img_input, dims=[2, 3])
                    
                    out = model(img_input, ''.join(input_contrast_code))
                    if flip_id == 1:
                        out = torch.flip(out, dims=[2])
                    elif flip_id == 2:
                        out = torch.flip(out, dims=[3])
                    elif flip_id == 3:
                        out = torch.flip(out, dims=[2, 3])

                    pred_prob = out.sigmoid().squeeze(0).detach().cpu().numpy()
                    pred_prob = np.transpose(pred_prob, (0, 2, 1))

                    save_prob_slices[flip_id].append(pred_prob)

                    out = (out.sigmoid()>=0.5)
                    pred = out.squeeze(0).detach().cpu().numpy()    
                    save_pred = np.transpose(pred, (0, 2, 1))
                    save_pred = save_pred.astype(float)

                    save_pred_slices[flip_id].append(save_pred)

        # fusion
        save_pred_list = [np.concatenate(save_pred_slices[i], axis=0) for i in range(num_flips)]
        save_pred = np.concatenate(save_pred_list, axis=0)
        save_pred_acc = 0
        save_prob_list = [np.concatenate(save_prob_slices[i], axis=0) for i in range(num_flips)]
        save_prob = np.concatenate(save_prob_list, axis=0)
        save_prob_acc = 0
        for ii in range(0, num_flips*6, 6):

            save_pred_sagittal = np.transpose(save_pred[opt['num_slices']*(ii+0):opt['num_slices']*(ii+1), :, :], permutes[0])
            save_pred_sagittal = save_pred_sagittal.astype(float)
            save_pred_sagittal = centerCropToOriSize(save_pred_sagittal, opt['ori_size'])

            save_pred_coronal = np.transpose(save_pred[opt['num_slices']*(ii+1):opt['num_slices']*(ii+2), :, :], permutes[1])
            save_pred_coronal = save_pred_coronal.astype(float)
            save_pred_coronal = centerCropToOriSize(save_pred_coronal, opt['ori_size'])

            save_pred_axial = np.transpose(save_pred[opt['num_slices']*(ii+2):opt['num_slices']*(ii+3), :, :], permutes[2])
            save_pred_axial = save_pred_axial.astype(float)
            save_pred_axial = centerCropToOriSize(save_pred_axial, opt['ori_size'])

            save_pred_acc += (save_pred_sagittal + save_pred_coronal + save_pred_axial)

            # process prob map
            save_prob_sagittal = np.transpose(save_prob[opt['num_slices']*(ii+0):opt['num_slices']*(ii+1), :, :], permutes[0])
            save_prob_sagittal = save_prob_sagittal.astype(float)
            save_prob_sagittal = centerCropToOriSize(save_prob_sagittal, opt['ori_size'])

            save_prob_coronal = np.transpose(save_prob[opt['num_slices']*(ii+1):opt['num_slices']*(ii+2), :, :], permutes[1])
            save_prob_coronal = save_prob_coronal.astype(float)
            save_prob_coronal = centerCropToOriSize(save_prob_coronal, opt['ori_size'])

            save_prob_axial = np.transpose(save_prob[opt['num_slices']*(ii+2):opt['num_slices']*(ii+3), :, :], permutes[2])
            save_prob_axial = save_prob_axial.astype(float)
            save_prob_axial = centerCropToOriSize(save_prob_axial, opt['ori_size'])

            save_prob_acc += (save_prob_sagittal + save_prob_coronal + save_prob_axial)

            if len(permutes) == 6:

                save_pred_sagittal2 = np.transpose(save_pred[opt['num_slices']*(ii+3):opt['num_slices']*(ii+4), :, :], permutes[3])
                save_pred_sagittal2 = save_pred_sagittal2.astype(float)
                save_pred_sagittal2 = centerCropToOriSize(save_pred_sagittal2, opt['ori_size'])

                save_pred_coronal2 = np.transpose(save_pred[opt['num_slices']*(ii+4):opt['num_slices']*(ii+5), :, :], permutes[4])
                save_pred_coronal2 = save_pred_coronal2.astype(float)
                save_pred_coronal2 = centerCropToOriSize(save_pred_coronal2, opt['ori_size'])

                save_pred_axial2 = np.transpose(save_pred[opt['num_slices']*(ii+5):opt['num_slices']*(ii+6), :, :], permutes[5])
                save_pred_axial2 = save_pred_axial2.astype(float)
                save_pred_axial2 = centerCropToOriSize(save_pred_axial2, opt['ori_size'])

                save_pred_acc += (save_pred_sagittal2 + save_pred_coronal2 + save_pred_axial2)

                # process prob map
                save_prob_sagittal2 = np.transpose(save_prob[opt['num_slices']*(ii+3):opt['num_slices']*(ii+4), :, :], permutes[3])
                save_prob_sagittal2 = save_prob_sagittal2.astype(float)
                save_prob_sagittal2 = centerCropToOriSize(save_prob_sagittal2, opt['ori_size'])

                save_prob_coronal2 = np.transpose(save_prob[opt['num_slices']*(ii+4):opt['num_slices']*(ii+5), :, :], permutes[4])
                save_prob_coronal2 = save_prob_coronal2.astype(float)
                save_prob_coronal2 = centerCropToOriSize(save_prob_coronal2, opt['ori_size'])

                save_prob_axial2 = np.transpose(save_prob[opt['num_slices']*(ii+5):opt['num_slices']*(ii+6), :, :], permutes[5])
                save_prob_axial2 = save_prob_axial2.astype(float)
                save_prob_axial2 = centerCropToOriSize(save_prob_axial2, opt['ori_size'])

                save_prob_acc += (save_prob_sagittal2 + save_prob_coronal2 + save_prob_axial2)

        for folder_idx in folder_indices:

            for folder_idx2 in folder_indices2:

                print("Post-processing with fusion threshold {} and union threshold {}".format(folder_idx, folder_idx2))

                fusion_thrd = folder_idx
                union_thrd = folder_idx2

                save_pred_fusion = (save_pred_acc >= fusion_thrd)
                save_pred_union = (save_pred_acc >= union_thrd)

                save_pred_save = merge_overlapping_lesions(save_pred_fusion, save_pred_union)

        if flag_x_cropping or flag_y_cropping or flag_z_cropping:
            save_pred_save = centerPaddingToOriSize(save_pred_save, opt['ori_size_before_cropping'])
            save_pred_acc = centerPaddingToOriSize(save_pred_acc, opt['ori_size_before_cropping'])
            save_prob_acc = centerPaddingToOriSize(save_prob_acc, opt['ori_size_before_cropping'])

        nib.save(nib.Nifti1Image(save_pred_save.astype(float), affine_input, None), opt['lesion_seg_path'])

        confidence_map_path = opt['lesion_seg_path'].replace('.nii.gz', '_confidence_map.nii.gz')
        nib.save(nib.Nifti1Image(save_pred_acc.astype(float), affine_input, None), confidence_map_path)

        confidence_map_path = opt['lesion_seg_path'].replace('.nii.gz', '_probability_map.nii.gz')
        nib.save(nib.Nifti1Image((save_prob_acc/len(permutes)/num_flips).astype(float), affine_input, None), confidence_map_path)


def main(args=None):

    opt = {}

    parser = argparse.ArgumentParser(description = "ms lesion segmentation using uniself")
    parser.add_argument("-m", "--model", type = str, default = "Unet2D5")
    parser.add_argument("-w", "--weight_path", type = str, default = '')
    parser.add_argument("--num_contrasts", type = int, default = 4)
    parser.add_argument("--use_bn", type = int, default = 2)
    parser.add_argument("-ns", "--num_slices", type = int, default = 224)
    parser.add_argument("--gpu_id", type = str, default = "0")
    parser.add_argument("--use_gpu", type = bool, default = True)
    parser.add_argument("--num_adj_slices", type = int, default = 1)
    parser.add_argument("-nf", "--num_filters", type = list, default = [2**i for i in range(6, 11)])
    parser.add_argument("--inference_mode", type = str, default = "standard")  # fast or standard
    parser.add_argument("--t1_path", type = str, default = "")
    parser.add_argument("--t2_path", type = str, default = "")
    parser.add_argument("--pd_path", type = str, default = "")
    parser.add_argument("--t2flair_path", type = str, default = "")
    parser.add_argument("--lesion_seg_path", type = str, default = "")
    parser.add_argument("--brain_mask_path", type = str, default = "")

    opt = {**opt, **vars(parser.parse_args(args))}

    singleRun(opt)
