"""
"""

import sys
sys.path.append("/home/users/piyushb/projects/Background-Matting/")

import os, glob, time, argparse, pdb, cv2
from os.path import join
#import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from termcolor import colored

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from functions import *
from networks import ResnetConditionHR

torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print('CUDA Device: ' + os.environ["CUDA_VISIBLE_DEVICES"])

MODEL_DIR = "/home/users/piyushb/projects/Background-Matting/Models"


# TODO: video mode support
class BackgroundMattingV1(object):
    """docstring for BackgroundMattingV1"""
    def __init__(self, trained_model: str, reso: tuple = (512, 512), K: int = 25, gblur_kernel: tuple = (31, 31)):
        super(BackgroundMattingV1, self).__init__()

        self.reso = reso
        self.gblur_kernel = gblur_kernel
        self.K = K

        # load model
        self.netM = self.load_and_initalize_model(trained_model)

    def load_and_initalize_model(self, model_name: str):
        print(colored("=> Loading ResnetConditionHR model ..", "yellow"))
        model_main_dir = join(MODEL_DIR, model_name)
        model_path = glob.glob(join(model_main_dir, 'netG_epoch_*'))[0]
        
        netM=ResnetConditionHR(input_nc=(3,3,1,4),output_nc=4,n_blocks1=7,n_blocks2=3)
        netM=nn.DataParallel(netM)
        netM.load_state_dict(torch.load(model_path))
        netM.cuda()
        netM.eval()
        print(colored("=> Done", "yellow"))

        cudnn.benchmark=True

        return netM

    @staticmethod
    def read_image(path, resize=None):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if resize is not None:
            assert isinstance(resize, tuple) and len(resize) == 2
            image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)

        return image

    @staticmethod
    def convert_to_torch(image, transpose=True, transpose_order=(2, 0, 1), unsqueeze_order=1):
        if transpose:
            assert isinstance(transpose_order, tuple) and len(transpose_order) == len(image.shape)
            image = image.transpose(transpose_order)

        image = torch.from_numpy(image)
        for _ in range(unsqueeze_order):
            image = image.unsqueeze(0)

        image = 2 * image.float().div(255) - 1

        return image

    def __call__(self, src_img_path, src_back_path, src_img_mask_path, tgt_back_path):

        # load all images
        src_img = self.read_image(src_img_path)
        src_back = self.read_image(src_back_path, resize=src_img.shape[:2])
        src_img_mask = cv2.imread(src_img_mask_path, 0)
        tgt_back = self.read_image(tgt_back_path, resize=src_img.shape[:2])

        #Green-screen background
        gs_back = np.zeros(tgt_back.shape)
        gs_back[...,0]=120; gs_back[...,1]=255; gs_back[...,2]=155;

        ## create the multi-frame
        multi_fr_w=np.zeros((src_img.shape[0], src_img.shape[1], 4))
        multi_fr_w[..., 0] = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY);
        multi_fr_w[..., 1] = multi_fr_w[..., 0]
        multi_fr_w[..., 2] = multi_fr_w[..., 0]
        multi_fr_w[..., 3] = multi_fr_w[..., 0]

        #crop tightly
        _src_img = src_img
        bbox = get_bbox(src_img_mask, R=_src_img.shape[0], C=_src_img.shape[1])

        crop_list = [src_img, src_back, src_img_mask, tgt_back, gs_back, multi_fr_w]
        crop_list = crop_images(crop_list, self.reso, bbox)
        crop_src_img, crop_src_back, crop_src_img_mask, crop_tgt_back, crop_gs_back, crop_multi_fr_w = crop_list

        #process segmentation mask
        kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        crop_src_img_mask = crop_src_img_mask.astype(np.float32) / 255
        crop_src_img_mask[crop_src_img_mask > 0.2] = 1

        zero_id = np.nonzero(np.sum(crop_src_img_mask, axis=1) == 0)
        del_id = zero_id[0][zero_id[0] > 250]
        if len(del_id) > 0:
            del_id = [del_id[0] - 2, del_id[0] - 1, *del_id]
            crop_src_img_mask = np.delete(crop_src_img_mask, del_id, 0)
        crop_src_img_mask = cv2.copyMakeBorder(crop_src_img_mask, 0, self.K + len(del_id), 0, 0, cv2.BORDER_REPLICATE)

        crop_src_img_mask = cv2.erode(crop_src_img_mask, kernel_er, iterations=10)
        crop_src_img_mask = cv2.dilate(crop_src_img_mask, kernel_dil, iterations=5)
        crop_src_img_mask = cv2.GaussianBlur(crop_src_img_mask.astype(np.float32), self.gblur_kernel, 0)
        crop_src_img_mask = (255 * crop_src_img_mask).astype(np.uint8)
        crop_src_img_mask = np.delete(crop_src_img_mask, range(self.reso[0], self.reso[0] + self.K), 0)

        #convert to torch
        img = self.convert_to_torch(crop_src_img)
        bg = self.convert_to_torch(crop_src_back)
        rcnn_al = self.convert_to_torch(crop_src_img_mask, transpose=False, unsqueeze_order=2)
        multi_fr = self.convert_to_torch(crop_multi_fr_w)

        with torch.no_grad():
            img, bg, rcnn_al, multi_fr = Variable(img.cuda()), Variable(bg.cuda()), Variable(rcnn_al.cuda()), Variable(multi_fr.cuda())
            input_im = torch.cat([img, bg, rcnn_al, multi_fr], dim=1)
            
            alpha_pred, fg_pred_tmp = self.netM(img, bg, rcnn_al, multi_fr)
            
            al_mask = (alpha_pred > 0.95).type(torch.cuda.FloatTensor)

            # for regions with alpha > 0.95, simply use the image as fg
            fg_pred = img * al_mask + fg_pred_tmp * (1 - al_mask)

            alpha_out = to_image(alpha_pred[0,...])

            #refine alpha with connected component
            labels = label((alpha_out > 0.05).astype(int))
            try:
                assert( labels.max() != 0 )
            except:
                return {}
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

            alpha_out = alpha_out * largestCC
            alpha_out = (255 * alpha_out[...,0]).astype(np.uint8)

            fg_out = to_image(fg_pred[0,...])
            fg_out = fg_out * np.expand_dims((alpha_out.astype(float) / 255 > 0.01).astype(float), axis=2)
            fg_out = (255 * fg_out).astype(np.uint8)

            #Uncrop
            R0, C0 = _src_img.shape[0], _src_img.shape[1]
            uncrop_alpha_out = uncrop(alpha_out, bbox, R0, C0)
            uncrop_fg_out = uncrop(fg_out, bbox, R0, C0)

        #compose
        tgt_back = cv2.resize(tgt_back, (C0, R0))
        gs_back = cv2.resize(gs_back, (C0, R0))
        composited_img = composite4(uncrop_fg_out, tgt_back, uncrop_alpha_out)
        matte_img = composite4(uncrop_fg_out, gs_back, uncrop_alpha_out)

        # return output images
        uncrop_fg_out = cv2.cvtColor(uncrop_fg_out, cv2.COLOR_BGR2RGB)
        composited_img = cv2.cvtColor(composited_img, cv2.COLOR_BGR2RGB)
        matte_img = cv2.cvtColor(matte_img, cv2.COLOR_BGR2RGB)
        output_images = {
            "fg": uncrop_fg_out,
            "alpha": uncrop_alpha_out,
            "compose": composited_img,
            "matte": matte_img
        }

        return output_images


if __name__ == '__main__':
    bmv1 = BackgroundMattingV1(trained_model="real-hand-held")

    src_img_path = join("../sample_data/input", '00026_img.png')
    src_back_path = join("../sample_data/input", '00026_back.png')
    src_img_mask_path = join("../sample_data/input", '00026_masksDL.png')
    tgt_back_path = join("../sample_data/background/00026.png")

    output_images = bmv1(src_img_path, src_back_path, src_img_mask_path, tgt_back_path)

    cv2.imwrite(join("../sample_data/output", "_00026_compose.png"), output_images['compose'])
    cv2.imwrite(join("../sample_data/output", "_00026_matte.png"), output_images['matte'])
    cv2.imwrite(join("../sample_data/output", "_00026_fg.png"), output_images['fg'])
    cv2.imwrite(join("../sample_data/output", "_00026_out.png"), output_images['alpha'])

    import ipdb; ipdb.set_trace()

