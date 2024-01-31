import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision.datasets.folder import default_loader

from .cp2tform import get_similarity_transform

class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
            __file__, super.__str__(self))

class WarpAffine:
    
    REFERENCE_FACIAL_POINTS = [
    [30.29459953,  51.69630051],
    [65.53179932,  51.50139999],
    [48.02519989,  71.73660278],
    [33.54930115,  92.3655014],
    [62.72990036,  92.20410156]
    ]

    DEFAULT_CROP_SIZE = (96, 112)
    
    @staticmethod
    def similarity(src_pts, dst_pts, reflective=True):
        trans, trans_inv = get_similarity_transform(src_pts, dst_pts, reflective=True)
        cv2_trans = trans[:, 0:2].T #cvt transform mat for cv2
        return cv2_trans
    
    @staticmethod
    def affine(src_pts, dst_pts):
        tfm = np.float32([ #default
            [1, 0, 1],
            [0, 1, 0],
        ])
        # define
        n_pts = src_pts.shape[0]
        ones = np.ones((n_pts, 1), src_pts.dtype)
        src_pts_ = np.hstack([src_pts, ones])
        dst_pts_ = np.hstack([dst_pts, ones])
        # linalg solve
        A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)
        if rank == 3:
            tfm = np.float32([
                [A[0, 0], A[1, 0], A[2, 0]],
                [A[0, 1], A[1, 1], A[2, 1]]
            ])
        elif rank == 2:
            tfm = np.float32([
                [A[0, 0], A[1, 0], 0],
                [A[0, 1], A[1, 1], 0]
            ])
        return tfm

    @staticmethod
    def get_reference_facial_points(output_size=None,
                                    inner_padding_factor=0.0,
                                    outer_padding=(0, 0),
                                    default_square=False):
        """
        Function:
        ----------
            get reference 5 key points according to crop settings:
            0. Set default crop_size:
                if default_square: 
                    crop_size = (112, 112)
                else: 
                    crop_size = (96, 112)
            1. Pad the crop_size by inner_padding_factor in each side;
            2. Resize crop_size into (output_size - outer_padding*2),
                pad into output_size with outer_padding;
            3. Output reference_5point;
        Parameters:
        ----------
            @output_size: (w, h) or None
                size of aligned face image
            @inner_padding_factor: (w_factor, h_factor)
                padding factor for inner (w, h)
            @outer_padding: (w_pad, h_pad)
                each row is a pair of coordinates (x, y)
            @default_square: True or False
                if True:
                    default crop_size = (112, 112)
                else:
                    default crop_size = (96, 112);
            !!! make sure, if output_size is not None:
                    (output_size - outer_padding) 
                    = some_scale * (default crop_size * (1.0 + inner_padding_factor))
        Returns:
        ----------
            @reference_5point: 5x2 np.array
                each row is a pair of transformed coordinates (x, y)
        """
        #print('\n===> get_reference_facial_points():')

        #print('---> Params:')
        #print('            output_size: ', output_size)
        #print('            inner_padding_factor: ', inner_padding_factor)
        #print('            outer_padding:', outer_padding)
        #print('            default_square: ', default_square)

        tmp_5pts = np.array(WarpAffine.REFERENCE_FACIAL_POINTS)
        tmp_crop_size = np.array(WarpAffine.DEFAULT_CROP_SIZE)

        # 0) make the inner region a square
        if default_square:
            size_diff = max(tmp_crop_size) - tmp_crop_size
            tmp_5pts += size_diff / 2
            tmp_crop_size += size_diff

        #print('---> default:')
        #print('              crop_size = ', tmp_crop_size)
        #print('              reference_5pts = ', tmp_5pts)

        if (output_size and
                output_size[0] == tmp_crop_size[0] and
                output_size[1] == tmp_crop_size[1]):
            #print('output_size == DEFAULT_CROP_SIZE {}: return default reference points'.format(tmp_crop_size))
            return tmp_5pts

        if (inner_padding_factor == 0 and
                outer_padding == (0, 0)):
            if output_size is None:
                #print('No paddings to do: return default reference points')
                return tmp_5pts
            else:
                raise FaceWarpException(
                    'No paddings to do, output_size must be None or {}'.format(tmp_crop_size))

        # check output size
        if not (0 <= inner_padding_factor <= 1.0):
            raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')

        if ((inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0)
                and output_size is None):
            output_size = tmp_crop_size * \
                (1 + inner_padding_factor * 2).astype(np.int32)
            output_size += np.array(outer_padding)
            #print('              deduced from paddings, output_size = ', output_size)

        if not (outer_padding[0] < output_size[0]
                and outer_padding[1] < output_size[1]):
            raise FaceWarpException('Not (outer_padding[0] < output_size[0]'
                                    'and outer_padding[1] < output_size[1])')

        # 1) pad the inner region according inner_padding_factor
        #print('---> STEP1: pad the inner region according inner_padding_factor')
        if inner_padding_factor > 0:
            size_diff = tmp_crop_size * inner_padding_factor * 2
            tmp_5pts += size_diff / 2
            tmp_crop_size += np.round(size_diff).astype(np.int32)

        #print('              crop_size = ', tmp_crop_size)
        #print('              reference_5pts = ', tmp_5pts)

        # 2) resize the padded inner region
        #print('---> STEP2: resize the padded inner region')
        size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
        #print('              crop_size = ', tmp_crop_size)
        #print('              size_bf_outer_pad = ', size_bf_outer_pad)

        if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
            raise FaceWarpException('Must have (output_size - outer_padding)'
                                    '= some_scale * (crop_size * (1.0 + inner_padding_factor)')

        scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
        #print('              resize scale_factor = ', scale_factor)
        tmp_5pts = tmp_5pts * scale_factor
#        size_diff = tmp_crop_size * (scale_factor - min(scale_factor))
#        tmp_5pts = tmp_5pts + size_diff / 2
        tmp_crop_size = size_bf_outer_pad
        #print('              crop_size = ', tmp_crop_size)
        #print('              reference_5pts = ', tmp_5pts)

        # 3) add outer_padding to make output_size
        reference_5point = tmp_5pts + np.array(outer_padding)
        tmp_crop_size = output_size
        #print('---> STEP3: add outer_padding to make output_size')
        #print('              crop_size = ', tmp_crop_size)
        #print('              reference_5pts = ', tmp_5pts)

        #print('===> end get_reference_facial_points\n')

        return reference_5point

def k2landmark(src_landmark, to_numpy=True):
    k2_landmark = [[src_landmark[i], src_landmark[i+1]]
                   for i in range(0, len(src_landmark), 2)]
    if to_numpy:
        k2_landmark = np.array(k2_landmark)
    return k2_landmark

class VggFace2(data.Dataset):
    
    BASE = Path(osp.join(osp.split(__file__)[0]))
    
    def __init__(self, data_type, 
                       transform=None,
                       mtcnn=None, 
                       get_standard_bb=False, 
                       img_aligen=None,
                       crop_size=None,
                       **kwargs) -> None:
        super().__init__()
        loader = kwargs.get("loader", default_loader)
        if img_aligen is not None and crop_size is None:
            TypeError("Required argument 'crop_size' not found.")
        flag = int(data_type == "train")
        self.root = VggFace2.BASE/"data"/data_type
        #get classes
        with open(VggFace2.BASE/"meta"/"clear_identity_meta.csv", 'r') as identity_file:
            identity_csv = pd.read_csv(identity_file)
            identity_csv = identity_csv[identity_csv[' Flag'] == flag]
            self.label_to_classid = sorted(identity_csv['Class_ID'].tolist())
        self.classid_to_label = {self.label_to_classid[i]:i 
                                 for i in range(len(self.label_to_classid))}
        #get image info
        with open(VggFace2.BASE/"data"/(data_type + "_list.txt"), 'r') as pathfile:
            self.samples = pathfile.read().strip('\n').split('\n')
        with open(VggFace2.BASE/"meta"/"bb_landmark"/"loose_bb_{}.csv".format(data_type), 'r') as bb_file:
            samples_bb = pd.read_csv(bb_file).drop('NAME_ID', axis=1)
            self.samples_bb = samples_bb.values.tolist()
        with open(VggFace2.BASE/"meta"/"bb_landmark"/"loose_landmark_{}.csv".format(data_type), 'r') as landmark_file:
            samples_landmark = pd.read_csv(landmark_file).drop('NAME_ID', axis=1)
            self.samples_landmark = samples_landmark.values.tolist()
        #transform
        self.require_bb = get_standard_bb
        self.align_type = img_aligen
        self.transform = transform
        self.mtcnn = mtcnn
        if img_aligen is not None:
            self.crop_size = tuple(crop_size)
            if crop_size[0] == 96 and crop_size[1] == 112:
                self.reference_pts = WarpAffine.REFERENCE_FACIAL_POINTS
            else:
                default_square = True
                inner_padding_factor = 0
                outer_padding = (0, 0)
                output_size = crop_size
                self.reference_pts = WarpAffine.get_reference_facial_points(output_size,
                                                                            inner_padding_factor,
                                                                            outer_padding,
                                                                            default_square)
    
    def __getitem__(self, index):
        imgpath = self.samples[index]
        cid = osp.split(imgpath)[0].strip('/')
        label = self.classid_to_label[cid]
        img = self.loader(self.root/imgpath)
        if self.align_type is not None:
            img_landmark = self.samples_landmark[index]
            affine_grid = self.align_type(k2landmark(img_landmark), self.reference_pts)
            img = cv2.warpAffine(np.array(img), affine_grid, self.crop_size)
            img = Image.fromarray(img.astype(np.uint8))
        if self.mtcnn is not None:
            img = self.mtcnn(img)
            if img is None:
                return self[index+1]
        if self.transform is not None:
            img = self.transform(img)
        if self.require_bb:
            img_bb = self.samples_bb[index]
            return img, label, img_bb
        else:
            return img, label
    
    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    from torchvision import transforms
    a = VggFace2('train', transform=transforms.ToPILImage(), img_aligen=WarpAffine.similarity, crop_size=(160, 160))
    print(a.label_to_classid)