import numpy as np
from torch.utils.data import Dataset, DataLoader
from imresize import imresize
from torchvision import transforms as T 
import random
import torch
import os  
import imageio
from util import read_image, create_gradient_map, im2tensor, create_probability_map, nn_interpolation

def create_dataset(conf):
    #dataset1 = DataGenerator(conf,conf.batch_size1)
    #dataset = DataGenerator(conf,conf.batch_size)
    dataset = DataGenerator(conf)
    #dataloader_one = DataLoader(dataset1, batch_size=conf.batch_size1, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False)

    return dataloader


class ToTensor(object):
    def __call__(self,sample):
        g_in = sample['HR']
        g_bq = sample['HR_bicubic']
        d_in = sample['LR']
        d_bq = sample['LR_up']

        g_in = np.ascontiguousarray(np.transpose(g_in,(2,0,1)))/255.0
        g_bq = np.ascontiguousarray(np.transpose(g_bq,(2,0,1)))/255.0
        d_in = np.ascontiguousarray(np.transpose(d_in,(2,0,1)))/255.0
        d_bq = np.ascontiguousarray(np.transpose(d_bq,(2,0,1)))/255.0

        return {"HR":torch.FloatTensor(g_in).cuda(),
                "HR_bicubic":torch.FloatTensor(g_bq).cuda(),
                "LR":torch.FloatTensor(d_in).cuda(),
                "LR_up":torch.FloatTensor(d_bq).cuda()}

class Flippe(object):
    def __call__(self,sample):
        '''
        sample:
            @'inputs':32 32 lr_img
            @'labels':128*128 hr_img 
        '''
        is_hor  = random.random()>0.5
    
        g_in = sample['HR']
        g_bq = sample['HR_bicubic']
        d_in = sample['LR']
        d_bq = sample['LR_up']
        #whether hor flip
        if is_hor:
            g_in = g_in[:,::-1,:]
            g_bq = g_bq[:,::-1,:]
            d_in = d_in[:,::-1,:]
            d_bq = d_bq[:,::-1,:]

        return {'HR':g_in,'HR_bicubic':g_bq, 'LR':d_in,'LR_up':d_bq}


class Rotation(object):
    def __call__(self,sample):
        is_rot = random.random()>0.5

        g_in = sample['HR']
        g_bq = sample['HR_bicubic']
        d_in = sample['LR']
        d_bq = sample['LR_up']

        if is_rot:
            g_in = np.transpose(g_in,(1,0,2))
            g_bq = np.transpose(g_bq,(1,0,2))
            d_in = np.transpose(d_in,(1,0,2))
            d_bq = np.transpose(d_bq,(1,0,2))

        return {'HR':g_in,'HR_bicubic':g_bq, 'LR':d_in,'LR_up':d_bq}


class DataGenerator(Dataset):
    """
    The data generator loads an image once, calculates it's gradient map on initialization and then outputs a cropped version
    of that image whenever called.
    """

    #def __init__(self, conf, batch_size):
    def __init__(self, conf):
        np.random.seed(0)
        self.conf = conf
        
        print('*' * 60 + '\nPreparing data ...')
        
        # Default shapes
        self.g_input_shape = conf.input_crop_size
        self.d_input_shape = int(conf.input_crop_size * conf.scale_factor_downsampler)
        self.transforms = T.Compose([Flippe(),Rotation(),ToTensor()])
        # Read input image
        #self.input_image = read_image(conf.input_image_path) / 255.
        self.input_image = imageio.imread(conf.input_image_path)
        self.shave_edges(scale_factor=conf.scale_factor_downsampler, real_image=False)

        self.in_rows, self.in_cols = self.input_image.shape[0:2]

        # Create prob map for choosing the crop
        self.crop_indices_for_g, self.crop_indices_for_d = self.make_list_of_crop_indices(conf=conf)
        
    def __len__(self):
        return self.conf.num_iters * self.conf.batch_size

    def __getitem__(self, idx):
        """Get a crop for both G and D """
        g_in = self.next_crop(for_g=True, idx=idx)##HR
        d_in = self.next_crop(for_g=False, idx=idx)##LR
        d_bq = imresize(im=d_in, scale_factor=int(1/self.conf.scale_factor_downsampler), kernel='cubic')#Up LR
        g_bq = imresize(im=g_in, scale_factor=self.conf.scale_factor_downsampler, kernel='cubic')#down LR

        sample = {'HR':g_in,'HR_bicubic':g_bq, 'LR':d_in,'LR_up':d_bq}
        sample = self.transforms(sample)
        return sample
        #return {'HR':im2tensor(g_in).squeeze(),'HR_bicubic':im2tensor(g_bq).squeeze(), 'LR':im2tensor(d_in).squeeze(),'LR_up':im2tensor(d_bq).squeeze() }
    def next_crop(self, for_g, idx):
        """Return a crop according to the pre-determined list of indices. Noise is added to crops for D"""
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_top_left(size, for_g, idx)
        #top = np.random.randint(0, self.in_rows - size)
        #left = np.random.randint(0, self.in_cols - size)
        crop_im = self.input_image[top:top + size, left:left + size, :]
        #if not for_g:  # Add noise to the image for d
        #    crop_im += np.random.randn(*crop_im.shape) / 255.0
        return crop_im

    def make_list_of_crop_indices(self, conf):
        iterations = conf.num_iters * conf.batch_size
        prob_map_big, prob_map_sml = self.create_prob_maps(scale_factor=conf.scale_factor_downsampler)
        crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=iterations, p=prob_map_sml)
        crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        return crop_indices_for_g, crop_indices_for_d

    def create_prob_maps(self, scale_factor):
        # Create loss maps for input image and downscaled one
        loss_map_big = create_gradient_map(self.input_image)
        loss_map_sml = create_gradient_map(imresize(im=self.input_image, scale_factor=scale_factor, kernel='cubic'))
        # Create corresponding probability maps
        prob_map_big = create_probability_map(loss_map_big, self.d_input_shape)
        prob_map_sml = create_probability_map(nn_interpolation(loss_map_sml, int(1 / scale_factor)), self.g_input_shape)
        return prob_map_big, prob_map_sml

    def shave_edges(self, scale_factor, real_image):
        """Shave pixels from edges to avoid code-bugs"""
        # Crop 10 pixels to avoid boundaries effects in synthetically generated examples
        if not real_image:
            self.input_image = self.input_image[10:-10, 10:-10, :]
        # Crop pixels for the shape to be divisible by the scale factor
        sf = int(1 / scale_factor)
        shape = self.input_image.shape
        self.input_image = self.input_image[:-(shape[0] % sf), :, :] if shape[0] % sf > 0 else self.input_image
        self.input_image = self.input_image[:, :-(shape[1] % sf), :] if shape[1] % sf > 0 else self.input_image

    def get_top_left(self, size, for_g, idx):
        """Translate the center of the index of the crop to it's corresponding top-left"""
        center = self.crop_indices_for_g[idx] if for_g else self.crop_indices_for_d[idx]
        row, col = int(center / self.in_cols), center % self.in_cols
        top, left = min(max(0, row - size // 2), self.in_rows - size), min(max(0, col - size // 2), self.in_cols - size)
        # Choose even indices (to avoid misalignment with the loss map for_g)
        return top - top % 2, left - left % 2
        