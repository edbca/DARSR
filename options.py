import argparse
import torch
import os


class options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Network')

        # Paths
        self.parser.add_argument('--input_dir', '-i', type=str, default='LR', help='path to image input directory.')
        self.parser.add_argument('--output_dir', '-o', type=str, default='results' , help='path to image output directory.')
        self.parser.add_argument('--gt_dir', '-g', type=str, default='', help='path to grand-truth image.')
        # Sizes
        self.parser.add_argument('--input_crop_size', type=int, default=128, help='crop size for HR patch')
        self.parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
        self.parser.add_argument('--scale_factor', type=int, default=2, help='The upscaling scale factor')
        self.parser.add_argument('--scale_factor_downsampler', type=float, default=0.5, help='scale factor for downsampler')
        
        #Lambda Parameters
        self.parser.add_argument('--lambda_cycle', type=int, default=5, help='lambda parameter for cycle consistency loss')
        self.parser.add_argument('--lambda_interp', type=int, default=2, help='lambda parameter for masked interpolation loss')
        self.parser.add_argument('--lambda_regularization', type=int, default=2, help='lambda parameter for downsampler regularization term')
        self.parser.add_argument('--psnr_max', type=int, default=0, help='psnr')
        
        # Learning rates
        self.parser.add_argument('--lr_G_UP', type=float, default=0.001, help='initial learning rate for upsampler generator')
        self.parser.add_argument('--lr_G_DN', type=float, default=0.0002, help='initial learning rate for downsampler generator')
        self.parser.add_argument('--lr_D_DN', type=float, default=0.0002, help='initial learning rate for downsampler discriminator')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Adam momentum')
        
        # Iterations
        self.parser.add_argument('--num_iters', type=int, default=3000, help='number of training iterations')
        self.parser.add_argument('--eval_iters', type=int, default=10, help='for debug purpose')
        
        self.conf = self.parser.parse_args()
        
        if not os.path.exists(self.conf.output_dir):
            os.makedirs(self.conf.output_dir)
            

        
    def get_config(self, img_name):
        self.conf.abs_img_name = os.path.splitext(img_name)[0]
        self.conf.input_image_path = os.path.join(self.conf.input_dir, img_name)
        self.conf.gt_path = os.path.join(self.conf.gt_dir, img_name) if self.conf.gt_dir != '' else None
        
        print('*' * 60 + '\nRunning...')
        print('input image: \'%s\'' %self.conf.input_image_path)
        print('grand-truth image: \'%s\'' %self.conf.gt_path)
        return self.conf
    



