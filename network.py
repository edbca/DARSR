import torch
import loss
import modules
import util
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch.nn.functional as F 
from scipy.io import loadmat
import os
import imageio
from torch_sobel import Sobel

from SRFBN.srfbn_arch import SRFBN
class Network:
  
    def __init__(self, conf):
        # Fix random seed
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True # slightly reduces throughput
    
        # Acquire configuration
        self.conf = conf
        # Define the modules
        self.G = modules.Generator_G(self.conf.scale_factor).cuda()
        self.D = modules.Discriminator_D().cuda()
        self.DARM = modules.DARM().cuda()
        #self.G_F = modules.G_F().cuda()
        
        # Losses
        self.criterion_gan = loss.GANLoss().cuda()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_interp = torch.nn.L1Loss()

        self.regularization = loss.DownsamplerRegularization(conf.scale_factor_downsampler, self.G.G_kernel_size)

        # Initialize modules weights
        
        self.G.apply(modules.weights_init_G)
        self.D.apply(modules.weights_init_D)
        self.DARM.apply(modules.weights_init_DARM)
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=conf.lr_G_DN, betas=(conf.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=conf.lr_D_DN, betas=(conf.beta1, 0.999))
        self.optimizer_DARM = torch.optim.Adam(self.DARM.parameters(), lr=conf.lr_G_UP, betas=(conf.beta1, 0.999))
    

        self.in_img = util.read_image(conf.input_image_path)
        self.in_img_t= util.im2tensor(self.in_img)
        self.gt_img = util.read_image(conf.gt_path) if conf.gt_path is not None else None
        
        self.gpu = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.gpu)
        # Define DBPN Super Resolver
        self.SR_model = SRFBN(in_channels=3, out_channels=3, num_features=32, num_steps=4, num_groups=3, upscale_factor=self.conf.scale_factor, act_type = 'prelu', norm_type = None).cuda()
        self.SR_model = torch.nn.DataParallel(self.SR_model, device_ids=[self.gpu], output_device=self.device)
        state_dict = torch.load('SRFBN/SRFBN-S_x%s_BI.pth'%self.conf.scale_factor, map_location=lambda storage, loc:storage.cuda())
        self.SR_model.load_state_dict(state_dict)
        self.SR_model = self.SR_model.module
        self.SR_model = self.SR_model.eval()
        
        self.DSEM = modules.DSEM().cuda()
        state_dict = torch.load('DSEM/DSEM.pth',map_location=lambda storage, loc:storage.cuda())
        self.DSEM.load_state_dict(state_dict)
        self.DSEM = self.DSEM.eval()
                
        self.iter = 0
        self.psnr_max=conf.psnr_max if self.gt_img is not None else None

        
    def train(self, data):
        self.set_input(data)
        self.train_G()
        self.train_D()

        if self.gt_img is not None:
            if self.iter == 0 or (self.iter>199 and self.iter % self.conf.eval_iters == 0):
                self.quick_eval()
        self.iter = self.iter + 1


        return self.psnr_max if self.gt_img is not None else None
        
    
    
    def set_input(self, data):
        self.real_HR = data['HR']
        self.real_HR_bicubic = data['HR_bicubic']
        self.real_LR = data['LR']
        self.real_LR_up = data['LR_up']
        
    
    
    def train_G(self):
        # Turn off gradient calculation for discriminator
        util.set_requires_grad([self.D], False)       
        
        # Rese gradient valus
        self.optimizer_DARM.zero_grad()
        self.optimizer_G.zero_grad()

        #HR-flow
        self.fake_LR = self.G(self.real_HR)# 58
        self.rec_HR_bicubic = self.DARM(self.fake_LR) #58
        self.sr_image = self.SR_model(self.rec_HR_bicubic*255.)[-1]/255.


        #LR-flow
        self.adapt_LR = self.DARM(self.real_LR) #64
        self.LR_SR = self.SR_model(self.adapt_LR*255.)[-1]/255.
        self.rec_LR = self.G(self.LR_SR)#58

        
        #anchor
        self.score_LR_bi = self.DSEM(util.shave_a2b(self.real_HR_bicubic, self.rec_HR_bicubic))##64-->58
        self.score_LR = self.DSEM(util.shave_a2b(self.real_LR, self.rec_LR))##64-->58

        ##HR-flow
        self.score_fake_LR = self.DSEM(self.fake_LR)##1 58x58
        self.score_rec_HR_bicubic = self.DSEM(self.rec_HR_bicubic)## 58x58
        
        ##LR-flow
        self.score_adapt_LR = self.DSEM(util.shave_a2b(self.adapt_LR, self.fake_LR))# 64x64 util.shave_a2b(self.adapt_LR, self.fake_LR)
        self.score_rec_LR = self.DSEM(self.rec_LR)#1 58x58

        # Losses
        self.loss_GAN = self.criterion_gan(self.D(self.fake_LR), True) 

        self.loss_cycle_forward = self.criterion_cycle(self.rec_HR_bicubic, util.shave_a2b(self.real_HR_bicubic, self.rec_HR_bicubic)) * self.conf.lambda_cycle + self.criterion_cycle(self.sr_image, util.shave_a2b(self.real_HR, self.sr_image)) * self.conf.lambda_cycle
    
        self.loss_cycle_backward = self.criterion_cycle(self.rec_LR, util.shave_a2b(self.real_LR, self.rec_LR)) * self.conf.lambda_cycle
        

        self.loss_zero = torch.mean((self.score_rec_HR_bicubic - self.score_LR_bi)**2) + torch.mean((self.score_adapt_LR - self.score_LR_bi)**2)
        self.loss_one = torch.mean((self.score_fake_LR - self.score_LR)**2) + torch.mean((self.score_rec_LR - self.score_LR)**2)
        

        sobel_A = Sobel()(self.real_LR_up.detach())
        loss_map_A = 1 - torch.clamp(sobel_A, 0, 1)
        self.loss_interp = self.criterion_interp(self.LR_SR * loss_map_A, self.real_LR_up * loss_map_A) * self.conf.lambda_interp 

        self.curr_k = util.calc_curr_k(self.G.parameters())
        
        self.loss_regularization = self.regularization(self.curr_k, self.real_HR, self.fake_LR) * self.conf.lambda_regularization 
              
        self.total_loss = self.loss_GAN + self.loss_cycle_forward + self.loss_cycle_backward + self.loss_interp + self.loss_regularization + self.loss_zero + self.loss_one
        self.total_loss.backward()
        
        self.optimizer_DARM.step()
        self.optimizer_G.step()

        
    def train_D(self):
        # Turn on gradient calculation for discriminator
        util.set_requires_grad([self.D], True)
        
        # Rese gradient valus
        self.optimizer_D.zero_grad()
               
        # Fake
        pred_fake = self.D(self.fake_LR.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)
        # Real
        pred_real = self.D(util.shave_a2b(self.real_LR, self.fake_LR))
        loss_D_real = self.criterion_gan(pred_real, True)

        self.loss_Discriminator = (loss_D_real + loss_D_fake) * 0.5 
        self.loss_Discriminator.backward()

        # Update weights
        self.optimizer_D.step()
       
               
    def eval(self,conf):
        # Read input image
        self.quick_eval()  
        plt.imsave(os.path.join(self.conf.output_dir, '%sx%s_corrected.png' %(self.conf.abs_img_name,self.conf.scale_factor)), self.corrected_img)
        plt.imsave(os.path.join(self.conf.output_dir, '%sx%s_ours.png' %(self.conf.abs_img_name,self.conf.scale_factor)), self.SR_img)        
        print('*' * 60 + '\nOutput is saved in \'%s\' folder\n' % self.conf.output_dir)
    
    def quick_eval(self):
        # Evaluate trained upsampler and downsampler on input data
        with torch.no_grad():
            corrected_img = self.DARM(self.in_img_t)
            SR_image = self.SR_model(corrected_img*255.)[-1]/255.
            
            if self.gt_img is None:
               SRFBN = self.SR_model(self.in_img_t*255.)[-1]/255.
               self.SRFBN = util.tensor2im(SRFBN)
               print('*' * 60)
               plt.imsave(os.path.join(self.conf.output_dir, '%sx%s_SRFBNs.png' %(self.conf.abs_img_name,self.conf.scale_factor)), self.SRFBN)            

            if self.iter==0 and self.gt_img is not None:
               SRFBN = self.SR_model(self.in_img_t*255.)[-1]/255.
               self.SRFBN = util.tensor2im(SRFBN)
               plt.imsave(os.path.join(self.conf.output_dir, '%sx%s_SRFBNs.png' %(self.conf.abs_img_name,self.conf.scale_factor)), self.SRFBN)
               psnr = util.cal_y_psnr(self.SRFBN, self.gt_img, border=self.conf.scale_factor)
               print('SRFBNs PSNR =',psnr)
        
        self.SR_img = util.tensor2im(SR_image)
        self.corrected_img = util.tensor2im(corrected_img)

        if self.gt_img is not None:
            psnr = util.cal_y_psnr(self.SR_img, self.gt_img, border=self.conf.scale_factor)
            if psnr>self.psnr_max:
                self.psnr_max=psnr
                plt.imsave(os.path.join(self.conf.output_dir, '%sx%s_corrected.png' %(self.conf.abs_img_name,self.conf.scale_factor)), self.corrected_img)
                plt.imsave(os.path.join(self.conf.output_dir, '%sx%s_ours.png' %(self.conf.abs_img_name,self.conf.scale_factor)), self.SR_img)
                print('best PSNR =',psnr) 
             


        
