import os
import tqdm
import torch
import modules
from options import options
from data import create_dataset
from network import Network
from learner import Learner


def train_and_eval(conf):

    #dataloader_one, dataloader = create_dataset(conf) 
    dataloader = create_dataset(conf) 
    #Degradation Score Estimation Module
    model = modules.DSEM().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))     
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    #model.apply(modules.default_init_weights)  
    print('*' * 60 + '\nTraining DSEM...')
    iter = 0 
    for iteration, data in enumerate(tqdm.tqdm(dataloader)):
        LR_uk = data['LR']
        LR_bi = data['HR_bicubic']
        optimizer.zero_grad()     
        # Forward path
        score_zero = model(LR_bi)
        score_one = model(LR_uk)
        if iter==2980: 
            print(score_zero)
            print(score_one)                    
        # Losses
        zero = torch.zeros(score_zero.size(),device=torch.device('cuda'))
        one = torch.ones(score_one.size(),device=torch.device('cuda'))
        loss_zero = torch.mean((score_zero - zero)**2)
        loss_one = torch.mean((score_one - one)**2) 
        total_loss = loss_zero + loss_one 
        total_loss.backward()        
        optimizer.step()  
        lr_scheduler.step()
        iter = iter + 1      
    save_model(model)

    ####SR
    if conf.gt_path is not None:
        for num in range(1,5):
            model_sr = Network(conf)
            learner = Learner(model_sr)
            print('*' * 60 + '\nTraining SR ...')
            for iteration, data in enumerate(tqdm.tqdm(dataloader)):
                conf.psnr_max = model_sr.train(data)
                learner.update(iteration, model_sr)
                if iteration == 1001:
                    print('best PSNR =',conf.psnr_max) 
                    break
    else:
        model_sr = Network(conf)
        learner = Learner(model_sr)
        print('*' * 60 + '\nTraining SR ...')
        for iteration, data in enumerate(tqdm.tqdm(dataloader)):
            model_sr.train(data)
            learner.update(iteration, model_sr)
        model_sr.eval(conf)

def save_model(model):
    model_out_path = "DSEM/" + "DSEM.pth"
    if not os.path.exists("DSEM/"):
           os.makedirs("DSEM/")
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))   


def main():
    opt = options()
    # Run Network on all images in the input directory
    for img_name in os.listdir(opt.conf.input_dir):
        conf = opt.get_config(img_name)
        conf.psnr_max = 0
        train_and_eval(conf)
    


if __name__ == '__main__':
    main()
