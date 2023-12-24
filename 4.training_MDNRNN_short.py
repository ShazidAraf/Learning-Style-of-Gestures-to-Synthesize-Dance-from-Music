# python 4.training_MDNRNN_short.py --resume=1
import torch

if torch.cuda.device_count()>0:
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import json
import sys
import numpy as np
import math
import time
import imageio

from tensorboardX import SummaryWriter

from custom_utils.datastft import single_spectrogram

import logging


from model_utils import PoseMusicDataset_new, CNNFeat, MDNRNN
from model_utils import save_checkpoint, load_checkpoint,load_data,criterion

import argparse
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--resume", required=True,help="Resume?")
args = vars(ap.parse_args())

resume = int(args["resume"])

seq_len = 100
prev_poses_cnt = 5
epochs = 200
batch_size = 10
print("Epochs to do:", epochs)

dset1 = PoseMusicDataset_new(1000, seq_len, 0, prev_poses_cnt,1)
dataloader1 = DataLoader(dset1, batch_size=batch_size,shuffle=False, num_workers=0)
print("Dataloader size:", dataloader1.__len__())
print("Dataset size:", dset1.__len__())

# dset2 = PoseMusicDataset_new(1000, seq_len, 0, prev_poses_cnt,0)
# dataloader2 = DataLoader(dset2, batch_size=batch_size,shuffle=False, num_workers=0)
# print("Dataloader size:", dataloader2.__len__())
# print("Dataset size:", dset2.__len__())


# In[5]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 


# In[6]:


output_dir = "output/motiondance_simplernn"
if not os.path.exists(output_dir+"/checkpoints"):
    os.makedirs(output_dir+"/checkpoints")
if not os.path.exists(output_dir+"/frozen"):
    os.makedirs(output_dir+"/frozen")

logging.basicConfig(format='%(asctime)s :%(message)s', 
                    level=logging.INFO,
                    filename='./{0}/{1}.log'.format(output_dir, 'info')
                    )
logging.info("Training Started:")
logging.info("----{0}----".format(0))
sys.stdout.write("created required directories for saving model\n")

writer = SummaryWriter(comment='mdn_simple_rnn_gpu_{0}'.format(epochs))





audio_convout_size = 28
z_size = 34 #output size
n_hidden = 256
n_gaussians = 5
n_layers = 2

gpu_cnt = torch.cuda.device_count()
if gpu_cnt == 1:
    sys.stdout.write("One GPU\n")
    model = MDNRNN(audio_convout_size, CNNFeat, z_size, n_hidden, n_gaussians, n_layers).cuda()
elif gpu_cnt > 1:
    sys.stdout.write("More GPU's: {0}\n".format(gpu_cnt))
    model = torch.nn.DataParellel( MDNRNN(audio_convout_size, CNNFeat, z_size, n_hidden, n_gaussians, n_layers).cuda() )
else:
    sys.stdout.write("No GPU\n")
    model = MDNRNN(audio_convout_size, CNNFeat, z_size, n_hidden, n_gaussians, n_layers)
    
model = model.double()
    
#criterion = torch.nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999), amsgrad=True)
optimizer = torch.optim.Adam(model.parameters())

g_p_time = time.time()
p_time = time.time()




# resume = 1
if resume ==1:
    model_saved_path = "output/motiondance_simplernn/checkpoints/latest_epoch.pth.tar"
    start_epoch, model, optimizer = load_checkpoint(model, optimizer, model_saved_path)
    projection_print_index = len(dataloader1)#dataloader.__len__()*5

else:
    start_epoch=0
    f= open("output/motiondance_simplernn/train.txt","w+")
    f= open("output/motiondance_simplernn/valid.txt","w+")






model = model.train()
for epoch in range(start_epoch,epochs):

    total_loss = 0.0
    hidden = model.init_hidden(batch_size)
    for i, data in enumerate(dataloader1):
        
        
        audio_inputs,prev_steps,next_steps = load_data(data,gpu_cnt)
        optimizer.zero_grad()
        
        hidden = detach(hidden)
        hidden = model.init_hidden(batch_size)
        
        
        (pi, mu, sigma), hidden = model(audio_inputs, prev_steps, hidden)        
        loss = criterion(next_steps, pi, mu, sigma)                
        
        loss.backward()
        optimizer.step()
                
        cur_loss = loss.cpu().detach().numpy()
        total_loss += cur_loss     
        
        n_iter = epoch*len(dataloader1) + i
        writer.add_scalar('cur_loss', cur_loss, n_iter+1)
        writer.add_scalar('loss', total_loss/(i+1), n_iter+1)
        
        sys.stdout.write('\r\r\r[{:8d}, {:3d}, {:5d}] tot_loss: {:12.6f} cur_loss: {:12.6f}  tot_time: {:17.4f}'.format(n_iter+1, epoch + 1, i + 1, total_loss/(i+1), cur_loss, (time.time()-g_p_time) ))
        logging.info('[{:8d}, {:3d}, {:5d}] tot_loss: {:12.6f} cur_loss: {:12.6f} tot_time: {:17.4f}'.format(n_iter+1, epoch + 1, i + 1, total_loss/(i+1), cur_loss, (time.time()-g_p_time) )) 
        
        
        
    n = len([f for f in os.listdir('data/train') if f.endswith('.json')])
    with open("output/motiondance_simplernn/train.txt", "a") as myfile:
        myfile.write(str(total_loss/n)+'\n')
        
        
    # total_loss = 0.0    
    # for i, data in enumerate(dataloader2):
        

    #     audio_inputs,prev_steps,next_steps = load_data(data,gpu_cnt)

    #     hidden = detach(hidden)
    #     hidden = model.init_hidden(batch_size)
        
    #     (pi, mu, sigma), hidden = model(audio_inputs, prev_steps, hidden)        
    #     loss = criterion(next_steps, pi, mu, sigma)      
        
        
    #     cur_loss = loss.cpu().detach().numpy()
    #     total_loss += cur_loss    
        
        
    # n = len([f for f in os.listdir('data/valid') if f.endswith('.json')])
    # sys.stdout.write('\r\r\r validation: tot_loss: {:12.6f}'.format(total_loss/n))
    # logging.info('validation:average tot_loss: {:12.6f}'.format(total_loss/n)) 
    
    # with open("output/motiondance_simplernn/valid.txt", "a") as myfile:
    #     myfile.write(str(total_loss/n)+'\n')
    
    
    
    logging.info('epoch {0:3d} finished'.format(epoch+1))
    sys.stdout.write('\nepoch {0:3d} finished\n'.format(epoch+1))
    writer.add_scalar('epoch_loss', total_loss/(i+1), epoch+1)
    save_checkpoint(output_dir+"/checkpoints", epoch, model, optimizer,save_per_iter=True)
    if(epoch+1)%10 == 0:
        save_checkpoint(output_dir+"/checkpoints", epoch, model, optimizer)
        sys.stdout.write("\ncheckpoint saved for epoch+1: {0}\n".format(epoch+1))
        logging.info("checkpoint saved for epoch+1: {0}".format(epoch+1))
                
sys.stdout.write("\n---- Finished processing ----\n")


# In[ ]:


final_freezed_path = output_dir+"/frozen/final_model_{0}.pt".format(epochs)
torch.save(model, final_freezed_path)

sys.stdout.write("---- Model Frozen ----\n")
logging.info("---- Model Frozen ----")

sys.stdout.write("\n---- DONE ----\n")
