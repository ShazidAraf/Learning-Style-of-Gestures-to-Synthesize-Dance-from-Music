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



class PoseMusicDataset_new(Dataset):

    def __init__(self, bundle_len = 1000, seq_len = 100, init_step = 0,  prev_poses_cnt = 5, training =1):
        self.prev_poses_cnt = prev_poses_cnt
        
        self.last_n_poses = None
        
        #self.max_raw_sample = 175000
        self.seq_len = seq_len
        self.bundle_len = bundle_len
        if training==1:
            self.data_files_path = "data/train"
            newlist = [f for f in os.listdir('data/train') if f.endswith('.json')]
#             print(len(newlist))
            self.seq_samples_cnt = int(bundle_len*len(newlist)/seq_len)
        else:
            self.data_files_path = "data/valid"
            newlist = [f for f in os.listdir('data/valid') if f.endswith('.json')]
#             print(len(newlist))
            self.seq_samples_cnt = int(bundle_len*len(newlist)/seq_len)
        
         #1750 
        # because max_raw_sample/seq_len = 1750 ie., each bundle has 10 i/o for this model as seq_len is 100
        
        self.seq_samples_ids_stack = []
        self.seq_samples_db = dict()
        self.max_db_size = 50 
        # better to have multiples of 10
#         print(self.seq_samples_cnt)
    def __len__(self):
        return self.seq_samples_cnt

    def __getitem__(self, idx):
        bundle_min = int(math.floor(idx/10))*self.bundle_len
        index_in_bundle = int(idx%10)*self.seq_len
        sample_key = "{0}_{1}".format(bundle_min, index_in_bundle)
        
        #check if in db
        if sample_key in self.seq_samples_db:
            return self.seq_samples_db[sample_key]        
        
        if len(self.seq_samples_db) >= self.max_db_size:            
            for index in range(10):
                del(self.seq_samples_db[self.seq_samples_ids_stack[index]])
            self.seq_samples_ids_stack = np.array(self.seq_samples_ids_stack)[10:].tolist()
            p_time = time.time()
        
        #add corresponding index file samples, these will be 10 always
        with open(self.data_files_path+"/bundle_{0:09d}_{1:09d}.json".format(bundle_min, bundle_min+self.bundle_len)) as f:
            p_time = time.time()
            file_data = json.load(f)
            input_audio_spect = np.array(file_data['input_audio_spect'])
            output_pose_point = np.array(file_data['output_pose_point'])
            del(file_data)
            for index in range(10):
                cur_index_in_bundle = index*self.seq_len
                audio_inputs = input_audio_spect[cur_index_in_bundle:cur_index_in_bundle+self.seq_len]
                audio_inputs = np.expand_dims(audio_inputs, 1)
                next_steps = output_pose_point[cur_index_in_bundle:cur_index_in_bundle+self.seq_len]
                
                if self.last_n_poses is None:
                    prev_poses_input = np.zeros((self.prev_poses_cnt, next_steps.shape[-1]))#, dtype=np.float32)
                    prev_poses_target = np.zeros((self.prev_poses_cnt, next_steps.shape[-1]))#, dtype=np.float32)
                else:
                    prev_poses_input = self.last_n_poses[:-1]
                    prev_poses_target = self.last_n_poses[1:]
                                
                sample = {
                    'audio_inputs': audio_inputs,
                    'prev_poses': prev_poses_input,  #np.zeros(34, dtype=np.float32),
                    'next_steps': np.concatenate((prev_poses_target, next_steps), 0)
                }
                cur_sample_key = "{0}_{1}".format(bundle_min, cur_index_in_bundle)
                self.seq_samples_ids_stack.append(cur_sample_key)
                self.seq_samples_db[cur_sample_key] = sample
                self.last_n_poses = sample['next_steps'][-(self.prev_poses_cnt+1):]
        
        return self.seq_samples_db[sample_key]




class CNNFeat(torch.nn.Module):
    def __init__(self, dim):
        super(CNNFeat, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(129, 2))
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(129, 2))
        self.conv3 = torch.nn.Conv2d(16, 24, kernel_size=(129, 2))
        self.conv4 = torch.nn.Conv2d(24, dim, kernel_size=(129, 2))
        self.cvbn1 = torch.nn.BatchNorm2d(8)
        self.cvbn2 = torch.nn.BatchNorm2d(16)
        self.cvbn3 = torch.nn.BatchNorm2d(24)
        self.cvbn4 = torch.nn.BatchNorm2d(dim)
        
    def forward(self, h):
        h = F.relu(self.cvbn1(self.conv1(h)))
        h = F.relu(self.cvbn2(self.conv2(h)))
        h = F.relu(self.cvbn3(self.conv3(h)))
        h = F.relu(self.cvbn4(self.conv4(h)))
        return h.view((h.size(0), -1))


# In[9]:


class MDNRNN(torch.nn.Module):
    def __init__(self, dim, cnnEncoder, z_size, n_hidden=256, n_gaussians=5, n_layers=1):
        super(MDNRNN, self).__init__()
        
        self.z_size = z_size
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers
        
        self.lstm = torch.nn.LSTM(dim, n_hidden, n_layers, batch_first=True)
        self.prev_steps_fc = torch.nn.Linear(z_size, dim)
        self.audiofeat = cnnEncoder(dim)        
        self.fc1 = torch.nn.Linear(n_hidden, n_gaussians)#*z_size)
        self.fc2 = torch.nn.Linear(n_hidden, n_gaussians*z_size)
        self.fc3 = torch.nn.Linear(n_hidden, n_gaussians)#*z_size)
        
    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)
        
        pi = pi.view(-1, rollout_length, self.n_gaussians)
        mu = mu.view(-1, rollout_length, self.z_size, self.n_gaussians)
        sigma = sigma.view(-1, rollout_length, self.n_gaussians)#, self.z_size)
        
        pi = F.softmax(torch.clamp(pi, 1e-8, 1.), -1)
        
        sigma = F.elu(sigma)+1.+1e-8
        return pi, mu, sigma
        
        
    def forward(self, audio_inputs, prev_steps, h):
        # Forward propagate LSTM
        x = []
        
        for i, input_t in enumerate(prev_steps.chunk(prev_steps.size(1), dim=1)):
            p_steps = self.prev_steps_fc(input_t)
            x += [p_steps.view((p_steps.size(0), -1))]
            
        for i, input_t in enumerate(audio_inputs.chunk(audio_inputs.size(1), dim=1)):
            input_t = input_t[:,0]
            h_ = self.audiofeat(input_t)
            x += [h_]
        
        x = torch.stack(x, 1).squeeze(2)
        y, (h, c) = self.lstm(x, h)
        pi, mu, sigma = self.get_mixture_coef(y)
        return (pi, mu, sigma), (h, c)
    
    def init_hidden(self, bsz):
        return (torch.zeros(self.n_layers, bsz, self.n_hidden).to(device),
                torch.zeros(self.n_layers, bsz, self.n_hidden).to(device))

class Nelson_model(torch.nn.Module):
    def __init__(self, dim, cnnEncoder, z_size, n_hidden=256, n_gaussians=5, n_layers=1):
        super(Nelson_model, self).__init__()
        
        self.z_size = z_size
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers


        self.audiofeat = cnnEncoder(dim)
        self.lstm = torch.nn.LSTM(dim, n_hidden, n_layers, batch_first=True)
        self.prev_steps_fc = torch.nn.Linear(z_size, dim)
        self.fc_out = torch.nn.Linear(n_hidden, z_size)#*z_size)

         
        
    def forward(self, audio_inputs, prev_steps, h):


    # Forward propagate LSTM
        x = []
        
        for i, input_t in enumerate(prev_steps.chunk(prev_steps.size(1), dim=1)):
            p_steps = self.prev_steps_fc(input_t)
            x += [p_steps.view((p_steps.size(0), -1))]
#         print('x:', len(x))
        
        for i, input_t in enumerate(audio_inputs.chunk(audio_inputs.size(1), dim=1)):
            input_t = input_t[:,0]
            h_ = self.audiofeat(input_t)
            x += [h_]
        
#         print('x:', len(x))
        # print(x[0].shape)
        x = torch.stack(x, 1).squeeze(2)
#         print(x.shape)
        
        x, (h, c) = self.lstm(x, h)
#         print('x_shape',x.shape)

        x = self.fc_out(x)
#         print('x_shape',x.shape)


        return x
    
    def init_hidden(self, bsz):
        return (torch.zeros(self.n_layers, bsz, self.n_hidden).to(device),
                torch.zeros(self.n_layers, bsz, self.n_hidden).to(device))





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detach(states):
    return [state.detach() for state in states] 



def log_sum_exp(x, dim=None):
    """Log-sum-exp trick implementation"""
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_log = torch.log(torch.sum(torch.exp(x - x_max), dim=dim, keepdim=True))
    return x_log+x_max
        
def mdn_loss_fn(y, pi, mu, sigma):    
    c = y.shape[-2]
    
    var = (sigma ** 2)
    log_scale = torch.log(sigma)    
    
    exponent = torch.log(pi) - .5 * float(c) * math.log(2 * math.pi)         - float(c) * log_scale         - torch.sum(((y - mu) ** 2), dim=2) / (2 * var)
    
    log_gauss = log_sum_exp(exponent, dim=2)
    res = - torch.mean(log_gauss)

    return res

def criterion(y, pi, mu, sigma):
    y = y.unsqueeze(3)
    return mdn_loss_fn(y, pi, mu, sigma)

def get_predicted_steps(pi, mu):
    pi = pi.cpu().detach().numpy()
    dim = pi.shape[2]
    z_next_pred = np.array([ [mu[i,seq,:,np.random.choice(dim,p=pi[i][seq])].cpu().detach().numpy() for seq in np.arange(pi.shape[1])] for i in np.arange(len(pi))])
    return z_next_pred


# In[11]:


def save_checkpoint(save_path, epoch, model, optimizer,save_per_iter = False):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    if save_per_iter == True:
        torch.save(state, "{0}/latest_epoch.pth.tar".format(save_path))
    else:
        torch.save(state, "{0}/epoch_{1}.pth.tar".format(save_path, epoch+1))

def load_checkpoint(model, optimizer, save_path):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(save_path):
        print("=> loading checkpoint '{}'".format(save_path))
        checkpoint = torch.load(save_path)#, map_location=lambda storage, loc: storage)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})" .format(save_path, checkpoint['epoch']))
        
        model = model.to(device)
        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return start_epoch, model, optimizer



def load_data(data,gpu_cnt):
    if gpu_cnt>0:
        audio_inputs = data['audio_inputs'].type(torch.cuda.DoubleTensor)
        prev_steps = data['prev_poses'].type(torch.cuda.DoubleTensor)
        next_steps = data['next_steps'].type(torch.cuda.DoubleTensor)    
    else:
        audio_inputs = data['audio_inputs'].type(torch.DoubleTensor)
        prev_steps = data['prev_poses'].type(torch.DoubleTensor)
        next_steps = data['next_steps'].type(torch.DoubleTensor)   
        
    return audio_inputs,prev_steps,next_steps




def compute_loss(audio_inputs,next_steps,model,hidden):

    # print('next_steps:',next_steps.shape)
    no_of_prev_steps = 5
    seq_len = next_steps.shape[1] - no_of_prev_steps
    prev_steps = next_steps[:,0:5,:]

    loss = 0
#     for i in range(seq_len-1):

#         prev_steps = next_steps[:,i:i+5,:]
#         print('audio_inputs:',audio_inputs.shape,'prev_steps:',prev_steps.shape,'next_steps:',next_steps.shape) 
    y= model.forward(audio_inputs, prev_steps, hidden)
    loss += F.mse_loss(next_steps, y)


    return  loss
