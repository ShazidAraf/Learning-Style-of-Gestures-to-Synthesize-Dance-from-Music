from scipy.io import wavfile
import os
import sys
import math
import numpy as np
import json
import math
import time
import imageio

import IPython
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from custom_utils.datastft import single_spectrogram

if torch.cuda.device_count()>0:
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')


import argparse
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,help="Test data")
ap.add_argument("-m", "--model", required=True,help="Trained model path")
ap.add_argument("-n", "--saved_name", required=True,help="saved_name")

args = vars(ap.parse_args())
 

from model_utils import PoseMusicDataset_new, CNNFeat, MDNRNN, Nelson_model
from model_utils import save_checkpoint, load_checkpoint,load_data,criterion,compute_loss, get_predicted_steps




bundle_min = 0
bundle_len = 1000
test_list = [f for f in os.listdir('data/{0}'.format(args["data"])) if f.endswith('.json')]
sample_inputs = []
next_steps_gt = []
for i in range(len(test_list)):
    
    with open('data/{0}'.format(args["data"])+"/bundle_{0:09d}_{1:09d}.json".format(i*bundle_len, (i+1)*bundle_len)) as f:

            file_data = json.load(f)
            
            input_audio_spect = np.array(file_data['input_audio_spect'])
            sample_inputs.append(input_audio_spect)
            
            output_pose_points = np.array(file_data['output_pose_point'])
            
            next_steps_gt.append(output_pose_points)
            
            del(file_data)
            
sample_inputs = np.reshape(sample_inputs,(-1,513,5))
next_steps_gt = np.reshape(next_steps_gt,(-1,17,2))
sample_inputs = np.array(sample_inputs)
sample_inputs = np.expand_dims(sample_inputs, 1) # for input add channel
sample_inputs = np.expand_dims(sample_inputs, 1) # make number of sequences as 1
sample_inputs = np.expand_dims(sample_inputs, 1) # make batch_size as 1



# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




gpu_cnt = torch.cuda.device_count()
dim = 28
z_size = 34
n_hidden = 512
n_gaussians = 5
n_layers = 2
if gpu_cnt == 1:
    sys.stdout.write("One GPU\n")
    model = MDNRNN(dim, CNNFeat, z_size, n_hidden, n_gaussians, n_layers).cuda()
elif gpu_cnt > 1:
    sys.stdout.write("More GPU's: {0}\n".format(gpu_cnt))
    model = torch.nn.DataParellel( MDNRNN(dim, CNNFeat, z_size, n_hidden, n_gaussians, n_layers).cuda() )
else:
    sys.stdout.write("No GPU\n")
    model = MDNRNN(dim, CNNFeat, z_size, n_hidden, n_gaussians, n_layers)
    
model = model.double()
    
#criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())#, lr=0.0001, betas=(0.5, 0.999), amsgrad=True)




#model_saved_path = "output/motiondance_simplernn/checkpoints/epoch_100_plus_{0}.pth.tar".format(frozen_after_n_epochs)
model_saved_path = "trained_models/{0}/latest_epoch.pth.tar".format(args["model"])

epoch, model, optimizer = load_checkpoint(model, optimizer, model_saved_path)



model = model.eval()
output_path = "output/Result"
if not os.path.exists(output_path):
    os.makedirs(output_path)



from tqdm import tqdm
prev_poses_cnt = 5
batch_size = 1
prev_poses_input = np.zeros((batch_size, prev_poses_cnt, z_size), dtype=np.float32)
cnt = 0
tmp_results = []
post_proc_results = []
print(sample_inputs.shape)

for input_index in tqdm(range(0, sample_inputs.shape[0], 100)):
    
    with torch.no_grad():
#         print(sample_inputs.shape)
        audio_input = torch.from_numpy(sample_inputs[input_index:input_index+100].reshape(1, 100, 1, 513, 5)).type(torch.DoubleTensor)
        prev_poses_input = torch.from_numpy(prev_poses_input).type(torch.DoubleTensor)
        model = model.to(device)
        hidden = model.init_hidden(batch_size)
        audio_input = audio_input.to(device)
        prev_poses_input =  prev_poses_input.to(device)
        
#         hidden = hidden.to(device)

        (pi, mu, sigma), hidden  = model(audio_input, prev_poses_input, hidden)
        next_steps = get_predicted_steps(pi, mu)
#         print(next_steps.shape)
        prev_poses_input = next_steps[:, -prev_poses_cnt:, :]
        
        #cur_step = np.zeros((1,34), dtype=np.float32)
        for seq_index in range(prev_poses_cnt, next_steps.shape[1]):
            results = next_steps[:, seq_index, :]
            results = results.reshape([17,2])
            tmp_results.append(results)
#             new_results = conv_cord(results)
#             post_proc_results.append(fitting(new_results))



np.save('output/{0}.npy'.format(args["saved_name"]),tmp_results)
# np.save('output/final.npy',post_proc_results)
np.save('output/gt_{0}.npy'.format(args["data"]),next_steps_gt)
