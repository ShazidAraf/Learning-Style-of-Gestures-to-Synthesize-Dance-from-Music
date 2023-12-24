from scipy.io import wavfile
import os
import sys
import math
import numpy as np
import json
import math
import time
import imageio
from pathlib import Path
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




from model_utils import PoseMusicDataset_new, CNNFeat, MDNRNN, Nelson_model
from model_utils import save_checkpoint, load_checkpoint,load_data,criterion,compute_loss, get_predicted_steps
from post_processing import post_processing_exp,post_processing_gt,shifting_scaling,conv_cord,pose_correction,points_dist
from audio_utils import _get_stft_spectogram


import argparse
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--dance_name", required=True,help="Dance Name")
ap.add_argument("-m", "--model", required=True,help="Trained model path")

args = vars(ap.parse_args())


import librosa
audio_file_name = args["dance_name"]
audio, audio_rate = librosa.load('song/audio/{0}.mp3'.format(audio_file_name), sr = 48000)

# print(audio.shape)
actual_audio_rate = audio_rate
# print("Audio rate:", audio_rate)

audio = audio[audio_rate*20:audio_rate*(20+10)]
librosa.output.write_wav('song/audio/test_audio/{0}}.wav'.format(args["dance_name"]), audio,audio_rate)

num_secs = math.floor(len(audio)/audio_rate)
fps = 25
num_secs = num_secs*fps
n_audio_rate = int(audio_rate/fps)
print(n_audio_rate, num_secs)


sample_inputs = []
for index in range(num_secs):
    a = _get_stft_spectogram(audio[n_audio_rate*index: n_audio_rate*(index+1)], actual_audio_rate)
#     print(len(a))
    sample_inputs.append(a)
print(len(sample_inputs))
sample_inputs = np.array(sample_inputs)
sample_inputs = np.expand_dims(sample_inputs, 1) # for input add channel
sample_inputs = np.expand_dims(sample_inputs, 1) # make number of sequences as 1
sample_inputs = np.expand_dims(sample_inputs, 1) # make batch_size as 1

print(sample_inputs.shape)


from model_utils import PoseMusicDataset_new, CNNFeat, MDNRNN, Nelson_model
from model_utils import save_checkpoint, load_checkpoint,load_data,criterion,compute_loss, get_predicted_steps
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
output_path = "song/result"
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



np.save('song/result/tmp_results.npy',tmp_results)


l = np.load('cords/limb_length.npy')
neck_hand_leg = np.load('cords/neck_hand_leg.npy')

results  = np.load('song/tmp_results.npy')


post_processing_gt(results,th_height=0.6,smoothing_len = 5,dance_name = args[])
mdn = np.load('song/result/{0}'.format(args[""]))


import numpy as np
from helper.utils import create_label_bw, create_label, image_to_video
import cv2


parts_id = np.load('cords/parts_id.npy')

for i in tqdm(range(mdn.shape[0])):
    
#     pose = pose_correction(cords_exp[i],neck_hand_leg)
    points = shifting_scaling(conv_cord(mdn[i]))
    label = create_label((512,512,3),points, parts_id)
    cv2.imwrite('output/MDN/{:05}.png'.format(i), label)
    
    label = create_label_bw((512,512),points, parts_id)
    cv2.imwrite('output/test_label/{:05}.png'.format(i), label)

img_dir = Path('./data/test_audio/')
path_in = str(img_dir)
path_out = 'MDN.avi'
image_to_video(path_in,path_out)
