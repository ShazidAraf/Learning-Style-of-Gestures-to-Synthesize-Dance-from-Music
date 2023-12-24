from scipy.io import wavfile
import os
import sys
import math
import numpy as np
import json
import math
import time
import imageio
import copy
from tqdm import tqdm


def points_dist(p1, p2):
    return math.sqrt(math.pow(p1[0]-p2[0], 2) + math.pow(p1[1] - p2[1], 2));

def length_correction(pose,p1_index,p2_index,l,sholdier=None):
    
    p1 = pose[p1_index]
    p2 = pose[p2_index]
    d = p2-p1
    
    if sholdier == 1:
        p1 =(pose[5]+pose[6])/2
    
    if points_dist(p1, p2)<0.8*l:
        
        part_size = 0.8*l
        theta = np.arctan2(d[0], d[1])
        p2_x = part_size * np.cos(theta)
        p2_y = part_size * np.sin(theta)          
        pose[p2_index][0] = p2_y + pose[p1_index][0]
        pose[p2_index][1] = p2_x + pose[p1_index][1]
        
    elif points_dist(p1, p2)>1.3*l:
        
        part_size = l
        theta = np.arctan2(d[0], d[1])
        p2_x = part_size * np.cos(theta)
        p2_y = part_size * np.sin(theta)          
        pose[p2_index][0] = p2_y + pose[p1_index][0]
        pose[p2_index][1] = p2_x + pose[p1_index][1]
        
        
    return pose



def head_correction(pose,l):
    
    nose_index = 0
    sholder_1 = 5
    sholder_2 = 6
    sholder_mid_point = (pose[sholder_1]+pose[sholder_2])/2
    
    
    p1 = sholder_mid_point
    p2 = copy.deepcopy(pose[nose_index])
    d = p2-p1

    part_size = l
    theta = np.arctan2(d[0], d[1])
    p2_x = part_size * np.cos(theta)
    p2_y = part_size * np.sin(theta)          

    pose[0,0] = p2_y + sholder_mid_point[0]
    pose[0,1] = p2_x + sholder_mid_point[1]
    
    shift = pose[0] - p2
    
    
    pose[1] = pose[1] +shift
    pose[2] = pose[2] +shift
    pose[3] = pose[3] +shift
    pose[4] = pose[4] +shift

    return pose


#hand

def pose_correction(pose,neck_hand_leg=None):


    neck_hand_leg = np.load('cords/neck_hand_leg.npy')


    
    pose = head_correction(pose,neck_hand_leg[0])
    
    pose = length_correction(pose,6,8,neck_hand_leg[2])
    pose = length_correction(pose,8,10,neck_hand_leg[3])
    
    pose = length_correction(pose,5,7,neck_hand_leg[2])
    pose = length_correction(pose,7,9,neck_hand_leg[3])

    
    pose = length_correction(pose,0,12,neck_hand_leg[4],1)
    pose = length_correction(pose,12,14,neck_hand_leg[5])
    pose = length_correction(pose,14,16,neck_hand_leg[6])
    
    
    pose = length_correction(pose,0,11,neck_hand_leg[4],1)
    pose = length_correction(pose,11,13,neck_hand_leg[5])
    pose = length_correction(pose,13,15,neck_hand_leg[6])

    
    
    return pose
    
def check_image(pose):
#     print(pose[0],pose[15],pose[16])
    h =np.maximum(pose[15,0],pose[16,0]) - pose[0,0]
    
    return h
    
    


def conv_cord(prev_cord):
    
    prev_cord = np.array(prev_cord)
    prev_cord = np.reshape(prev_cord,(-1,2))
    tran = np.array([0,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3])
    
    t = copy.deepcopy(prev_cord[:,0])
    prev_cord[:,0] = prev_cord[:,1]
    prev_cord[:,1] = t

    body = np.zeros([18,2])
    j = 0
    for i in range(18):
        if i==1:
            body[i] = (prev_cord[5]+prev_cord[6])/2
        else:
            body[i] = prev_cord[tran[j]]
            j=j+1
            
    return body

def shifting_scaling(prev_cord):
    
    
    # H = 324
    # W = 110
    # head = np.array([256,112])/512
    H = 310
    W = 100
    head = np.array([256,150])/512
    h =np.maximum(prev_cord[10,1],prev_cord[13,1]) - prev_cord[0,1]
#     w = 
    #shifting
    new_cord = copy.deepcopy(prev_cord)
    x = new_cord[0] - head
    new_cord = new_cord - x
    
    m = H/h
    

    new_cord[:,1] = new_cord[:,1]*m
    new_cord[:,0] = (new_cord[:,0]-0.5)*(m/1.2)+256

    
    
    return new_cord




def post_processing_exp(gt,exp,th_height=0.6,th_err= 1,smoothing_len = 5, mode = None):
    
    
    l = np.load('cords/limb_length.npy')
    neck_hand_leg = np.load('cords/neck_hand_leg.npy')
    
    
#     th_height = 0.6
#     th_err = 1


    idx = []

    for i in range(exp.shape[0]):

        a = check_image(exp[i])
        err = np.sum((gt[i]-exp[i])**2)


        if(a>th_height and err<th_err):
            idx.append(i)
            
            
            
            
    from helper.utils import transition

    stand = pose_correction(np.load('cords/stand.npy'))

    cords = []
    if idx[0]!=0:
        cords.append(np.expand_dims(stand, axis=0))
    else:
        cords.append(np.expand_dims(pose_correction(exp[0]), axis=0))


    prev_idx = 0
    prev_step = cords[0][0]

    for i in tqdm(range(len(idx))):

        g = idx[i]-prev_idx

        if g>=smoothing_len:
            prev_idx = idx[i]
            t = transition(g,prev_step,pose_correction(exp[idx[i]]))
            cords.append(t)
            prev_step = pose_correction(exp[idx[i]])
        else:
            continue
            
            
    if (exp.shape[0]>prev_idx):
        # print(exp.shape[0],prev_idx)
        g = exp.shape[0] - prev_idx-1
        # print(g,prev_step.shape,stand.shape)
        t = transition(g,prev_step,stand)
        cords.append(t)
            
    arr =  np.concatenate(cords,axis=0)


    # np.save('output/Result/MDN_modified_{0}.npy'.format(smoothing_len),arr)


    np.save('output/Result/{0}.npy'.format(mode),arr)
 



    
    
    
    
def post_processing_gt(exp,th_height=0.6,smoothing_len = 3, dance_name = None):
    
    
    l = np.load('cords/limb_length.npy')
    neck_hand_leg = np.load('cords/neck_hand_leg.npy')
    

    idx = []

    for i in range(exp.shape[0]):

        a = check_image(exp[i])

        if(a>th_height):
            idx.append(i)
            
            
            
            
    from helper.utils import transition

    stand = np.load('cords/stand.npy')

    cords = []
    if idx[0]!=0:
        cords.append(np.expand_dims(stand, axis=0))
    else:
        cords.append(np.expand_dims(exp[0], axis=0))


    prev_idx = 0
    prev_step = cords[0][0]

    for i in tqdm(range(len(idx))):

        g = idx[i]-prev_idx

        if g>=smoothing_len:
            prev_idx = idx[i]
#             print(g,prev_step,exp[idx[i]])
            t = transition(g,prev_step,pose_correction(exp[idx[i]]))
            cords.append(t)
            prev_step = pose_correction(exp[idx[i]])
        else:
            continue
            
            
    if (exp.shape[0]>prev_idx):
        g = exp.shape[0] - prev_idx-1
        t = transition(g,prev_step,stand)
        cords.append(t)
            
    arr =  np.concatenate(cords,axis=0)


    np.save('output/Result/gt_modified_{0}.npy'.format(dance_name),arr)

    
def chinese_cords_transformation(cd1):

    cd1[:,1] = cd1[:,1]*(-1)
    cd2 = copy.deepcopy(cd1)
    cd2[:,0]-=np.amin(cd1[:,0])
    cd2[:,1]-=np.amin(cd1[:,1])

    H = np.amax(cd2[:,1])
    W = np.amax(cd2[:,0])


#     sh_x = W*.15
#     sh_y = H*.15

    sh_x = 22
    sh_y = 22
    



    cd2[:,0] = cd2[:,0] + sh_x
    cd2[:,1] = cd2[:,1] + sh_y



#     H1 = 1.3*H
#     W1 = 1.3*W
#     print(H1,W1)
    H1 = 195
    W1 = 120

    cd3 = copy.deepcopy(cd2)
    cd3[:,0] = cd3[:,0]/W1
    cd3[:,1] = cd3[:,1]/H1

    prev_cord = cd3

    cords = np.zeros([17,2])
    s1 = np.array([1,0,12,3,13,4,14,5,16,7,17,8,19,10])

    stand = np.load('cords/stand.npy')

    t1 = stand[3]-stand[1]
    t2 = stand[4]-stand[2]
    
    j=0
    for i in range(17):

        if i==0:
            cords[i]= (prev_cord[0]+prev_cord[1])/2
            cords[i,1]= cords[i,1]+0.02
        elif i==3:
            cords[i]= cords[1]+t1
        elif i==4:
            cords[i] = cords[2]-t2

        else:
            cords[i] = prev_cord[s1[j]]
            j+=1
    #     print(cords)

    new_cords = copy.deepcopy(cords)
    t = copy.deepcopy(new_cords[:,0])
    new_cords[:,0] = new_cords[:,1]
    new_cords[:,1] =t
    
    return new_cords
        
        
# print(stand,new_cords)
