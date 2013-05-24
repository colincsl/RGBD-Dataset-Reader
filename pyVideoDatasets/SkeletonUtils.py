
import numpy as np
import cv2
import time
from pyVideoDatasets.DepthUtils import skel2depth, depth2world, world2depth

N_MSR_JOINTS = 20
N_KINECT_JOINTS = 14


def transform_skels(skels, transformation, output='image'):
    '''
    ---Parameters---
    skels : list of skeletons in frame 1
    transformation : 4x4 transform from frame 1 to 2
    output : 'image' or 'world' for either coordinate system
    ---Result---
    skels_out : skeletons in frame 2
    '''
    skels_out = []
    for skel_c1 in skels:
        if np.all(skel_c1 != -1):
            skels_mask = skel_c1 == 0
            # Convert to depth coordinate system
            skel_c1 = depth2world(skel2depth(skel_c1, [240,320]), [240,320])
            # Transform from cam1 -> cam2
            skel_c2 = np.dot(transformation[:3,:3], skel_c1.T).T + transformation[:3,3]

            if len(skel_c2) != N_MSR_JOINTS:
                skel_c2 = kinect_to_msr_skel(skel_c2)

            skel_c2[skels_mask] = 0

            if output=='world':
                skels_out += [skel_c2]
            elif output=='image':
                # Get skel in image (cam2) coordinates
                skel_im2 = world2depth(skel_c2, [240,320])
                skels_out += [skel_im2]


    return skels_out


# ----------------------------------------------------------------
def kinect_to_msr_skel(skel):
    # SKEL_JOINTS = [0, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 17, 19] # Low

    skel_msr = np.zeros([N_MSR_JOINTS, 3])
    skel_msr[3,:] = skel[0,:] #head
    skel_msr[1,:] = skel[1,:] #torso
    skel_msr[0,:] = skel[1,:] #torso
    skel_msr[4,:] = skel[2,:] #l shoulder
    skel_msr[5,:] = skel[3,:] #l elbow
    skel_msr[7,:] = skel[4,:] #l hand
    skel_msr[8,:] = skel[5,:] #r shoudler
    skel_msr[9,:] = skel[6,:] #r elbow
    skel_msr[11,:] = skel[7,:] #r hand
    skel_msr[12,:] = skel[8,:] #l hip
    skel_msr[13,:] = skel[9,:] #l knee
    skel_msr[15,:] = skel[10,:] #l foot
    skel_msr[16,:] = skel[11,:] #r hip
    skel_msr[17,:] = skel[12,:] #r knee
    skel_msr[19,:] = skel[13,:] #r foot

    return skel_msr.astype(np.int16)

def msr_to_kinect_skel(skel):
    # SKEL_JOINTS = [0, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 17, 19] # Low

    skel_kinect = np.zeros([N_KINECT_JOINTS, 3], dtype=np.int16)
    skel_kinect[0,:] = skel[3,:] #head
    # skel_kinect[1,:] = skel[1,:] #torso
    skel_kinect[1,:] = skel[0,:] #torso
    skel_kinect[2,:] = skel[4,:] #l shoulder
    skel_kinect[3,:] = skel[5,:] #l elbow
    skel_kinect[4,:] = skel[7,:] #l hand
    skel_kinect[5,:] = skel[8,:] #r shoudler
    skel_kinect[6,:] = skel[9,:] #r elbow
    skel_kinect[7,:] = skel[11,:] #r hand
    skel_kinect[8,:] = skel[12,:] #l hip
    skel_kinect[9,:] = skel[13,:] #l knee
    skel_kinect[10,:] = skel[15,:] #l foot
    skel_kinect[11,:] = skel[16,:] #r hip
    skel_kinect[12,:] = skel[17,:] #r knee
    skel_kinect[13,:] = skel[17,:] #r foot

    return skel_kinect

def mhad_to_kinect_skel(skel):
    # SKEL_JOINTS = [0, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 17, 19] # Low
    dims = skel.shape[1]
    skel_kinect = np.zeros([N_KINECT_JOINTS, dims], dtype=np.int16)

    if skel[2,0]!=0: #head
        skel_kinect[0,:] = skel[2,:]
    else:
        try:
            skel_kinect[0,:] = skel[0:3,:][np.argwhere(skel[0:3,2]!=0)[0]]
        except:
            pass

    if skel[3,0]!=0:#torso
        skel_kinect[1,:] = skel[3,:]
    else:
        try:
            skel_kinect[1,:] = skel[[3,4,6],:][np.argwhere(skel[[3,4,6],2]!=0)[0]]
        except:
            pass

    skel_kinect[2,:] = skel[12,:] #l shoulder
    skel_kinect[3,:] = skel[13,:] #l elbow
    # skel_kinect[4,:] = skel[15,:] if skel[15,0]!=0 else np.max(skel[16:19,:], 0) #l hand
    if skel[15,0]!=0: #l hand
        skel_kinect[4,:] = skel[15,:]
    else:
        try:
            skel_kinect[4,:] = skel[16:19,:][np.argwhere(skel[16:19,2]!=0)[0]]
        except:
            pass

    skel_kinect[5,:] = skel[20,:] #r shoudler
    skel_kinect[6,:] = skel[21,:] #r elbow
    # skel_kinect[7,:] = skel[23,:] if skel[23,0]!=0 else np.max(skel[23:27,:], 0) #r hand
    if skel[23,0]!=0:#r hand
        skel_kinect[7,:] = skel[23,:]
    else:
        try:
            skel_kinect[7,:] = skel[23:27,:][np.argwhere(skel[23:27,2]!=0)[0]]
        except:
            pass

    skel_kinect[8,:] = skel[28,:] #l hip
    skel_kinect[9,:] = skel[31,:] #l knee
    skel_kinect[10,:] = skel[33,:] if skel[33,0]!=0 else skel[34,:] #l foot
    skel_kinect[11,:] = skel[37,:] #r hip
    skel_kinect[12,:] = skel[39,:] #r knee
    skel_kinect[13,:] = skel[41,:] if skel[41,0]!=0 else skel[42,:] #r foot
    # print skel_kinect
    return skel_kinect

# def mhad_to_kinect_skel(skel):
#     # SKEL_JOINTS = [0, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 17, 19] # Low
#     dims = skel.shape[1]
#     skel_kinect = np.zeros([N_KINECT_JOINTS, dims], dtype=np.int16)
#     skel_kinect[0,:] = skel[2,:] if skel[2,0]!=0 else np.max(skel[0:3,:], 0) #head
#     skel_kinect[1,:] = skel[3,:] if skel[3,0]!=0 else np.max(skel[[3,4,7,8,19],:], 0)#torso
#     skel_kinect[2,:] = skel[12,:] #l shoulder
#     skel_kinect[3,:] = skel[13,:] #l elbow
#     skel_kinect[4,:] = skel[15,:] if skel[15,0]!=0 else np.max(skel[16:19,:], 0) #l hand
#     skel_kinect[5,:] = skel[20,:] #r shoudler
#     skel_kinect[6,:] = skel[21,:] #r elbow
#     skel_kinect[7,:] = skel[23,:] if skel[23,0]!=0 else np.max(skel[23:27,:], 0) #r hand
#     skel_kinect[8,:] = skel[28,:] #l hip
#     skel_kinect[9,:] = skel[31,:] #l knee
#     skel_kinect[10,:] = skel[33,:] if skel[33,0]!=0 else skel[34,:] #l foot
#     skel_kinect[11,:] = skel[37,:] #r hip
#     skel_kinect[12,:] = skel[39,:] #r knee
#     skel_kinect[13,:] = skel[41,:] if skel[41,0]!=0 else skel[42,:] #r foot
#     # print skel_kinect
#     return skel_kinect

def j11_to_kinect_skel(skel):
    skel_kinect = np.zeros([N_KINECT_JOINTS, 3], dtype=np.int16)
    skel_kinect[0,:] = skel[0,:] #head
    skel_kinect[2,:] = skel[1,:] #l shoulder
    skel_kinect[3,:] = skel[2,:] #l elbow
    skel_kinect[4,:] = skel[3,:] #l hand
    skel_kinect[5,:] = skel[4,:] #r shoudler
    skel_kinect[6,:] = skel[5,:] #r elbow
    skel_kinect[7,:] = skel[6,:] #r hand
    skel_kinect[9,:] = skel[7,:] #l knee
    skel_kinect[10,:] = skel[8,:] #l foot
    skel_kinect[12,:] = skel[9,:] #r knee
    skel_kinect[13,:] = skel[10,:] #r foot
    return skel_kinect

def j13_to_kinect_skel(skel):
    skel_kinect = np.zeros([N_KINECT_JOINTS, 3], dtype=np.int16)
    skel_kinect[0,:] = skel[0,:] #head
    skel_kinect[2,:] = skel[1,:] #l shoulder
    skel_kinect[3,:] = skel[2,:] #l elbow
    skel_kinect[4,:] = skel[3,:] #l hand
    skel_kinect[5,:] = skel[4,:] #r shoudler
    skel_kinect[6,:] = skel[5,:] #r elbow
    skel_kinect[7,:] = skel[6,:] #r hand
    skel_kinect[8,:] = skel[7,:] #l hip
    skel_kinect[9,:] = skel[8,:] #l knee
    skel_kinect[10,:] = skel[9,:] #l foot
    skel_kinect[11,:] = skel[10,:] #r hip
    skel_kinect[12,:] = skel[11,:] #r knee
    skel_kinect[13,:] = skel[12,:] #r foot
    return skel_kinect

def j14_to_kinect_skel(skel):
    skel_kinect = np.zeros([N_KINECT_JOINTS, 3], dtype=np.int16)
    skel_kinect[0,:] = skel[0,:] #head
    skel_kinect[1,:] = skel[1,:] #torso
    skel_kinect[2,:] = skel[2,:] #l shoulder
    skel_kinect[3,:] = skel[3,:] #l elbow
    skel_kinect[4,:] = skel[4,:] #l hand
    skel_kinect[5,:] = skel[5,:] #r shoudler
    skel_kinect[6,:] = skel[6,:] #r elbow
    skel_kinect[7,:] = skel[7,:] #r hand
    skel_kinect[8,:] = skel[8,:] #l hip
    skel_kinect[9,:] = skel[9,:] #l knee
    skel_kinect[10,:] = skel[10,:] #l foot
    skel_kinect[11,:] = skel[11,:] #r hip
    skel_kinect[12,:] = skel[12,:] #r knee
    skel_kinect[13,:] = skel[13,:] #r foot
    return skel_kinect

def j15_to_kinect_skel(skel):
    skel_kinect = np.zeros([N_KINECT_JOINTS, 3], dtype=np.int16)
    skel_kinect[0,:] = skel[0,:] #head
    skel_kinect[1,:] = skel[2,:] #torso
    skel_kinect[2,:] = skel[3,:] #l shoulder
    skel_kinect[3,:] = skel[4,:] #l elbow
    skel_kinect[4,:] = skel[5,:] #l hand
    skel_kinect[5,:] = skel[6,:] #r shoudler
    skel_kinect[6,:] = skel[7,:] #r elbow
    skel_kinect[7,:] = skel[8,:] #r hand
    skel_kinect[8,:] = skel[9,:] #l hip
    skel_kinect[9,:] = skel[10,:] #l knee
    skel_kinect[10,:] = skel[11,:] #l foot
    skel_kinect[11,:] = skel[12,:] #r hip
    skel_kinect[12,:] = skel[13,:] #r knee
    skel_kinect[13,:] = skel[14,:] #r foot
    return skel_kinect

from skimage.draw import circle, line
def display_skeletons(img, skel, color=(200,0,0), skel_type='MSR', skel_contraints=None):
    '''
    skel_type : 'MSR' or 'Low' ##or 'Upperbody'
    '''

    if img.shape[0] == 480:
        pt_radius = 6
        tube_radius = 3
    else:
        pt_radius = 5
        tube_radius = 2

    img = np.ascontiguousarray(img)
    if skel_type == 'MSR':
        joints = range(N_MSR_JOINTS)
        joint_names = ['torso1', 'torso2', 'neck', 'head',
                        'r_shoulder', 'r_elbow', 'r_wrist', 'r_hand',
                         'l_shoulder', 'l_elbow', 'l_wrist', 'l_hand',
                         'r_pelvis', 'r_knee', 'r_ankle', 'r_foot',
                         'l_pelvis', 'l_knee', 'l_ankle', 'l_foot'
                         ]
        connections = [
                        [3, 2],[2,1],[1,0], #Head to torso
                        [2, 4],[4,5],[5,6],[6,7], # Left arm
                        [2, 8],[8,9],[9,10],[10,11], # Right arm
                        [0,12],[12,13],[13,14],[14,15], #Left foot
                        [0,16],[16,17],[17,18],[18,19] #Right foot
                        ]
        head = 3
    elif skel_type == 'Low':
        joints = [0, 3, 4, 5, 7, 8, 9, 11, 13, 15, 17, 19] # Low
        # joints = [0, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 17, 19]
        # joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 17, 19]
        connections = [
                        # [3, 2],[2,0], #Head to torso
                        [3, 0], #Head to torso
                        # [3, 2],[2,1],[1,0], #Head to torso
                        [4,8], #connect shoulders
                        [0, 4],[4,5],[5,7], # Left arm
                        [0, 8],[8,9],[9,11], # Right arm
                        [0,13],[13,15], #Left foot
                        [0,17],[17,19]
                        ]
        head = 3
    elif skel_type == 'Upperbody':
        joints = [0, 3, 4, 5, 7, 8, 9, 11]
        # joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11]
        connections = [
                        [3,0], #Head to torso
                        [0, 4],[4,5],[5,7], # Left arm
                        [4,8], #connect shoudlers
                        [0, 8],[8,9],[9,11], # Right arm
                        ]
        head = 3

    elif skel_type == 'Kinect':
        joints = range(14)
        connections = [
                        [0,1], #Head to torso
                        [1,2],[2,3],[3,4], # Left arm
                        [2,5],# connect shoulders
                        [1,5],[5,6],[6,7], # Right arm
                        [1,8],[8,9],[9,10], #Left foot
                        [1,11],[11,12],[12,13], #Right foot
                        [8,11] #Bridge hips
                        ]
        joint_names = ['head', 'torso', 'l_shoulder', 'l_elbow', 'l_hand',
                    'r_shoulder', 'r_elbow', 'r_hand',
                    'l_hip', 'l_knee', 'l_foot',
                    'r_hip', 'r_knee', 'r_foot']
        head = 0
    elif skel_type == 'Ganapathi':
        joints = range(15)
        connections = [
            [0,1],[1,2],#Head to neck, neck to torso
            [1,3],[3,4],[4,5], # Left arm
            [3,9],[6,12], # shoudlers to hips
            [1,6],[6,7],[7,8], # Right arm
            [2,9],[9,10],[10,11], #Left foot
            [2,12],[12,13],[13,14], #Right foot
            [9,12] #Bridge hips
            ]
        joint_names = ['head', 'neck', 'torso', 'l_shoulder', 'l_elbow', 'l_hand',
                    'r_shoulder', 'r_elbow', 'r_hand',
                    'l_hip', 'l_knee', 'l_foot',
                    'r_hip', 'r_knee', 'r_foot']
        head = 0
    elif skel_type == 'MHAD':
        joints = range(43)
        connections = [
                        [0,1],[0,2] #Head to torso
                        [1,2],[2,3],[3,4], # Left arm
                        [2,5],# connect shoulders
                        [1,5],[5,6],[6,7], # Right arm
                        [1,8],[8,9],[9,10], #Left foot
                        [1,11],[11,12],[12,13], #Right foot
                        [8,11] #Bridge hips
                        ]
        joint_names = ['head', 'torso', 'l_shoulder', 'l_elbow', 'l_hand',
                    'r_shoulder', 'r_elbow', 'r_hand',
                    'l_hip', 'l_knee', 'l_foot',
                    'r_hip', 'r_knee', 'r_foot']
        head = 0

    elif skel_type == 'Other':
        joints = range(len(skel))
        connections = skel_contraints
        head = 0

    for i in joints:
        j = skel[i]
        # Remove zero nodes
        try:
            if not (j[0] <= 0 or j[1] <= 0):
                # circ = skimage.draw.circle(j[0],j[1], 5)
                # img[circ[0], circ[1]] = color
                cv2.circle(img, (j[1], j[0]), pt_radius, color, -1)
        except:
            pass

    # Make head a bigger node
    cv2.circle(img, (skel[head,1], skel[head,0]), pt_radius*3, color)

    for c in connections:
        # Remove zero nodes
        if not ( (skel[c[0],0]==0 and skel[c[0],1]==0) or (skel[c[1],0]==0 and skel[c[1],1]==0)):
        # if not ( (skel[c[0],0]<=0 and skel[c[0],1]<=0) or (skel[c[1],0]<=0 and skel[c[1],1]<=0)):
            cv2.line(img, (skel[c[0],1], skel[c[0],0]), (skel[c[1],1], skel[c[1],0]), color, tube_radius)

    return img
