from torch.utils.data.dataloader import Dataset
import torchvision.transforms.functional as TF
import numpy as np
import random

class AugmentPcData():
    def __init__(self, rand_rotate=True, jitter=True):
        self.rand_rotate=rand_rotate
        self.jitter = jitter

class RotationTransform():
    def __init__(self, angles):
        """Rotate by one of the given angles.
           Usage:  MyRotationTransform(angles=[-30, -15, 0, 15, 30])
        """
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class PcDataset(Dataset):
    def __init__(self, pc_file, size=64000):
        """ 
        pc_file: numpy array of point_clouds and their landmarks: each data point is a  
                 concatenation of [pc_array, landmark_array]
        """
        self.pc_file = pc_file 
        self.size = size
        self.pc_data = np.load(pc_file, allow_pickle=True)
        self.samples, self.num_cordinates = self.pc_data.shape

    def normalize(self, pc_data):
        # normalize pcloud dataset around the centroid of the training set
        # the centroid here is specific to the train-split. 
        pc_tensor = TF.to_tensor(pc_data)
        norm_pc_data = TF.normalize(pc_tensor, mean=(0.005), std=(0.2))
       
        #norm_pc_data.reshape() 
        return norm_pc_data

    def __getitem__(self, idx):
        data_point = self.pc_data[idx]
        _3d_points, _, landmarks = data_point
        # select "size" number of points in the point cloud for use 
        if self.size > _3d_points.shape[0]:
            self.size = _3d_points.shape[0]
        
        select = np.random.choice(range(self.size), self.size, replace=False)
        _3d_points = _3d_points[select]
        
        norm_data_point = self.normalize(_3d_points).squeeze(0).transpose(0,1)
        labels_vec = landmarks.flatten()
        return norm_data_point, labels_vec

    def __len__(self):
        return self.samples
