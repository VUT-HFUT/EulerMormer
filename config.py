import os
import time
import numpy as np

class Config(object):
    def __init__(self):

        # General
        self.epochs = 100   
        self.batch_size = 4   
        self.date = 'Euler'     
        self.numdata = 100000  # 100000
        self.workers = 16  # 16
        # Data
        self.test_batch_size = 1  
        self.test_workers = 16
        self.numtestdata = 600 
        # Data
        self.data_dir = './datasets'  
        self.dir_train = os.path.join(self.data_dir, 'train')
        # Test Data
        self.dir_baby = os.path.join(self.data_dir, 'train/train_vid_frames/val_baby')
        self.dir_gun = os.path.join(self.data_dir, 'train/train_vid_frames/val_gun')
        self.dir_water = os.path.join(self.data_dir, 'train/train_vid_frames/val_water')
        self.dir_drone = os.path.join(self.data_dir, 'train/train_vid_frames/val_drone')
        self.dir_guitar = os.path.join(self.data_dir, 'train/train_vid_frames/val_guitar')
        self.dir_cattoy = os.path.join(self.data_dir, 'train/train_vid_frames/val_cattoy')
        self.dir_drum = os.path.join(self.data_dir, 'train/train_vid_frames/val_drum')
        self.dir_wrist = os.path.join(self.data_dir, 'train/train_vid_frames/val_wrist')
        self.dir_eye = os.path.join(self.data_dir, 'train/train_vid_frames/val_eye')
        self.dir_cranecrop = os.path.join(self.data_dir, 'train/train_vid_frames/val_cranecrop')
        self.dir_crane = os.path.join(self.data_dir, 'train/train_vid_frames/val_crane')
        self.dir_engine = os.path.join(self.data_dir, 'train/train_vid_frames/val_engine')
        self.dir_throat = os.path.join(self.data_dir, 'train/train_vid_frames/val_throat')
        self.dir_face = os.path.join(self.data_dir, 'train/train_vid_frames/val_face')
        self.dir_camera = os.path.join(self.data_dir, 'train/train_vid_frames/val_camera')
        self.dir_sha = os.path.join(self.data_dir, 'train/train_vid_frames/val_sha')
        self.dir_woman = os.path.join(self.data_dir, 'train/train_vid_frames/val_woman')



        self.frames_train = 'coco100000'        # you can adapt 100000 to a smaller number to train
        self.cursor_end = int(self.frames_train.split('coco')[-1])  
        self.coco_amp_lst = np.loadtxt(os.path.join(self.dir_train, 'train_mf.txt'))[:self.cursor_end]
        self.videos_train = []
        self.load_all = False        # Don't turn it on, unless you have such a big mem.
                                     # On coco dataset, 100, 000 sets -> 850G

        # Training
        self.lr = 2e-4 
        self.betas = (0.9, 0.999) 
        self.weight_decay=0.0
        self.batch_size_test = 1 
        self.preproc = ['poisson'] 
        self.pretrained_weights = ''

        # Callbacks
        self.num_val_per_epoch = 1000 
        self.save_dir = 'weights_date{}'.format(self.date)
        self.time_st = time.time()
        self.losses = []


import json

""" configuration json """
class Configjson(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Configjson(config)
