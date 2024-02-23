import torch
import pandas as pd
import numpy as np
import collections
import termcolor
import functools 
import random
import pickle
import os
import gc
import timm
import omegaconf
import wandb
import transformers
import datasets
import sklearn
import sklearn.metrics
import sklearn.model_selection

from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

class DiabeticRetinopathyDataset(torch.utils.data.Dataset):
    
    def __init__(self, img_paths, labels, img_size):
        self.img_paths = img_paths
        self.labels = labels
        self.img_size = img_size
        
    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = np.array(Image.open(img_path).resize((self.img_size, self.img_size)))
        img = np.transpose(img, (2, 0, 1))
        one_hot_label = np.eye(5)[label]
        
        return {
            'img': torch.tensor(img, dtype=torch.float32),
            'label': torch.tensor(one_hot_label, dtype=torch.float32),
        }
    
    def __len__(self):
        return len(self.img_paths)