import torch
from torch import nn
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

class DiabeticRetinopathyRepViTModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.num_classes, 5)
    def forward(self, img):
        x = self.backbone(img)
        return self.classifier(x)

