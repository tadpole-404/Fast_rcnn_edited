# -*- coding: utf-8 -*-
"""
@date: 2020/3/31 下午2:55
@file: vgg16_roi.py
@author: zj
@description: VGG16-based model for Region of Interest (RoI) classification and bounding box regression.
"""

import torch
import torch.nn as nn
import torchvision.models as models

import models.roi_pool as roi_pool

class VGG16_RoI(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        """
        Initialize the VGG16_RoI model.

        Args:
            num_classes (int): Number of classes (excluding background class).
            init_weights (bool): If True, initialize model weights.
        """
        super(VGG16_RoI, self).__init__()

        # Define VGG16-like convolutional layers with modifications for RoI tasks
        feature_list = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.features = models.vgg.make_layers(feature_list)

        # RoI pooling layer with output size (7, 7)
        self.roipool = roi_pool.ROI_Pool((7, 7))

        # Classifier and regression 
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.softmax = nn.Linear(4096, num_classes + 1)  # Softmax layer for classification
        self.bbox = nn.Linear(4096, num_classes * 4)  # Linear layer for bounding box regression

        # Initialize model weights if specified
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        Forward pass through the VGG16_RoI model.
        Returns:Tuple containing classification and regression outputs.
        """
        x = self.features(x)
        x = self.roipool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        classify = self.softmax(x)  # Class
