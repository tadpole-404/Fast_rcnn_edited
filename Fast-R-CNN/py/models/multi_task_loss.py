# -*- coding: utf-8 -*-
"""
@date: 2020/3/31 下午3:33
@file: multi_task_loss.py
@author: zj
@description: Implementation of a Multi-Task Loss function for object detection tasks.
"""

import torch
import torch.nn as nn

from models.smooth_l1_loss import SmoothL1Loss

class MultiTaskLoss(nn.Module):
    def __init__(self, lam=1):
        """
        Constructor for the MultiTaskLoss class.
            lam: A hyperparameter for controlling the balance between classification and localization loss.
        """
        super(MultiTaskLoss, self).__init__()
        self.lam = lam
        
        # L_cls uses cross-entropy loss for classification
        self.cls = nn.CrossEntropyLoss()
        
        # L_loc is a custom smooth L1 loss for localization
        self.loc = SmoothL1Loss()

    def forward(self, scores, preds, targets):
        """
        Compute the multi-task loss. Here, N represents the number of RoIs (Region of Interest).

      
        Returns:
            The computed multi-task loss.
        """
        N = targets.shape[0]
        # Combine classification loss (L_cls) and localization loss (L_loc)
        return self.cls(scores, targets) + self.loc(scores[range(N), self.indicator(targets)],
                                                    preds[range(N), self.indicator(preds)])

    def indicator(self, cate):
       
        # to filter out background class.

        # Args:
        #     cate : Tensor representing class labels.

        # Returns:
        #      Boolean tensor indicating whether a class is not background (0).
        
        return cate != 0
