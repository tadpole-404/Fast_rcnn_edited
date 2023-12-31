# -*- coding: utf-8 -*-
"""
@date: 2020/3/31 下午8:26
@file: custom_finetune_dataset.py
@author: zj
@description: Custom dataset class for fine-tuning object detection models.
"""
import random
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import utils.util as util

class CustomFinetuneDataset(Dataset):
    def __init__(self, root_dir, transform):
        """
        Load all images and positive/negative bounding boxes for fine-tuning.

        Args:
            root_dir : Root directory of the dataset.
            transform : Image transformations.
        """
        self.transform = transform

        # Parse car CSV to get samples
        samples = util.parse_car_csv(root_dir)

        jpeg_images = list()
        annotation_dict = dict()

        # Load images and annotations for each sample
        for idx in range(len(samples)):
            sample_name = samples[idx]
            img = cv2.imread(os.path.join(root_dir, 'JPEGImages', sample_name + ".jpg"))
            h, w = img.shape[:2]
            jpeg_images.append(img)

            positive_annotation_path = os.path.join(root_dir, 'Annotations', sample_name + '_1.csv')
            positive_annotations = np.loadtxt(positive_annotation_path, dtype=np.float, delimiter=' ')
            if len(positive_annotations.shape) == 1:
                positive_annotations = positive_annotations[np.newaxis, :]
            positive_annotations[:, 0] /= w
            positive_annotations[:, 1] /= h
            positive_annotations[:, 2] /= w
            positive_annotations[:, 3] /= h

            negative_annotation_path = os.path.join(root_dir, 'Annotations', sample_name + '_0.csv')
            negative_annotations = np.loadtxt(negative_annotation_path, dtype=np.float, delimiter=' ')
            negative_annotations[:, 0] /= w
            negative_annotations[:, 1] /= h
            negative_annotations[:, 2] /= w
            negative_annotations[:, 3] /= h

            annotation_dict[str(idx)] = {'positive': positive_annotations, 'negative': negative_annotations}

        self.jpeg_images = jpeg_images
        self.annotation_dict = annotation_dict

    def __getitem__(self, index: int):
        """
        Sample 64 RoIs from the image at the specified index, with 16 positive and 48 negative samples.


             Index of the image in the dataset.

      
             Tuple containing image, targets, and rectangle array.
        """
        assert index < len(self.jpeg_images), 'Current dataset size: %d, Input Index: %d' % (len(self.jpeg_images), index)

        image = self.jpeg_images[index]
        annotation_dict = self.annotation_dict[str(index)]
        positive_annotations = annotation_dict['positive']
        negative_annotations = annotation_dict['negative']

        positive_num = 16
        negative_num = 48

        # Adjust positive and negative sample numbers based on availability
        if len(positive_annotations) < positive_num:
            positive_num = len(positive_annotations)
            negative_num = 64 - positive_num

            positive_array = positive_annotations
        else:
            positive_array = positive_annotations[random.sample(range(positive_annotations.shape[0]), positive_num)]
        negative_array = negative_annotations[random.sample(range(negative_annotations.shape[0]), negative_num)]

        rect_array = np.vstack((positive_array, negative_array))
        targets = np.hstack((np.ones(positive_num), np.zeros(negative_num)))

        if self.transform:
            image = self.transform(image)

        return image, targets, rect_array

    def __len__(self) -> int:
        """
        Get the total number of images in the dataset.

        Returns:int: Number of images in the dataset.
        """
        return len(self.jpeg_images)

if __name__ == '__main__':
    # Example usage of the CustomFinetuneDataset class
    root_dir = '../../data/finetune_car/train'
    s = 600
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(s),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_set = CustomFinetuneDataset(root_dir, transform)
    print(len(data_set.jpeg_images))

    image, targets, rect_array = data_set.__getitem__(10)
    print(image.shape)
    print(targets)
    print(rect_array.shape)
