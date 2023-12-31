# -*- coding: utf-8 -*-
"""
@author: zj
@file:   selectivesearch.py
@time:   2020-02-25
"""

import sys
import cv2

def get_selective_search():
    """
    Create and return an instance of SelectiveSearchSegmentation.

    Returns:
       instance of class
    """
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return gs

def config(gs, img, strategy='q'):
  
    #  Selective Search with the provided image and strategy.

    # Set the base image for Selective Search
    gs.setBaseImage(img)

    if strategy == 's':
        gs.switchToSingleStrategy()
    elif strategy == 'f':
        gs.switchToSelectiveSearchFast()
    elif strategy == 'q':
        gs.switchToSelectiveSearchQuality()
    else:
        # Print documentation and exit if an invalid strategy is provided
        print(__doc__)
        sys.exit(1)

def get_rects(gs):

    # Obtain region proposals using Selective Search and adjust rectangle coordinates.

    # Returns:
    #     numpy.ndarray: Processed rectangles with adjusted coordinates.

    # Obtain region proposals using Selective Search
    rects = gs.process()

    # Adjust the rectangle coordinates to get the right and bottom edges
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]

    return rects

if __name__ == '__main__':
   
    gs = get_selective_search()

    img = cv2.imread('./data/lena.jpg', cv2.IMREAD_COLOR)

  
    config(gs, img, strategy='q')

    rects = get_rects(gs)
    print(rects)
