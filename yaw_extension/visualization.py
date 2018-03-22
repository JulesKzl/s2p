import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import json
from skimage.transform import ProjectiveTransform, warp

import ipytools
import ipytools2

import sys
sys.path.append('../')
from s2plib import rpc_model

def get_images():
    images_list = []
    with open('config.json', 'r') as f:
        user_cfg = json.load(f)
    if (user_cfg["full_img"]):
        roi = None
    else:
        roi = user_cfg["roi"]
    for el in user_cfg['images']:
        im_name = el['img']
        print(im_name)
        im = ipytools.readGTIFF(im_name)
        if (roi != None):
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            im = im[y:y+h, x:x+w]
        images_list.append(ipytools.simple_equalization_8bit(im))
    return images_list


def get_tiles_path():
    """ Get paths of different tiles """
    tiles_path = []
    for dirname, dirnames, filenames in os.walk('./output/tiles'):
        if (dirname.split('/')[-1] == 'pair_1'):
            tiles_path.append(dirname)
    return tiles_path

def get_correction_matrix(tile):
    A = np.loadtxt(tile + '/pointing.txt', usecols=range(3))
    return A

def get_matches(tile):
    m = np.loadtxt(tile + '/sift_matches.txt')
    return m

def get_affine_fundamental_matrix(tile):
    F = np.loadtxt(tile + '/affine_fundamental_matrix.txt', usecols=range(3))
    return F

def visualize_transformation(im1, im2, A):
    tform = ProjectiveTransform(A)
    out = warp(im2, tform)

    f, ax = plt.subplots(1, 3, figsize=(20,10))
    ax[0].imshow(im1.squeeze(), cmap='gray')
    ax[1].imshow(im2.squeeze(), cmap='gray')
    ax[2].imshow(out.squeeze(), cmap='gray')
    plt.show()
    return out

def visualize_epipolar(im1, im2, F, m):
    """ Visualize epipolar line in the second image from a point in
        the first image
    """
    plt.figure()
    f, ax = plt.subplots(1, 2, figsize=(20,10))
    for (x1, y1, x2, y2) in m:

        p1 = np.array([x1, y1, 1]).reshape(3, 1)
        Fp = np.dot(F, p1)
        [a, b, c] = Fp.flatten()

        y3 = np.array([0, 1000])
        x3 =  - (b*y3 + c)/a

        ax[0].plot(x1, y1, 'r+')
        ax[1].plot(x3, y3, 'r-')
        ax[1].plot(x2, y2, 'r+', color='blue')
        ax[0].imshow(im1.squeeze(), cmap='gray')
        ax[1].imshow(im2.squeeze(), cmap='gray')

def visualize_pointing_error(F, m, A, plot=True):
    """ Visualize epipolar line in the second image from a point in
        the first image
    """
    if (plot):
        plt.figure()
        f, ax = plt.subplots(1, 2, figsize=(20,10))
    e_2 = 0
    e_3 = 0
    for (x1, y1, x2, y2) in m:
        p1 = np.array([x1, y1, 1]).reshape(3, 1)
        Fp = np.dot(F, p1)
        [a, b, c] = Fp.flatten()

        # Compute orignal picture
        proj_p2 = np.abs((a*x2 + b*y2 + c)/(a**2 + b**2))*np.array([a, b])
        if (plot):
            ax[0].plot(x2, y2, '.', alpha=0)
            ax[0].arrow(x2, y2, -proj_p2[0], -proj_p2[1], head_width=3, head_length=5, color='green')

        # Compute after transformation
        p2 = np.array([x2, y2, 1]).reshape(3, 1)
        Ap2 = np.dot(A, p2)
        [x3, y3, _] = Ap2.flatten()
        proj_p3 = np.abs((a*x3 + b*y3 + c)/(a**2 + b**2))*np.array([a, b])

        # Compute global error
        e_2 += np.abs((a*x2 + b*y2 + c)/np.sqrt(a**2 + b**2))
        e_3 += np.abs((a*x3 + b*y3 + c)/np.sqrt(a**2 + b**2))

        # Display
        if (plot):
            ax[1].plot(x3, y3, '.', alpha=0)
            ax[1].arrow(x3, y3, -proj_p3[0], -proj_p3[1], head_width=3, head_length=5, color='green')
    return e_2, e_3
