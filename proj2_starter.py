# --------------------------------------------------------
# Written by Yufei Ye and modified by Sheng-Yu Wang (https://github.com/JudyYe)
# Convert from MATLAB code https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj3/gradient_starter.zip
# --------------------------------------------------------
from __future__ import print_function

import argparse
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr

def toy_recon(im):
    im_h, im_w = im.shape

    num_eq = 2 * im_h * im_w + 1 # three equations to minimize 

    # known vector 
    b = np.zeros((num_eq, 1))
    # coefficient matrix 
    A = np.zeros((num_eq, im_h * im_w))

    # number at location (y, x) stores index (y*width + x)
    pix2ind = np.arange(im_h * im_w).reshape((im_h, im_w)).astype(int)

    # A: (e by (im_h*im_w)) sparse matrix (for each eq, each pixel is 1 or 0)
    # v: ((im_h*im_w) * 1)
    # b: (e * 1)
    eq = 0

    # Gradient in x
    for y in range(im_h):
        for x in range(im_w - 1): # minus one since we access x+1
            # first equation: x gradient  
            # v01 - v00 = s01 - s00
            A[eq, pix2ind[y,x+1]] = 1 # v01
            A[eq, pix2ind[y,x]] = -1 # -v00 

            b[eq] = im[y, x + 1] - im[y, x] # s01 - s00

            eq += 1

    # Gradient in y
    for x in range(im_w):
        for y in range(im_h - 1): # minus one since we access y + 1
            # v10 - v00 = s10 - s00
            A[eq, pix2ind[y+1,x]] = 1 # v10
            A[eq, pix2ind[y, x]] = -1 # v00

            b[eq] = im[y + 1, x] - im[y, x] # s10 - s00
            eq += 1


    # Finally, constrain top left corner pixel intensity 
    # 1*v00 = s00
    A[eq, pix2ind[0, 0]] = 1 
    b[eq] = im[0,0]

    # solve least squares 
    A = csc_matrix(A)
    v = scipy.sparse.linalg.lsqr(A, b, show=True)[0]
    
    # reshape to im size 
    v = v.reshape((im_h, im_w))

    return v


def poisson_blend(fg, mask, bg):
    """
    Poisson Blending.
    :param fg: (H, W, C) source texture / foreground object
    :param mask: (H, W, 1)
    :param bg: (H, W, C) target image / background
    :return: (H, W, C)
    """
    return fg * mask + bg * (1 - mask)


def mixed_blend(fg, mask, bg):
    """EC: Mix gradient of source and target"""
    return fg * mask + bg * (1 - mask)


def color2gray(rgb_image):
    """Naive conversion from an RGB image to a gray image."""
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def mixed_grad_color2gray(rgb_image):
    """EC: Convert an RGB image to gray image using mixed gradients."""
    return np.zeros_like(rgb_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Poisson blending.")
    parser.add_argument("-q", "--question", required=True, choices=["toy", "blend", "mixed", "color2gray"])
    args, _ = parser.parse_known_args()

    # Example script: python proj2_starter.py -q toy
    if args.question == "toy":
        image = imageio.imread('./data/toy_problem.png') / 255.
        image_hat = toy_recon(image)

        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(image_hat, cmap='gray')
        plt.title('Output')
        plt.show()

    # Example script: python proj2_starter.py -q blend -s data/source_01_newsource.png -t data/target_01.jpg -m data/target_01_mask.png
    if args.question == "blend":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = poisson_blend(fg, mask, bg)

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Poisson Blend')
        plt.show()

    if args.question == "mixed":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = mixed_blend(fg, mask, bg)

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Mixed Blend')
        plt.show()

    if args.question == "color2gray":
        parser.add_argument("-s", "--source", required=True)
        args = parser.parse_args()

        rgb_image = imageio.imread(args.source)
        gray_image = color2gray(rgb_image)
        mixed_grad_img = mixed_grad_color2gray(rgb_image)

        plt.subplot(121)
        plt.imshow(gray_image, cmap='gray')
        plt.title('rgb2gray')
        plt.subplot(122)
        plt.imshow(mixed_grad_img, cmap='gray')
        plt.title('mixed gradient')
        plt.show()

    plt.close()
