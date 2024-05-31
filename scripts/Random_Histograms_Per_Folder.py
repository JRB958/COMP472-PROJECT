#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:30:01 2024

@author: joudbabik

This script will randomly pick 15 images from a specified directory (and its subdirectories) and will 
draw the histogram based on the image mode (RGB or L) it will also save the histograms and images in 
another specified directory to be used for reporting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import shutil


def mask_background(img):
    """
    Parameters:
    img (PIL.Image): The image from which to create the mask.
    
    Returns:
    numpy.ndarray: A mask array that is True where the pixel color is not the background.
    
    - Provides a mask (a 2D boolean array that matches the size of the input image).
    - The boolean values in the mask array will inidcate if the corresponding pixels in the input image are background or not. 
    - The most frequent pixel color in the input image is the one considered to be the background color.
    - If a pixel color in the input image matches the background color, its corresponding pixel in the mask will be false.
    """
    #conver the input image into an array (3D if RGB or 2D if Grayscale)
    img_array = np.array(img)
    
    #process the RGB images (3D array)
    if img.mode == 'RGB':
        
        # convert the 3D array in into a 2D array of pixel values 
        # use the np.unique function to get 2 arrays: 
        # - all the colors
        # - the count of each different color
        colors, counts = np.unique(img_array.reshape(-1, 3), axis=0, return_counts=True)
        
        # get the most frequent color and set it as the background color
        # using the counts and colors variables  
        background_color = colors[counts.argmax()]
        
        # compares every pixel in the image to the background color 
        # returns a 2D boolean array 
        mask = np.all(img_array != background_color, axis=-1)
        
    # process when the image is black and white 
    else:
        
        # returns the grayscale color with the highest count in the 
        background_color = np.bincount(img_array.flatten()).argmax()
        
        # generate the 2D boolean array 
        mask = img_array != background_color
    return mask

def is_grayscale(img):
    """
    some black and white images have 3 channels of identical pixel values.
    manually checks if the images is gray scale by manually checking if all 3 pixel values
    are identical for all pixels in an image.
    """
    # process a potentially black and white image with 3 channels of identical pixel values per pixel
    if img.mode == 'RGB':
        # split the image into 3 single-band images
        r, g, b = img.split()
        
        # create an array from each band
        r_data = np.array(r)
        g_data = np.array(g)
        b_data = np.array(b)
        
        #return true for an RGB if all pixel values are the same
        return np.array_equal(r_data, g_data) and np.array_equal(r_data, b_data)
    
    # return true if image mode is already black and white (single channel)
    return img.mode == 'L'

def plot_histograms(image_path, save_dir):
    """
    Parameters: 
        image_path (str): Path of the image to be processed.
        save_dir (str): Path to the folder where the histograms will be saved.
        
    This function plots the histogram of a specific image and saves all the histograms and images to a specified directory. 
    It will mask the image and plot based on the mode of the image, either an RGB or Grayscale histogram, with the use of two helper functions:
        - is_grayscale: Determines if an RGB image is effectively a grayscale image.
        - mask_background: Creates a mask to exclude the most frequent background color from histogram calculations.
    """
    
    # Open the selected image and get its name
    img = Image.open(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Grayscale image processing
    if is_grayscale(img):
        
        # If it's an RGB image with identical channel values, change the mode to 'L' (grayscale)
        gray_img = img.convert('L')
        
        # Get the mask of the image to exclude background pixels
        mask = mask_background(gray_img)
        
        # Create a 1D array of non-background pixel values
        gray_data = np.array(gray_img)[mask]
        
        # Create the grayscale histogram based on the mask filter
        gray_hist, _ = np.histogram(gray_data, bins=256, range=(0, 255))

        # Plot and save the grayscale histogram
        plt.figure(figsize=(12, 6))
        plt.bar(range(256), gray_hist, color='gray', alpha=0.5, label='Grayscale Channel', width=1.0)
        plt.title(f'Pixel Intensity Histogram (Grayscale) - {image_name}')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{image_name}_gray_histogram.png'))
        plt.close()
        
    # RGB image processing
    elif img.mode == 'RGB':
        # Get the mask of the image to exclude background pixels
        mask = mask_background(img)
        
        # Split the image into its R, G, B channels
        r, g, b = img.split()
        
        # Create 1D arrays of non-background pixel values for each channel
        r_data = np.array(r)[mask]
        g_data = np.array(g)[mask]
        b_data = np.array(b)[mask]
        
        # Create histograms for each channel based on the mask filter
        r_hist, _ = np.histogram(r_data, bins=256, range=(0, 255))
        g_hist, _ = np.histogram(g_data, bins=256, range=(0, 255))
        b_hist, _ = np.histogram(b_data, bins=256, range=(0, 255))

        # Plot and save the RGB histogram
        plt.figure(figsize=(12, 6))
        plt.bar(range(256), r_hist, color='red', alpha=0.5, label='Red Channel', width=1.0)
        plt.bar(range(256), g_hist, color='green', alpha=0.5, label='Green Channel', width=1.0)
        plt.bar(range(256), b_hist, color='blue', alpha=0.5, label='Blue Channel', width=1.0)
        plt.title(f'Pixel Intensity Histogram (RGB) - {image_name}')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{image_name}_rgb_histogram.png'))
        plt.close()

def process_images(directory, save_dir):
    """
    Processes a directory of images by randomly selecting 15 images, plotting their histograms,
    and saving the histograms and images to a specified directory.

    Parameters:
    directory (str): The path to the directory containing the images.
    save_dir (str): The path to the directory where the histograms and images will be saved.

    Returns:
    None
    """
    images = []  # List to store the paths of all valid images found in the directory

    # Traverse the directory and its subdirectories to find image files
    for root, _, files in os.walk(directory):
        for filename in files:
            # Check if the file is an image based on its extension
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                images.append(os.path.join(root, filename))

    # Check if there are at least 15 images in the directory
    if len(images) < 15:
        print(f"Directory doesn't exist or Not enough images in the directory. Found only {len(images)} images.")
        return

    # Randomly select 15 images from the list
    selected_images = random.sample(images, 15)

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Process each selected image
    for image_path in selected_images:
        # Plot and save the histogram of the image
        plot_histograms(image_path, save_dir)
        
        # Copy the original image to the save directory
        shutil.copy(image_path, os.path.join(save_dir, os.path.basename(image_path)))

    print(f"Processed and saved histograms and images for {len(selected_images)} images.")

# **********************************************
# Example usage - MAKE SURE TO ENTER DIRECTORIES
# **********************************************

# Replace with your directory path
directory = ''  
# Directory to save histograms
save_dir = ''  

process_images(directory, save_dir)

