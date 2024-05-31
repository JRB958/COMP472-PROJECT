#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:39:14 2024

@author: joudbabik

This script processes all images in a specified directory (and its subdirectories) to create aggregated histograms
of pixel intensities for both RGB and grayscale images. It identifies grayscale images stored in RGB format and
handles them accordingly. Histograms are normalized and displayed.
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def mask_background(img):
    """
    Creates a mask to exclude the most frequent background color from histogram calculations.

    Parameters:
    img (PIL.Image): The image from which to create the mask.

    Returns:
    numpy.ndarray: A mask array that is True where the pixel color is not the background.
    """
    # Convert image to numpy array for processing
    img_array = np.array(img)
    
    if img.mode == 'RGB':
        # Find the most frequent color in RGB images and assume it's the background
        colors, counts = np.unique(img_array.reshape(-1, 3), axis=0, return_counts=True)
        background_color = colors[counts.argmax()]
        mask = np.all(img_array != background_color, axis=-1)
    else:
        # Find the most frequent color in grayscale images and assume it's the background
        background_color = np.bincount(img_array.flatten()).argmax()
        mask = img_array != background_color
    
    return mask

def is_grayscale(img):
    """
    Determines if an RGB image is effectively a grayscale image.

    Parameters:
    img (PIL.Image): The image to check.

    Returns:
    bool: True if the image is grayscale, False otherwise.
    """
    if img.mode == 'RGB':
        r, g, b = img.split()
        r_data, g_data, b_data = np.array(r), np.array(g), np.array(b)
        return np.array_equal(r_data, g_data) and np.array_equal(r_data, b_data)
    return img.mode == 'L'

def aggregate_histograms(directory):
    """
    Aggregates histograms for all images in a directory and its subdirectories.

    Parameters:
    directory (str): The path to the directory containing the images.
    """
    aggregated_histograms = {
        'red': np.zeros(256),
        'green': np.zeros(256),
        'blue': np.zeros(256),
        'gray': np.zeros(256)
    }

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                filepath = os.path.join(root, filename)
                img = Image.open(filepath)

                if is_grayscale(img):
                    # Process grayscale image
                    gray_img = img.convert('L')
                    mask = mask_background(gray_img)
                    gray_data = np.array(gray_img)[mask]
                    gray_hist, _ = np.histogram(gray_data, bins=256, range=(0, 255))
                    aggregated_histograms['gray'] += gray_hist
                    print(f"Processing grayscale image: {filename}")
                elif img.mode == 'RGB':
                    # Process RGB image
                    mask = mask_background(img)
                    r, g, b = img.split()
                    r_data, g_data, b_data = np.array(r)[mask], np.array(g)[mask], np.array(b)[mask]
                    r_hist, _ = np.histogram(r_data, bins=256, range=(0, 255))
                    g_hist, _ = np.histogram(g_data, bins=256, range=(0, 255))
                    b_hist, _ = np.histogram(b_data, bins=256, range=(0, 255))
                    aggregated_histograms['red'] += r_hist
                    aggregated_histograms['green'] += g_hist
                    aggregated_histograms['blue'] += b_hist
                    print(f"Processing RGB image: {filename}")

    # Normalize the histograms
    normalize_and_plot(aggregated_histograms)

def normalize_and_plot(aggregated_histograms):
    """
    Normalizes and plots the aggregated histograms.

    Parameters:
    aggregated_histograms (dict): Dictionary of histograms for red, green, blue, and gray.
    """
    total_pixels_rgb = sum(np.sum(hist) for key, hist in aggregated_histograms.items() if key in ['red', 'green', 'blue'])
    total_pixels_gray = np.sum(aggregated_histograms['gray'])

    # Normalize RGB histograms
    if total_pixels_rgb > 0:
        for key in ['red', 'green', 'blue']:
            aggregated_histograms[key] = aggregated_histograms[key] / total_pixels_rgb
        plot_histogram(aggregated_histograms, 'RGB')

    # Normalize grayscale histograms
    if total_pixels_gray > 0:
        aggregated_histograms['gray'] = aggregated_histograms['gray'] / total_pixels_gray
        plot_histogram(aggregated_histograms, 'Gray')

def plot_histogram(aggregated_histograms, mode):
    """
    Plots the aggregated histograms.

    Parameters:
    aggregated_histograms (dict): Dictionary of histograms for red, green, blue, and gray.
    mode (str): The mode of the histogram to plot ('RGB' or 'Gray').
    """
    plt.figure(figsize=(12, 6))
    
    if mode == 'RGB':
        if np.any(aggregated_histograms['red']):
            plt.bar(range(256), aggregated_histograms['red'], color='red', alpha=0.5, label='Red Channel', width=1.0)
        if np.any(aggregated_histograms['green']):
            plt.bar(range(256), aggregated_histograms['green'], color='green', alpha=0.5, label='Green Channel', width=1.0)
        if np.any(aggregated_histograms['blue']):
            plt.bar(range(256), aggregated_histograms['blue'], color='blue', alpha=0.5, label='Blue Channel', width=1.0)
        plt.title('Aggregated RGB Pixel Intensity Histogram for Neutral Class Images')
    elif mode == 'Gray':
        if np.any(aggregated_histograms['gray']):
            plt.bar(range(256), aggregated_histograms['gray'], color='gray', alpha=0.5, label='Grayscale Channel', width=1.0)
        plt.title('Aggregated Grayscale Pixel Intensity Histogram for Neutral Class Images')

    plt.xlabel('Pixel Intensity')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.show()

    # Debugging: Print total pixel counts
    total_pixels_rgb = sum(np.sum(hist) for key, hist in aggregated_histograms.items() if key in ['red', 'green', 'blue'])
    total_pixels_gray = np.sum(aggregated_histograms['gray'])
    print(f"Total RGB pixels: {total_pixels_rgb}")
    print(f"Total Grayscale pixels: {total_pixels_gray}")

# Run the script
directory = input("Enter the path to the directory containing the images: ")
aggregate_histograms(directory)
