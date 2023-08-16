#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities: Functions/Methods frequently used
Mostly plotting
"""
# import packages
import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2

from restoration_toolbox import *

#######################################################################
# NORMAL PLOTS
#######################################################################
def plot_image(image, title="Image"):
    """ Plot an image, either from a file path (str) or numpy array """
    plt.figure(figsize=(6,6))
    
    if isinstance(image, str):  # If image is a file path, read the image
        image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    
    # plot single image with title
    plt.imshow(image)
    plt.title(title)
    plt.axis('off'); plt.show()

def plot_gray_image(img, title, cmap='viridis'):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off'); plt.show()
    
def plot_histogram(img, color='gray', alpha=0.7, title='Histogram', xlabel='Pixel Intensity', ylabel='Frequency'):
    flat_img = img.flatten()
    plt.hist(flat_img, bins=256, color='gray', alpha=0.7)
    plt.title(title); plt.xlabel(xlabel);plt.ylabel(ylabel);plt.show()
    
def plot_image_list(image_list):
    """ Plot a list of images, which can either be file paths (str) or numpy arrays """
    for i, image in enumerate(image_list):
        plt.figure(figsize=(6,6))
        
        # Check if the image is a file path string
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            
        plt.imshow(image)
        plt.title("Image " + str(i+1))
        plt.axis('off')
        plt.show()
        
def plot_original_and_processed(image_original, image_processed):
    """ Plot an image, either from a file path (str) or numpy array """
    plt.figure(figsize=(6,6))
    
    if isinstance(image_original, str):  # If image is a file path, read the image
        image_original = cv2.imread(image_original)
        image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    
    if isinstance(image_processed, str):  # If image is a file path, read the image
        image_processed = cv2.imread(image_processed)
        image_processed = cv2.cvtColor(image_processed, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # plot original image
    plt.imshow(image_original)
    plt.title("Original")
    plt.axis('off')
    plt.show()
    
    # plot processed image
    plt.imshow(image_processed)
    plt.title("Processed")
    plt.axis('off')
    plt.show()
    


#######################################################################
# IMAGE ANALYSIS - INSPECTION PLOTS
#######################################################################
def visualize_images_and_histograms(image):
    # Calculate features
    ndbi = calculate_ndbi(image)
    intensity = calculate_intensity(image)
    background_index = calculate_background_index(image)
    segmented_background = segment_background(image)

    # Prepare data for plotting
    images = [ndbi, intensity, background_index, segmented_background]
    titles = ["NDBI", "Intensity", "Background Index", "Segmented Background"]

    # Create 2x4 subplot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, ax in enumerate(axes[0]):
        # Plot image
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')

    for i, ax in enumerate(axes[1]):
        # Plot histogram
        ax.hist(images[i].ravel(), bins=256, color='blue', alpha=0.7)
        ax.set_title(titles[i] + ' Histogram')

    # Display the plot
    plt.tight_layout()
    plt.show()
    
def plot_hist_rgb(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Split the image into R,G,B channels
    r, g, b = cv2.split(image_rgb)
    
    fig, ax = plt.subplots()
    # Calculate and plot the histogram for each channel
    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        histogram = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        ax.plot(histogram, color=color)
    
    ax.set_title('RGB Histogram')
    ax.set_xlabel('Bins')
    ax.set_ylabel('# of Pixels')
    plt.show()
    
def plot_image_and_hist_rgb(image_rgb):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display the image
    ax[0].imshow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))  # assuming the image is in BGR format
    ax[0].axis('off')

    # Plot the histogram
    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        histogram = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        ax[1].plot(histogram, color=color)
    
    ax[1].set_title('RGB Histogram')
    ax[1].set_xlabel('Bins')
    ax[1].set_ylabel('# of Pixels')

    plt.tight_layout()
    plt.show()



    


