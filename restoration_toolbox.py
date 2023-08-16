#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROV Preprocessing functions

Steps:
1) Noise reduction - smoothing
2) UDCP(Underwater Dark Channel Prior) - underwater color restoration
3) White balancing
4) Contrast enhancement - CLAHE on LAB (doesn't change color balance and hue)
5) Color correction - CLAHE on RGB (obs; can alter color balance and hue), or
6) Edge enhancement - Sobel/Canny
7) Sharpening - Unsharp mask

"""
# import libraries
import os
import numpy as np
import cv2
import time

from plot_toolbox import *

#######################################################################
# NOISE REDUCTION - SMOOTHING
#######################################################################
# Bilateral filter and Non-local means filter
def apply_bilateral_filter(img, diameter=15, sigma_color=20, sigma_space=70):
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
# (img, diameter=15, sigma_color=20, sigma_space=70) # save param

# obs: slow algorithm
def apply_non_local_means_denoising(img, h=4, h_color=10, template_window_size=9, search_window_size=21):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h_color, template_window_size, search_window_size)
# (img, h=4, h_color=10, template_window_size=9, search_window_size=21) # save param
### == So similar with these params than can't distinguish them == ###

#######################################################################
# COLOR CORRECTION
#######################################################################
# UDCP - Underwater Dark Channel Prior
def guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r,r))
    mean_Ip = cv2.boxFilter(I*p, cv2.CV_64F, (r,r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I*I, cv2.CV_64F, (r,r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r,r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r,r))
    q = mean_a * I + mean_b
    return q

def UDCP(img, omega=0.95, t0=0.1, r=15, eps=1e-3, radius=7):
    # Convert to float
    src = img.astype(np.float32) / 255
    I = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Dark Channel
    J_dark = cv2.min(cv2.min(src[:,:,0], src[:,:,1]), src[:,:,2])
    J_dark = cv2.blur(J_dark, (radius, radius))

    # Atmospheric light
    A = np.zeros([1, 1, 3])
    for i in range(3):
        bool_idx = (J_dark == np.max(J_dark))
        A[0,0,i] = np.max(src[bool_idx,i])

    # Transmission map
    trans = 1 - omega * J_dark[:,:,np.newaxis] / A

    # Guided filter for each channel
    trans_guide = np.zeros_like(trans)
    for i in range(3):
        trans_guide[:,:,i] = guided_filter(I, trans[:,:,i], r, eps)

    trans_guide = np.clip(trans_guide, t0, 1)

    # Recover
    result = np.empty_like(src)
    for i in range(3):
        result[:,:,i] = (src[:,:,i] - A[0,0,i]) / trans_guide[:,:,i] + A[0,0,i]

    result = cv2.convertScaleAbs(result * 255)
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

#######################################################################
# WHITE BALANCING
#######################################################################
# White Balance - "Gray World Assumption"
def white_balance(img, gamma=0.85):
    result = cv2.xphoto.createSimpleWB().balanceWhite(img)
    result = result ** gamma  # Apply gamma correction
    result_normalized = (result - np.min(result)) / (np.max(result) - np.min(result))  # Normalize the image
    
    # Convert back to 8-bit range and uint8 datatype
    result_normalized = (result_normalized * 255).astype('uint8')
    return result_normalized

# Option to blend with original image = Best so far
def white_balance_blend(img, gamma=0.85, percent_to_ignore=0.05, blend_factor=0.5):
    assert 0 <= blend_factor <= 1, "blend_factor should be between 0 and 1"
    
    result = cv2.xphoto.createSimpleWB()
    
    # Set the percent of top/bottom values to ignore
    result.setP(percent_to_ignore)
    
    result = result.balanceWhite(img)
    
    result = result ** gamma  # Apply gamma correction
    result_normalized = (result - np.min(result)) / (np.max(result) - np.min(result))  # Normalize the image
    
    # Convert back to 8-bit range and uint8 datatype
    result_normalized = (result_normalized * 255).astype('uint8')
    
    # Blend original image with white balanced image
    result_blended = cv2.addWeighted(img, 1 - blend_factor, result_normalized, blend_factor, 0)
    return result_blended

#######################################################################
# CONTRAST ENHANCEMENT - CLAHE on LAB space
#######################################################################
def preprocess_clahe(img, clipLimit=1.0, tileGridSize=(8,8)):
    """ Preprocess the image
    Good parameters:
        clahe = cv2.createCLAHE(clipLimit=0.55, tileGridSize=(20,20)) """
    # Convert to LAB color space
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])

    # Convert back to RGB color space
    img_corrected = cv2.cvtColor(img_lab, cv2.COLOR_Lab2RGB)
    return img_corrected

#######################################################################
# COLOR BALANCE - CLAHE on RGB-channels
#######################################################################
def color_balance_and_saturation_enhancement(img, saturation_factor=1.2, clip_limit=2.0, tile_grid_size=(8,8)):
    # Split the image into its respective RGB channels
    channels = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    out_channels = []
    for channel in channels:
        # Apply CLAHE to each channel
        channel = clahe.apply(channel)
        out_channels.append(channel)

    # Merge the channels
    img_equalized = cv2.merge(out_channels)

    # Convert the image to HSV
    img_hsv = cv2.cvtColor(img_equalized, cv2.COLOR_BGR2HSV)

    # Split the image into H, S, V channels
    h, s, v = cv2.split(img_hsv)

    # Increase the saturation
    s = s * saturation_factor
    s = np.clip(s, 0, 255).astype(np.uint8)  # Clip to [0, 255] and convert back to 8-bit integer

    # Merge the H, S, V channels
    img_hsv_enhanced = cv2.merge([h, s, v])

    # Convert the image back to BGR
    img_enhanced = cv2.cvtColor(img_hsv_enhanced, cv2.COLOR_HSV2BGR)
    return img_enhanced

#######################################################################
# EDGE ENHANCEMENT
#######################################################################
#  Edge Enhancement with Sobel operator (a popular edge detection operator)
def edge_enhancement_sobel(img): 
    # Load image
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Sobel Operator
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    
    # Combine Sobel X and Y outputs
    combined_sobel = cv2.sqrt(cv2.addWeighted(sobel_x**2, 0.5, sobel_y**2, 0.5, 0))

    # Normalize to 8-bit
    sobel_8u = cv2.normalize(combined_sobel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert grayscale edge map to BGR
    edge_map = cv2.cvtColor(sobel_8u, cv2.COLOR_GRAY2BGR)

    # Add the edge map to the original image
    enhanced = cv2.addWeighted(img, 1, edge_map, 1, 0)

    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return enhanced_rgb

# Edge Enhancement with Canny (another popular edge detection operator)
## Canny, a bit more subtle than Sobel above
def edge_enhancement_canny(img, edge_intensity=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Adjust the intensity of the edges
    edges = (edges * edge_intensity).astype(np.uint8)

    # Convert grayscale edge map to BGR
    edge_map = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Add the edge map to the original image
    enhanced = cv2.addWeighted(img, 1, edge_map, 1, 0)
    return enhanced

#######################################################################
# NOISE REDUCTION + SHARPENING
#######################################################################
def unsharp_mask(img, sigma, strength):
    """Apply unsharp mask to input image
    unsharp_mask(image_path, 1.0, 3.0) save param"""
    # Ensure the input image is in floating point format
    image_float = img.astype(np.float64)

    # Blur the image
    blurred = cv2.GaussianBlur(image_float, (0, 0), sigma)

    # Compute the mask as the difference between original and blurred image
    mask = image_float - blurred

    # Modify the mask to ensure values are within valid range
    mask = np.clip(strength * mask, -255, 255)

    # Add the mask to the original image to create a sharpened image
    sharpened = image_float + mask
    sharpened = np.clip(sharpened, 0, 255)  # ensure valid pixel value range

    # Convert back to 8-bit integer for display
    sharpened = sharpened.astype(np.uint8)
    # sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return sharpened


#######################################################################
# FULL PROCESSING PROCEDURE - STEPS 2-7
#######################################################################
# SPECIFIC FOR 2018 SD VIDEOS
def full_preprocessing2_7_2018sd(image_rgb, enable_udcp=True, enable_wb=True, enable_clahe1=True,
                       enable_clahe2=True, enable_canny=True, enable_unsharp=True):
    if enable_udcp:
        image_rgb = UDCP(image_rgb, omega=0.50, t0=0.7, r=15, eps=1e-3, radius=25)
    if enable_wb:
        image_rgb = white_balance_blend(image_rgb, gamma=0.95, blend_factor=0.25)
    if enable_clahe1:
        image_rgb = preprocess_clahe(image_rgb, clipLimit=0.2, tileGridSize=(8,8))
    if enable_clahe2:
        image_rgb = color_balance_and_saturation_enhancement(image_rgb, saturation_factor=1.0, clip_limit=0.2, tile_grid_size=(8,8))
    if enable_canny:
        image_rgb = edge_enhancement_canny(image_rgb, edge_intensity=0.00005)
    if enable_unsharp:
        image_rgb = unsharp_mask(image_rgb, sigma=1.0, strength=1.0)
    
    return image_rgb.astype('uint8')

#######################################################################
# FULL PROCESSING PROCEDURE - STEPS 3-7
#######################################################################
# SPECIFIC FOR 2019 HD VIDEOS
def full_preprocessing3_7_2019hd(image_rgb, enable_udcp=False, enable_wb=True, enable_clahe1=True,
                       enable_clahe2=True, enable_canny=True, enable_unsharp=True):
    if enable_udcp:
        image_rgb = UDCP(image_rgb, omega=0.50, t0=0.7, r=15, eps=1e-3, radius=25)
    if enable_wb:
        image_rgb = white_balance_blend(image_rgb, gamma=0.85, blend_factor=0.5)
    if enable_clahe1:
        image_rgb = preprocess_clahe(image_rgb, clipLimit=0.2, tileGridSize=(8,8))
    if enable_clahe2:
        image_rgb = color_balance_and_saturation_enhancement(image_rgb, saturation_factor=1.1, clip_limit=0.2, tile_grid_size=(8,8))
    if enable_canny:
        image_rgb = edge_enhancement_canny(image_rgb, edge_intensity=0.05)
    if enable_unsharp:
        image_rgb = unsharp_mask(image_rgb, sigma=1.0, strength=3.0)
    
    return image_rgb.astype('uint8')

#######################################################################
# STEPS 2-7  - WITH RUNTIME OVERVIEW
#######################################################################
def full_preprocessing_timed(image_rgb, enable_udcp=True, enable_wb=True, enable_clahe1=True, enable_clahe2=True, enable_canny=True, enable_unsharp=True):
    if enable_udcp:
        start = time.time()
        image_rgb = UDCP(image_rgb, omega=0.50, t0=0.7, r=15, eps=1e-3, radius=25)
        end = time.time()
        print("1) UDCP:                                    ", (end - start)*10**3, "milliseconds")
    
    if enable_wb:
        start = time.time()
        image_rgb = white_balance_blend(image_rgb, gamma=0.85, blend_factor=0.5)
        end = time.time()
        print("2) White Balance Blend:                     " , (end - start)*10**3, "milliseconds")

    if enable_clahe1:
        start = time.time()
        image_rgb = preprocess_clahe(image_rgb, clipLimit=0.2, tileGridSize=(8,8))
        end = time.time()
        print("3) CLAHE1:                                  ", (end - start)*10**3, "milliseconds")
    
    if enable_clahe2:
        start = time.time()
        image_rgb = color_balance_and_saturation_enhancement(image_rgb, saturation_factor=1.1, clip_limit=0.2, tile_grid_size=(8,8))
        end = time.time()
        print("4) Color Balance and Saturation Enhancement:", (end - start)*10**3, "milliseconds")
    
    if enable_canny:
        start = time.time()
        image_rgb = edge_enhancement_canny(image_rgb, edge_intensity=0.05)
        end = time.time()
        print("5) Canny Edge Enhancement:                  ", (end - start)*10**3, "milliseconds")
    
    if enable_unsharp:
        start = time.time()
        image_rgb = unsharp_mask(image_rgb, sigma=1.0, strength=3.0)
        end = time.time()
        print("6) Unsharp Mask:                            ", (end - start)*10**3, "milliseconds")

    return image_rgb.astype('uint8')

#######################################################################
# WRITE IMAGE TO DISK
#######################################################################
def save_image(image, file_path="my_output_image.jpg"):
    # Convert from BGR to RGB
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    
    # Save the image to the specified file_path
    cv2.imwrite(file_path, image)

#######################################################################
# FULL PROCESSING OF FOLDER
#######################################################################
def preprocess_folder(input_folder, output_folder):
    
    # Ensure the output directory exists, here same as input_folder
    os.makedirs(input_folder, exist_ok=True)

    # Loop over all files in the input directory
    for filename in os.listdir(input_folder):
        # Check if the file is an image (you may need to adjust this to match your image types)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Construct the full file path
            input_path = os.path.join(input_folder, filename)
            image_bgr = cv2.imread(input_path) # BGR by open-cv default
            # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            
            ############## PERFORM PROCESSING ##########################
            processed_image = full_preprocessing3_7_2019hd(image_bgr, enable_udcp=False, enable_wb=True, enable_clahe1=True,
                                    enable_clahe2=True, enable_canny=True, enable_unsharp=True)
            ############################################################
            
            # Add '_processed' to the output filename
            base, extension = os.path.splitext(filename)
            output_filename = f"{base}_processed{extension}"
            output_path = os.path.join(output_folder, output_filename)
            
            # write to disc
            save_image(image=processed_image, 
                        file_path=output_path)
    print("FINISHED PROCESSING OF FOLDER: {}".format(input_folder))
    

#######################################################################
# FULL PROCESSING OF VIDEO
#######################################################################
def process_video(video_for_processing, output_vid_name, processing_func, video_codec):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_for_processing)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    # MP4 is lossless, but can take up much disk space
    # H.264 video codec provides good video quality and efficient compression (Preferred)
    fourcc = cv2.VideoWriter_fourcc(*video_codec) # H264 / mp4v
    # H.264 compromise between quality and disk space
    # MP4 best quality, highest disk space
    
    out = cv2.VideoWriter(output_vid_name, 
                          fourcc, 
                          cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Loop over frames from the video file stream
    while cap.isOpened():
        # Grab the current frame
        ret, frame = cap.read()

        # If we grabbed a frame process it
        if ret:
            # Perform processing steps on the frame
            processed_frame = processing_func(frame) 

            # Write the processed frame to the new video
            out.write(processed_frame)
            
        # If we did not grab a frame then we have reached the end of the video
        else:
            break

    # Release the video file pointers
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
