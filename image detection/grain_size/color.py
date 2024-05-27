import cv2
import numpy as np

def calculate_color_ratios(image_path):
    img = cv2.imread(image_path)
    
    total_pixels = img.shape[0] * img.shape[1]
    
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 100, 100])
    
    lower_green = np.array([0, 100, 0])
    upper_green = np.array([100, 255, 100])
    
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([100, 100, 255])
    
    blue_mask = cv2.inRange(img, lower_blue, upper_blue)
    green_mask = cv2.inRange(img, lower_green, upper_green)
    red_mask = cv2.inRange(img, lower_red, upper_red)
    
    blue_pixels = np.sum(blue_mask)
    green_pixels = np.sum(green_mask)
    red_pixels = np.sum(red_mask)
    sum = blue_pixels+green_pixels+red_pixels

    blue_ratio = blue_pixels / sum
    green_ratio = green_pixels / sum
    red_ratio = red_pixels / sum
    
    return blue_ratio, green_ratio, red_ratio


samples = ["101e.jpg","102.jpg","103.jpg","104.jpg","105.jpg","106.jpg"]
for sample in samples:
    blue_ratio, green_ratio, red_ratio = calculate_color_ratios("first/"+sample)
    print("Sample Name:",sample)
    print("Blue Ratio:", blue_ratio)
    print("Green Ratio:", green_ratio)
    print("Red Ratio:", red_ratio)

