# import cv2
# import numpy as np
# import csv

# def find_longest_line(contour):
#     max_length = 0
#     max_line = None
#     for i in range(len(contour)):
#         for j in range(i+1, len(contour)):
#             length = np.linalg.norm(contour[i] - contour[j])
#             if length > max_length:
#                 max_length = length
#                 max_line = (contour[i], contour[j])
#     return max_line, max_length

# def pixels_to_microns(pixels, image_width_mm, image_height_mm, img_width_px, img_height_px):
#     width_ratio = image_width_mm / img_width_px
#     height_ratio = image_height_mm / img_height_px
#     return pixels * max(width_ratio, height_ratio)

# def find_grain_sizes(image_path, line_color_lower, line_color_upper, grain_size_limit, longest_line_limit, image_width_mm, image_height_mm):

#     img = cv2.imread(image_path)

#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     mask = cv2.inRange(hsv, line_color_lower, line_color_upper)

#     kernel = np.ones((7, 7), np.uint8)  
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#     grain_sizes = []
#     line_lengths = []  
#     grain_data = []  

#     for idx, contour in enumerate(contours):
#         grain_size_pixels = cv2.arcLength(contour, True)
#         grain_size_microns = pixels_to_microns(grain_size_pixels, image_width_mm, image_height_mm, img.shape[1], img.shape[0])

#         if grain_size_microns > grain_size_limit:
#             continue

#         cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

#         M = cv2.moments(contour)
#         if M["m00"] != 0:
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#             cv2.putText(img, f"{grain_size_microns:.2f}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         longest_line, line_length = find_longest_line(contour)
#         if longest_line is not None and line_length <= longest_line_limit:
#             line_length_microns = pixels_to_microns(line_length, image_width_mm, image_height_mm, img.shape[1], img.shape[0])
#             line_lengths.append(line_length_microns)
#             cv2.line(img, tuple(longest_line[0][0]), tuple(longest_line[1][0]), (255, 0, 0), 2)
#             cv2.putText(img, f"{line_length_microns:.2f}", tuple(longest_line[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#         grain_sizes.append(grain_size_microns)
#         grain_data.append([idx + 1, grain_size_microns, line_length_microns])
#         print(f"Grain Size: {grain_size_microns:.2f} mikron")  # Print grain size
#         print(f"Longest Line Length: {line_length_microns:.2f} mikron")  # Print longest line length

 
#     if line_lengths:
#         max_line_length = max(line_lengths)
#         min_line_length = min(line_lengths)
#         max_length_pos = line_lengths.index(max_line_length)
#         min_length_pos = line_lengths.index(min_line_length)
#         max_contour = contours[max_length_pos]
#         min_contour = contours[min_length_pos]

#         max_line, _ = find_longest_line(max_contour)
#         if max_line is not None:
#             cv2.line(img, tuple(max_line[0][0]), tuple(max_line[1][0]), (0, 255, 255), 2)
#             cv2.putText(img, f"Max: {max_line_length:.2f}", tuple(max_line[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#         min_line, _ = find_longest_line(min_contour)
#         if min_line is not None:
#             cv2.line(img, tuple(min_line[0][0]), tuple(min_line[1][0]), (255, 255, 255), 2)  # Beyaz renkte çiz
#             cv2.putText(img, f"Min: {min_line_length:.2f}", tuple(min_line[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Beyaz renkte yazı

#         print(f"Maximum Length: {max_line_length:.2f}")
#         print(f"Minimum Length: {min_line_length:.2f}")
#     else:
#         print("There is no length.")

#     with open('grain_size/grain_sizes.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Grain Size Number", "Grain Size (microns)", "Longest Line Length (microns)"])
#         writer.writerows(grain_data)

#     return img

# image_path = "first/104-gs.jpeg"  # Path to your EBSD image
# line_color_lower = (20, 100, 100)  # Lower HSV threshold for yellow color
# line_color_upper = (30, 255, 255)  # Upper HSV threshold for yellow color
# grain_size_limit = 2000  # Grain size limit in microns
# longest_line_limit = 600  # Longest line length limit in pixels
# image_width_mm = 63.40  # Width of the image in millimeters
# image_height_mm = 47.53  # Height of the image in millimeters

# result_image = find_grain_sizes(image_path, line_color_lower, line_color_upper, grain_size_limit, longest_line_limit, image_width_mm, image_height_mm)
# cv2.imwrite("grain_size/result_with_grain_sizes.jpg", result_image)

# print("Result image saved as result_with_grain_sizes.jpg")
# print("Grain size data saved as grain_sizes.csv")

import cv2
import numpy as np
import csv

def find_longest_line(contour):
    max_length = 0
    max_line = None
    for i in range(len(contour)):
        for j in range(i+1, len(contour)):
            length = np.linalg.norm(contour[i] - contour[j])
            if length > max_length:
                max_length = length
                max_line = (contour[i], contour[j])
    return max_line, max_length

def pixels_to_microns(pixels, image_width_mm, image_height_mm, img_width_px, img_height_px):
    width_ratio = image_width_mm / img_width_px
    height_ratio = image_height_mm / img_height_px
    return pixels * max(width_ratio, height_ratio)

def find_grain_sizes(image_path, line_color_lower, line_color_upper, grain_size_limit, longest_line_limit, image_width_mm, image_height_mm):

    img = cv2.imread(image_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, line_color_lower, line_color_upper)

    kernel = np.ones((7, 7), np.uint8)  
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    grain_sizes = []
    line_lengths = []  
    grain_data = []  

    for idx, contour in enumerate(contours):
        grain_size_pixels = cv2.arcLength(contour, True)
        grain_size_microns = pixels_to_microns(grain_size_pixels, image_width_mm, image_height_mm, img.shape[1], img.shape[0])

        if grain_size_microns > grain_size_limit:
            continue

        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        longest_line, line_length = find_longest_line(contour)
        if longest_line is not None and line_length <= longest_line_limit:
            line_length_microns = pixels_to_microns(line_length, image_width_mm, image_height_mm, img.shape[1], img.shape[0])
            line_lengths.append(line_length_microns)
            cv2.line(img, tuple(longest_line[0][0]), tuple(longest_line[1][0]), (255, 0, 0), 2)
            cv2.putText(img, f"{line_length_microns:.2f}", tuple(longest_line[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 0, 0), 3)

        grain_sizes.append(grain_size_microns)
        grain_data.append([idx + 1, grain_size_microns, line_length_microns])
        print(f"{grain_size_microns:.2f} mikron")  # Print grain size
        print(f"Longest Line Length: {line_length_microns:.2f} mikron")  # Print longest line length

 
    if line_lengths:
        max_line_length = max(line_lengths)
        min_line_length = min(line_lengths)
        max_length_pos = line_lengths.index(max_line_length)
        min_length_pos = line_lengths.index(min_line_length)
        max_contour = contours[max_length_pos]
        min_contour = contours[min_length_pos]

        max_line, _ = find_longest_line(max_contour)
        if max_line is not None:
            cv2.line(img, tuple(max_line[0][0]), tuple(max_line[1][0]), (0, 255, 255), 2)
            cv2.putText(img, f"Max: {max_line_length:.2f}", tuple(max_line[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 255, 255), 3)

        min_line, _ = find_longest_line(min_contour)
        if min_line is not None:
            cv2.line(img, tuple(min_line[0][0]), tuple(min_line[1][0]), (255, 255, 255), 2)  # Beyaz renkte çiz
            cv2.putText(img, f"Min: {min_line_length:.2f}", tuple(min_line[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 3)  # Beyaz renkte yazı

        print(f"Maximum Length: {max_line_length:.2f}")
        print(f"Minimum Length: {min_line_length:.2f}")
    else:
        print("There is no length.")

    with open('grain_size/grain_sizes.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Grain Size Number", "Grain Size (microns)", "Longest Line Length (microns)"])
        writer.writerows(grain_data)

    return img

image_path = "first/104-gs.jpeg"  # Path to your EBSD image
line_color_lower = (20, 100, 100)  # Lower HSV threshold for yellow color
line_color_upper = (30, 255, 255)  # Upper HSV threshold for yellow color
grain_size_limit = 2000  # Grain size limit in microns
longest_line_limit = 600  # Longest line length limit in pixels
image_width_mm = 63.40  # Width of the image in millimeters
image_height_mm = 47.53  # Height of the image in millimeters

result_image = find_grain_sizes(image_path, line_color_lower, line_color_upper, grain_size_limit, longest_line_limit, image_width_mm, image_height_mm)
cv2.imwrite("grain_size/result_with_grain_sizes1.jpg", result_image)

print("Result image saved as result_with_grain_sizes1.jpg")
print("Grain size data saved as grain_sizes.csv")
