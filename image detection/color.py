# import cv2
# import numpy as np

# # Görüntüdeki renk oranlarını ve bölgelerini hesapla
# def calculate_color_ratio_and_show_regions(image_path):
#     # Görüntüyü oku
#     img = cv2.imread(image_path)
    
#     # Görüntüyü boyutları
#     total_pixels = img.shape[0] * img.shape[1]
    
#     # Renk kanallarını ayır
#     blue_channel = img[:,:,0]
#     green_channel = img[:,:,1]
#     red_channel = img[:,:,2]
    
#     # Renklerin ortalamasını hesapla
#     blue_mean = blue_channel.mean() / 255
#     green_mean = green_channel.mean() / 255
#     red_mean = red_channel.mean() / 255
    
#     # Renk oranlarını döndür
#     print("Blue Ratio:", blue_mean)
#     print("Green Ratio:", green_mean)
#     print("Red Ratio:", red_mean)
    
#     # Renk alanlarını belirle
#     blue_area = np.zeros_like(img)
#     blue_area[:,:,0] = blue_channel
#     blue_area[:,:,1] = blue_channel
#     blue_area[:,:,2] = blue_channel
    
#     green_area = np.zeros_like(img)
#     green_area[:,:,1] = green_channel
#     green_area[:,:,2] = green_channel
    
#     red_area = np.zeros_like(img)
#     red_area[:,:,2] = red_channel
    
#     # Renk alanlarını ekranda göster
#     cv2.imshow("Blue Regions", blue_area)
#     cv2.imshow("Green Regions", green_area)
#     cv2.imshow("Red Regions", red_area)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Görüntüdeki renk oranlarını hesapla ve renk bölgelerini göster
# calculate_color_ratio_and_show_regions("new_file/first/color6.png")


import cv2
import numpy as np

# Görüntüdeki renk oranlarını hesapla
def calculate_color_ratios(image_path):
    # Görüntüyü oku
    img = cv2.imread(image_path)
    
    # Görüntüyü boyutları
    total_pixels = img.shape[0] * img.shape[1]
    
    # Mavi rengin aralığını tanımla (BGR formatında)
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 100, 100])
    
    # Yeşil rengin aralığını tanımla (BGR formatında)
    lower_green = np.array([0, 100, 0])
    upper_green = np.array([100, 255, 100])
    
    # Kırmızı rengin aralığını tanımla (BGR formatında)
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([100, 100, 255])
    
    # Renk maskelerini oluştur
    blue_mask = cv2.inRange(img, lower_blue, upper_blue)
    green_mask = cv2.inRange(img, lower_green, upper_green)
    red_mask = cv2.inRange(img, lower_red, upper_red)
    
    # Renklerin toplam piksel sayısını hesapla
    blue_pixels = np.sum(blue_mask)
    green_pixels = np.sum(green_mask)
    red_pixels = np.sum(red_mask)
    sum = blue_pixels+green_pixels+red_pixels
    print(sum)
    # Renk oranlarını hesapla
    blue_ratio = blue_pixels / sum
    green_ratio = green_pixels / sum
    red_ratio = red_pixels / sum
    
    return blue_ratio, green_ratio, red_ratio


# Görüntüdeki renk oranlarını hesapla ve yazdır
blue_ratio, green_ratio, red_ratio = calculate_color_ratios("image_detection/first/I07-106e-r.jpeg")

print("Blue Ratio:", blue_ratio)
print("Green Ratio:", green_ratio)
print("Red Ratio:", red_ratio)

