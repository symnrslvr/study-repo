# import cv2
# import numpy as np

# def preprocess_image(image_path):
#     # Load the image
#     img = cv2.imread(image_path)
#     return img

# def find_grains(mask):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def calculate_areas(contours, pixel_to_micron):
#     areas = []
#     for contour in contours:
#         area = cv2.contourArea(contour) * (pixel_to_micron ** 2)  # Convert pixel area to micron area
#         areas.append(area)
#     return areas

# def main(image_path):
#     # Load the image to get its dimensions
#     img = preprocess_image(image_path)
#     height, width, _ = img.shape

#     # Define the real-world dimensions of the image in microns
#     real_width_microns = 39.5  # Replace with your image's real width in microns
#     real_height_microns = 31.6  # Replace with your image's real height in microns

#     # Calculate pixel to micron ratio
#     pixel_to_micron_x = real_width_microns / width
#     pixel_to_micron_y = real_height_microns / height
#     pixel_to_micron = np.mean([pixel_to_micron_x, pixel_to_micron_y])

#     # Convert the image to HSV color space
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
#     # Define the range for yellow color in HSV
#     lower_yellow = np.array([20, 100, 100])
#     upper_yellow = np.array([30, 255, 255])
    
#     # Create a mask for yellow color
#     mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
#     # Invert the mask to get the non-yellow regions
#     mask_inv = cv2.bitwise_not(mask)
    
#     # Find contours in the inverted mask
#     contours = find_grains(mask_inv)
#     areas = calculate_areas(contours, pixel_to_micron)
    
#     max_area = max(areas) if areas else 0
#     min_area = min(areas) if areas else 0

#     print(f"Max grain area: {max_area:.2f} µm²")
#     print(f"Min grain area: {min_area:.2f} µm²")

#     # Draw the grains and annotate the max and min areas
#     for contour in contours:
#         area = cv2.contourArea(contour) * (pixel_to_micron ** 2)
#         if area == max_area:
#             color = (0, 0, 0)  # Black for max area
#             label = f"Max: {area:.2f} µm²"
#         elif area == min_area:
#             color = (0, 0, 0)  # Black for min area
#             label = f"Min: {area:.2f} µm²"
#         else:
#             continue  # Skip other contours
#         cv2.drawContours(img, [contour], -1, color, 2)
#         M = cv2.moments(contour)
#         if M["m00"] != 0:
#             cx = int(M["m10"] / M["m00"])
#             cy = int(M["m01"] / M["m00"])
#             cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Save the image with annotations
#     output_image_path = "output_with_grains.jpg"
#     cv2.imwrite(output_image_path, img)
#     print(f"Output image saved to {output_image_path}")



# if __name__ == "__main__":
#     image_path = "first/I07-104e-gb.jpeg"  # Replace with your image path
#     main(image_path)

import cv2
import numpy as np

def find_internal_areas_and_mark(image_path, output_path, pixel_width_microns, pixel_height_microns):
    # Görüntüyü yükle
    image = cv2.imread(image_path)

    # BGR'dan HSV'ye dönüşüm yap
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Sarı renk için maske oluştur
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Maskeyi iyileştir
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    
    # Maskeyi kullanarak konturları bul (Sadece en dıştaki konturlar)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    areas_microns = []
    # Alanları hesapla ve göster
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Çok küçük alanları filtrele
            x, y, w, h = cv2.boundingRect(contour)
            # Çizgilerin içini doldur
            cv2.drawContours(image, [contour], -1, (0, 255, 0), thickness=2)
            roi = mask[y:y+h, x:x+w]
            internal_area_pixels = np.sum(roi == 255) / 255  # Beyaz alanın toplam piksel sayısını hesapla
            internal_area_microns = internal_area_pixels * pixel_width_microns * pixel_height_microns
            areas_microns.append(internal_area_microns)
            cv2.putText(image, f"{internal_area_microns:.2f} um^2", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    if areas_microns:
        max_area = max(areas_microns)
        min_area = min(areas_microns)
        print(f"Maksimum Alan: {max_area:.2f} mikron^2")
        print(f"Minimum Alan: {min_area:.2f} mikron^2")
    else:
        print("Herhangi bir alan bulunamadı.")
    
    # İşlenmiş görüntüyü kaydet
    cv2.imwrite(output_path, image)
    print(f"Görüntü başarıyla kaydedildi: {output_path}")

# Fonksiyonu çağır
find_internal_areas_and_mark('first/new.jpeg', 'output_marked_areas.jpeg', 63.40, 47.53)  # Giriş ve çıkış dosya yollarınızı güncelleyin

