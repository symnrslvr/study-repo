# import cv2

# def detect_triangles_and_circles(image_path):
#     image = cv2.imread(image_path)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     triangle_count = 0
#     circle_count = 0

#     for contour in contours:
#         if len(contour) < 5:
#             continue

#         epsilon = 0.01 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)

#         if len(approx) == 3:
#             triangle_count += 1
#             cv2.putText(image, "Triangle", (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#         elif len(approx) > 8 or len(approx) < 4:  # Assuming circles will have more than 8 vertices and less than 4 vertices are triangles
#             circle_count += 1
#             cv2.putText(image, "Circle", (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     print("Triangle count:", triangle_count)
#     print("Circle count:", circle_count)

#     cv2.imshow("Shapes Detected", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# detect_triangles_and_circles("new_file/first/pic.jpeg")
import cv2

# Görüntüyü işleme ve çokgenleri tespit etme fonksiyonu
def detect_polygons(image_path):
    # Görüntüyü oku
    img = cv2.imread(image_path)
    
    # Görüntüyü gri tonlamalı hale getir
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Kenarları belirleme (Canny edge detection)
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    
    # Contour tespiti
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Üçgen ve daire sayacı
    triangle_count = 0
    circle_count = 0
    
    # Her kontur için işlem yap
    for contour in contours:
        # Konturu kaplayan en küçük dörtgeni hesapla
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        
        # Eğer dörtgenin 3 kenarı varsa (üçgen ise)
        if len(approx) < 7:
            triangle_count += 1
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)  # Üçgeni yeşil renkte çiz

        # elif len(approx) >= 5:
        #     circle_count += 1
        #     cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)  # Daireyi kırmızı renkte çiz
        
    # Sonuçları ekranda göster
    cv2.imshow('Polygons', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Üçgen ve daire sayısını döndür
    return triangle_count, circle_count

# Çokgenleri tespit et ve say
triangle_count, circle_count = detect_polygons("new_file/output/31a.jpeg")
print("Triangle Count:", triangle_count)
print("Circle Count:", circle_count)
