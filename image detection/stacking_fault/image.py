import cv2

def detect_polygons(image_path):

    img = cv2.imread(image_path)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    triangle_count = 0
    circle_count = 0
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        
        if len(approx) < 7:
            triangle_count += 1
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)  # Üçgeni yeşil renkte çiz

        # elif len(approx) >= 5:
        #     circle_count += 1
        #     cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)  # Daireyi kırmızı renkte çiz
        
    cv2.imshow('Polygons', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return triangle_count, circle_count

triangle_count, circle_count = detect_polygons("new_file/output/31a.jpeg")
print("Triangle Count:", triangle_count)
print("Circle Count:", circle_count)
