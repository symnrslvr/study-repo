import cv2
import numpy as np

def find_triangles(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    triangles = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 3:
            triangles.append(approx)

    return triangles, img

def calculate_triangle_side_lengths(vertices, pixel_to_micron):
    vertices_micron = vertices * pixel_to_micron
    side_lengths = []
    for i in range(3):
        j = (i + 1) % 3
        side_length = np.linalg.norm(vertices_micron[i] - vertices_micron[j])
        side_lengths.append((side_length, vertices[i], vertices[j]))
    return side_lengths

def main(image_path):
    triangles, img = find_triangles(image_path)
    height, width, _ = img.shape

    real_width_microns = 26.38
    real_height_microns = 24.92

    pixel_to_micron_x = real_width_microns / width
    pixel_to_micron_y = real_height_microns / height
    pixel_to_micron = np.array([pixel_to_micron_x, pixel_to_micron_y])

    num_triangles = len(triangles)
    print(f"Number of triangles found: {num_triangles}")

    if num_triangles == 0:
        print("No triangles found.")
        return

    max_length = 0
    max_length_info = None
    min_length = float('inf')
    min_length_info = None

    for i, tri in enumerate(triangles):
        side_lengths = calculate_triangle_side_lengths(tri.reshape(3, 2), pixel_to_micron)
        
        for length, pt1, pt2 in side_lengths:
            if length < 2.0:
                midpoint = tuple(((pt1 + pt2) // 2).astype(int))
                cv2.putText(img, f"{length:.3f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.line(img, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0, 255, 0), 2)
                print(f"Triangle {i+1} side length: {length:.3f}")
                
                if length > max_length:
                    max_length = length
                    max_length_info = (length, (pt1, pt2))
                
                if length < min_length:
                    min_length = length
                    min_length_info = (length, (pt1, pt2))

    if max_length_info is not None:
        length, (pt1, pt2) = max_length_info
        midpoint = tuple(((pt1 + pt2) // 2).astype(int))
        cv2.putText(img, f"Longest: {length:.3f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.line(img, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (255, 0, 0), 2)

    if min_length_info is not None:
        length, (pt1, pt2) = min_length_info
        midpoint = tuple(((pt1 + pt2) // 2).astype(int))
        cv2.putText(img, f"Shortest: {length:.3f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.line(img, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0, 255, 255), 2)

    print(f"Longest side length under 2.0: {max_length:.3f}")
    print(f"Shortest side length under 2.0: {min_length:.3f}")

    output_image_path = "stacking_fault/output_with_side_lengths_longest_shortest.jpg"
    cv2.imwrite(output_image_path, img)
    print(f"Output image saved to {output_image_path}")

if __name__ == "__main__":
    image_path = "first/104a.jpeg" 
    main(image_path)

