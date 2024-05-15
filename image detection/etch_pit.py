import cv2
import numpy as np

def find_triangles(image_path):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    triangles = []
    for contour in contours:
        # Approximate the contour
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the approximated contour has 3 vertices, it's a triangle
        if len(approx) == 3:
            triangles.append(approx)

    return triangles, img

def calculate_triangle_area(vertices, pixel_to_micron):
    # Convert pixel coordinates to microns
    vertices_micron = vertices * pixel_to_micron
    a = np.linalg.norm(vertices_micron[0] - vertices_micron[1])
    b = np.linalg.norm(vertices_micron[1] - vertices_micron[2])
    c = np.linalg.norm(vertices_micron[2] - vertices_micron[0])
    s = (a + b + c) / 2
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    return area

def main(image_path):
    # Load the image to get its dimensions
    triangles, img = find_triangles(image_path)
    height, width, _ = img.shape

    # Define the real-world dimensions of the image in microns
    real_width_microns = 39.5
    real_height_microns = 31.6

    # Calculate pixel to micron ratio
    pixel_to_micron_x = real_width_microns / width
    pixel_to_micron_y = real_height_microns / height
    pixel_to_micron = np.array([pixel_to_micron_x, pixel_to_micron_y])

    num_triangles = len(triangles)
    print(f"Number of triangles found: {num_triangles}")

    if num_triangles == 0:
        print("No triangles found.")
        return

    areas = []
    for i, tri in enumerate(triangles):
        area = calculate_triangle_area(tri.reshape(3, 2), pixel_to_micron)
        areas.append(area)
        # Draw the triangle on the image
        cv2.drawContours(img, [tri], -1, (0, 255, 0), 2)
        # Put the area text on the image
        center = np.mean(tri.reshape(3, 2), axis=0).astype(int)
        cv2.putText(img, f"{area:.2f} µm²", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    print("Areas of detected triangles (in µm²):")
    for i, area in enumerate(areas):
        print(f"Triangle {i+1}: {area:.2f} µm²")

    # Save the image with drawn triangles and area annotations
    output_image_path = "output_with_triangles106.jpg"
    cv2.imwrite(output_image_path, img)
    print(f"Output image saved to {output_image_path}")



if __name__ == "__main__":
    image_path = "first/106a.jpeg"  # Replace with your image path
    main(image_path)
