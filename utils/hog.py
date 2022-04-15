import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def img_gradient(img):
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)
    gradient_angle = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)
    return gradient_magnitude, gradient_angle


def get_closest_bins(angle, bin_size, angle_unit):
    idx = int(angle / angle_unit)
    mod = angle % angle_unit
    return idx, (idx + 1) % bin_size, mod


def cell_gradient(magnitude, angle, bin_size=8):
    angle_unit = 360 / bin_size
    orientation_centers = [0] * bin_size
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            gradient_strength = magnitude[i][j]
            gradient_angle = angle[i][j]
            min_angle, max_angle, mod = get_closest_bins(gradient_angle, bin_size, angle_unit)
            orientation_centers[min_angle] += gradient_strength * (1 - (mod / angle_unit))
            orientation_centers[max_angle] += gradient_strength * (mod / angle_unit)
    return orientation_centers


def render_gradient(image, cell_gradient_v, cell_size, angle_unit):
    cell_width = cell_size / 2
    max_mag = np.max(np.array(cell_gradient_v))
    for x in range(cell_gradient_v.shape[0]):
        for y in range(cell_gradient_v.shape[1]):
            cell_grad = cell_gradient_v[x][y]
            cell_grad /= max_mag
            angle = 0
            angle_gap = angle_unit
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x * cell_size + magnitude * cell_width * np.cos(angle_radian))
                y1 = int(y * cell_size + magnitude * cell_width * np.sin(angle_radian))
                x2 = int(x * cell_size - magnitude * cell_width * np.cos(angle_radian))
                y2 = int(y * cell_size - magnitude * cell_width * np.sin(angle_radian))
                cv2.line(image, (y1, x1), (y2, x2), int(255 * np.sqrt(magnitude)))
                angle += angle_gap
    return image


def get_vector(cell_gradient_vector):
    hog_vector = []
    for i in range(cell_gradient_vector.shape[0] - 1):
        for j in range(cell_gradient_vector.shape[1] - 1):
            block_vector = []
            block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            hog_vector.append(block_vector)
    return hog_vector


def hog(img_path, cell_size=8, bin_size=8):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.sqrt(img / float(np.max(img)))
    h, w = img.shape
    gradient_magnitude, gradient_angle = img_gradient(img)
    gradient_magnitude = abs(gradient_magnitude)
    cell_gradient_vector = np.zeros((int(h / cell_size), int(w / cell_size), bin_size))
    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle, bin_size)
    hog_image = render_gradient(np.zeros((h, w)), cell_gradient_vector, cell_size, 360 / bin_size)
    hog_vector = get_vector(cell_gradient_vector)
    hog_vector = np.array(hog_vector)
    print(hog_vector.shape)
    plt.imshow(hog_image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    path = "/home/chenzhentao/fgfv_data/aligned/twins/1-1.jpg"
    hog(img_path=path, cell_size=8, bin_size=8)
