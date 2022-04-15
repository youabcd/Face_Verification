import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = "/home/chenzhentao/fgfv_data/aligned/twins/1-1.jpg"


def gaussian_filter(img, k_size=3, sigma=1.5):
    pad = k_size // 2
    # pad_img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    pad_img = np.pad(img, ((pad, pad), (pad, pad)), 'edge')
    tmp = pad_img.copy()
    kernel = np.zeros((k_size, k_size))
    for i in range(k_size):
        for j in range(k_size):
            kernel[i][j] = np.exp(-((i - pad) ** 2 + (j - pad) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    kernel = np.round(kernel / kernel.sum(), 3)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pad_img[i + pad, j + pad] = np.sum(kernel * tmp[i:i + k_size, j:j + k_size])
    pad_img = np.clip(pad_img, 0, 255)
    out = pad_img[pad: pad + img.shape[0], pad:pad + img.shape[1]]
    return out.astype(np.uint8)


def sift():
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    out = cv2.GaussianBlur(img.copy(), (3, 3), 1.5)
    img1 = gaussian_filter(img.copy())
    plt.imshow(img1, cmap='gray')
    plt.show()
    plt.imshow(out, cmap='gray')
    plt.show()


if __name__ == '__main__':
    sift()
