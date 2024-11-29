import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb


def get_interest_points(img):
    """
    Implement the LoG corner detector to start with.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)

    Returns:
    -   x_sort: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y_sort: A numpy array of shape (N,) containing y-coordinates of interest points

    The code will automatically select the first 100 key points for evaluation, 
    so please return x_sort and y_sort with decreasing confidence.

    please notice that cv2 returns the image with [h,w], which corrsponces [y,x] dim respectively. 
    ([vertically direction, horizontal direction])

    """

    def get_Logkernel(kernel_size=5, sigma=3):
        gauss_ksize = kernel_size + 2
        Lap_kernel = np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0]))
        Log_kernel = np.zeros((kernel_size, kernel_size))
        gauss_kvec = np.arange(gauss_ksize) - int(gauss_ksize / 2)
        gauss_svec = np.exp(-(gauss_kvec ** 2) / (2 * (sigma ** 2)))
        gauss_kernel = np.dot(gauss_svec.reshape(gauss_ksize, 1), gauss_svec.reshape(1, gauss_ksize))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        for i in range(kernel_size):
            for j in range(kernel_size):
                Log_kernel[i, j] = np.sum(Lap_kernel * gauss_kernel[i:i + 3, j:j + 3])
        return Log_kernel

    def NMS(img_f):
        kepoint_x = []
        kepoint_y = []
        kepoint_value = []
        for j in range(3, img_f.shape[0] - 3):
            for i in range(3, img_f.shape[1] - 3):
                if img_f[j, i] == np.max(img_f[j - 1:j + 2, i - 1:i + 2]) and img_f[j, i] > 0:
                    kepoint_y.append(j)
                    kepoint_x.append(i)
                    kepoint_value.append(img_f[j, i])
        index = np.argsort(kepoint_value)[::-1]
        kepoint_x = np.array(kepoint_x)[index]
        kepoint_y = np.array(kepoint_y)[index]
        return kepoint_x, kepoint_y

    Log_kernel = get_Logkernel(7, 1)            # 7,1
    img_f = cv2.filter2D(img, -1, Log_kernel)
    threshold = 0.30                             # 0.3
    img_threshold = img_f.max() * threshold
    img_f[img_f < img_threshold] = 0
    x_sort, y_sort = NMS(img_f)

    return x_sort, y_sort




