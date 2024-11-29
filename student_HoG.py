import numpy as np
import cv2
import pdb


def get_features(image, x, y, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. 

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your HoG-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the HoG paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points

    You may also detect and describe features at particular orientations.

    Returns:
    -   HoG_normed: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard HoG).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'

    def get_gradn(img):
        dx = cv2.filter2D(img, -1, np.array([-1, 0, 1]))
        dy = cv2.filter2D(img, -1, np.array([[-1, 0, 1]]))
        gradn = np.sqrt(dx ** 2 + dy ** 2)
        thea = np.arctan2(dx, dy) / np.pi * 180
        return gradn, thea

    def get_cell_hog(grad, the):
        cell_hog = np.zeros(12)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                cell_hog[int(np.ceil((the[i, j] +180)/ 30))-1] += grad[ i, j]

        return cell_hog

    def get_Hog(gradn, thea, x, y, win_size=48, cell_size=16, block_size=4):
        HoG_normed = []
        for i in range(x.shape[0]):
            x_i = x[i] - win_size // 2  # The center of each point of interest/ 每一个兴趣点的中心
            x_i_ = x[i] - win_size // 2  + 2 * cell_size
            y_i = y[i] - win_size // 2  #  Select a window around the point of interest to extract the hog feature descriptor /围绕兴趣点选择一个窗口用于提取Hog特征描述子
            y_i_ = y[i] - win_size // 2 + 2 * cell_size
            HoG = []
            for c_y in range(y_i, y_i_, cell_size):  # cell
                for c_x in range(x_i, x_i_, cell_size):
                    block_hog = []
                    for k in range(block_size // 2):   #block
                        for v in range(block_size // 2):
                            cellx = c_x + v * cell_size
                            celly = c_y + k * cell_size
                            sub_gradn = gradn[celly:celly+cell_size, cellx:cellx+cell_size]
                            sub_thea = thea[celly:celly+cell_size, cellx:cellx+cell_size]
                            cell_hog = get_cell_hog(sub_gradn, sub_thea)
                            block_hog.append(cell_hog)
                    block_hog = np.concatenate(block_hog)
                    block_hog = block_hog / np.linalg.norm(block_hog)
                    HoG.append(block_hog)
            HoG = np.concatenate(HoG)
            HoG_normed.append(HoG)
        return np.array(HoG_normed)
    gradn, thea = get_gradn(image)
    HoG_normed = get_Hog(gradn, thea, x, y, win_size=48, cell_size=16, block_size=4)
    return HoG_normed
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################



