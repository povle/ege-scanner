import cv2
import numpy as np

def pad_to_size(img, size, black=True):
    h, w = img.shape[:2]
    if size[1] < w or size[0] < h:
        raise ValueError

    top = bottom = (size[0] - h) // 2
    if top + bottom != size[0] - h:
        top += 1

    left = right = (size[1] - w) // 2
    if left + right != size[1] - w:
        left += 1

    color = (0, 0, 0) if black else (255, 255, 255)
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)

def pad_to_square(img, border=1, black=True):
    h, w = img.shape[:2]
    a = max(h+border, w+border)
    return pad_to_size(img, (a, a), black)

def crop_to_fit(img, threshold=0, border=4):
    # modified https://codereview.stackexchange.com/a/132933
    mask = img > threshold
    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)
    if coords.size != 0:
        # Bounding box of non-black pixels.
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
        # Get the contents of the bounding box.
        img = img[x0:x1, y0:y1]
    return pad_to_square(img, border)
