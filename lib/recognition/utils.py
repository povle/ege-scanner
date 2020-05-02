import cv2

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


def pad_to_square(img, black=True):
    h, w = img.shape[:2]
    a = max(h+1, w+1)
    return pad_to_size(img, (a, a), black)
