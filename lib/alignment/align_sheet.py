import cv2
import numpy as np
from lib.alignment.utils import show_image

def align_sheet(orig, preprocessed, out_mult=30, debug=False):
    conts, hier = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_cont = None
    max_area = None
    for cont in conts:
        area = cv2.contourArea(cont)
        if largest_cont is None or area > max_area:
            largest_cont = cont
            max_area = area
    assert largest_cont is not None

    approx = cv2.approxPolyDP(largest_cont, 0.05*cv2.arcLength(largest_cont, True), True)

    if debug:
        print(len(approx), 'points')
        show_image(cv2.drawContours(orig, [approx], -1, (255, 0, 0), thickness=1))

    assert len(approx) == 4

    # top right, top left, bottom left, bottom right
    src_points = [list(p[0]) for p in approx]
    src_points.sort(key=lambda x: x[1])
    if src_points[0][0] < src_points[1][0]:
        src_points[0], src_points[1] = src_points[1], src_points[0]
    if src_points[2][0] > src_points[3][0]:
        src_points[2], src_points[3] = src_points[3], src_points[2]
    src_points = np.array(src_points)

    out_x = int(out_mult * 70)
    out_y = int(out_mult * 99)

    dest_points = np.array([[out_x, 0], [0, 0], [0, out_y], [out_x, out_y]])

    homography, status = cv2.findHomography(src_points, dest_points)

    out = cv2.warpPerspective(orig, homography, (out_x, out_y))
    return out
