import numpy as np
import cv2

# [0]BR, [1]BL, [2]M, [3]TR, [4]TL
markers_placement = [[0.949, 0.815], [0.047, 0.815],
                     [0.859, 0.181], [0.859, 0.031],
                     [0.22, 0.031]]

def align_markers(image):
    i_h, i_w, i_s = image.shape
    image_area = i_h * i_w
    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.GaussianBlur(processed, (5, 5), 0)
    processed = cv2.threshold(processed, 160, 255, cv2.THRESH_BINARY)[1]
    processed = cv2.bitwise_not(processed)
    conts, hier = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = []
    for n, cont in enumerate(conts):
        approx = cv2.approxPolyDP(cont, 0.05*cv2.arcLength(cont, True), True)
        c, size, r = cv2.minAreaRect(cont)
        w, h = size
        if w == 0 or h == 0 or len(approx) != 4:
            continue
        k = w/h
        s = w*h
        if 0.75 < k < 1.5 and 0.00018 < s/image_area < 0.0005:
            M = cv2.moments(approx)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            markers.append([cx, cy])

    assert len(markers) == 5

    # [0]BR, [1]BL, [2]M, [3]TR, [4]TL
    markers.sort(key=lambda x: x[1], reverse=True)
    if markers[0][0] < markers[1][0]:
        markers[0], markers[1] = markers[1], markers[0]
    if markers[3][0] < markers[4][0]:
        markers[3], markers[4] = markers[4], markers[3]
    src_points = np.array(markers)

    dest_points = [[x*i_w, y*i_h] for x, y in markers_placement]
    dest_points = np.array(dest_points)

    homography, status = cv2.findHomography(src_points, dest_points)

    out = cv2.warpPerspective(image, homography, (i_w, i_h))
    return out
