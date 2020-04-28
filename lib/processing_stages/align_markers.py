import numpy as np
import cv2
from lib.utils import show_image

# [0]BR, [1]BL, [2]M, [3]TR, [4]TL
markers_placement = [[0.949, 0.815], [0.047, 0.815],
                     [0.859, 0.181], [0.859, 0.031],
                     [0.22, 0.031]]

def align_markers(orig, preprocessed, debug=False):
    i_h, i_w, i_s = orig.shape
    image_area = i_h * i_w
    conts, hier = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        debug_image = orig.copy()
        debug_image = cv2.drawContours(debug_image, conts, -1, (255, 0, 0), 1)
        show_image(debug_image)

    markers = []
    for n, cont in enumerate(conts):
        approx = cv2.approxPolyDP(cont, 0.03*cv2.arcLength(cont, True), True)
        c, size, r = cv2.minAreaRect(cont)
        w, h = size
        M = cv2.moments(approx)
        debug_marker = ''

        if w == 0 or h == 0:
            debug_marker += 'Z'
            k = s = 0
        else:
            k = w/h
            s = w*h
        if len(approx) != 4:
            debug_marker += f'A{len(approx)}'
        if not 0.75 < k < 1.5: #not square enough
            debug_marker += f'K{round(k, 2)}'
        if not 0.0002 < s/image_area < 0.0006:
            debug_marker += f'S{round(s/image_area, 5)}'
        if M['m00'] == 0:
            debug_marker += 'M'
            cx, cy = map(int, c)
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

        if debug_marker:
            if debug:
                debug_image = cv2.putText(debug_image, debug_marker, (cx, cy),
                                          cv2.FONT_HERSHEY_PLAIN, 0.75,
                                          (0, 0, 255))
            continue

        markers.append([cx, cy])

        if debug:
            debug_image = cv2.drawMarker(debug_image, (cx, cy), (0, 255, 255), thickness=3)

    if debug:
        print(len(markers), 'markers')
        show_image(debug_image)

    assert len(markers) == 5

    # [0]BR, [1]BL, [2]M, [3]TR, [4]TL
    markers.sort(key=lambda x: x[1], reverse=True)
    if markers[0][0] < markers[1][0]:
        markers[0], markers[1] = markers[1], markers[0]
    if markers[3][0] < markers[4][0]:
        markers[3], markers[4] = markers[4], markers[3]

    #check if the image is upside down
    if abs(markers[2][1] - markers[0][1]) < abs(markers[2][1] - markers[3][1]):
        markers[3], markers[1] = markers[1], markers[3]
        markers[4], markers[0] = markers[0], markers[4]

    src_points = np.array(markers)

    dest_points = [[x*i_w, y*i_h] for x, y in markers_placement]
    dest_points = np.array(dest_points)

    homography, status = cv2.findHomography(src_points, dest_points)

    out = cv2.warpPerspective(orig, homography, (i_w, i_h))
    return out
