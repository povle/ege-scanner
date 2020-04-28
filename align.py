import cv2
from lib.processing_stages import canny, threshold, align_markers, align_sheet


def sheet_align(orig, kwargs, debug=False):
    processed = canny(orig, **kwargs)
    aligned = align_sheet(orig, processed, debug=debug)
    return aligned


def adaptive_sheet_align(orig, debug=False):
    cn = {'threshold1': 60, 'threshold2': 120, 'L2gradient': True}
    while True:
        try:
            a = align(orig, {'canny': cn}, debug)
            print(cn)
            return a
        except Exception:
            cn['threshold1'] -= 3
            cn['threshold2'] -= 6
            if cn['threshold1'] < 0:
                raise


def marker_align(orig, kwargs, debug=False):
    a_processed = threshold(orig, invert=True,
                            gaussian={'ksize': (11, 11), 'sigmaX': 0},
                            **kwargs)
    aligned = align_markers(orig, a_processed, debug)
    return aligned


def adaptive_marker_align(orig, debug=False):
    thr = {'thresh': 250, 'maxval': 255, 'type': cv2.THRESH_BINARY}
    while True:
        try:
            a = marker_align(orig, {'threshold': thr}, debug)
            print(thr)
            return a
        except Exception:
            thr['thresh'] -= 5
            if thr['thresh'] < 0:
                raise


def align(image, debug=False):
    aligned = adaptive_sheet_align(image, debug)
    aligned = adaptive_marker_align(aligned, debug)
    return aligned
