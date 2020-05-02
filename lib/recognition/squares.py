square_width = 30.15 / 1240
square_height = 44.2 / 1753

def get_coordinates(size, task, square):
    h, w = size[:2]
    DX = square_width * w
    DY = square_height * h
    x0 = 101 * w / 1240
    y0 = 448 * h / 1753

    x = x0 + square * DX
    if task > 20:
        x += 18.55 * DX

    row = (task-1) % 20
    y = y0 + row * DY
    y += (row // 5) * DY/3

    return (round(x), round(y))

def get_square(img, task, square):
    h, w = img.shape[:2]
    x0, y0 = get_coordinates((h, w), task, square)
    x1 = x0 + round(w * square_width)
    y1 = y0 + round(h * square_height)
    return img[y0:y1, x0:x1]
