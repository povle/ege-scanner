def get_coordinates(size, task, square):
    h, w = size[:2]
    DX = 30.15 * w / 1240
    DY = 44.2 * h / 1753
    x0 = 101 * w / 1240
    y0 = 448 * h / 1753

    x = x0 + square * DX
    if task > 20:
        x += 18.55 * DX

    row = (task-1) % 20
    y = y0 + row * DY
    y += (row // 5) * DY/3

    return (round(x), round(y))
