import random

def generate_random_colour():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    return (r, g, b)

def get_box(xyxy, w, h):
        x1 = int(xyxy[0] * w)
        x2 = int(xyxy[2] * w)
        y1 = int(xyxy[1] * h)
        y2 = int(xyxy[3] * h)

        return x1, y1, x2, y2
