def centroid(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def is_above(bbox1, bbox2):
    _, y1a, _, y2a = bbox1
    _, y1b, _, y2b = bbox2
    return y2a <= y1b  # parte bassa di A sopra parte alta di B

def is_left_of(bbox1, bbox2):
    x1a, _, x2a, _ = bbox1
    x1b, _, x2b, _ = bbox2
    return x2a <= x1b

def is_right_of(bbox1, bbox2):
    x1a, _, x2a, _ = bbox1
    x1b, _, x2b, _ = bbox2
    return x1a >= x2b

def overlaps(bbox1, bbox2, class_a, class_b):

    x1a, y1a, x2a, y2a = bbox1
    x1b, y1b, x2b, y2b = bbox2

    xi1 = max(x1a, x1b)
    yi1 = max(y1a, y1b)
    xi2 = min(x2a, x2b)
    yi2 = min(y2a, y2b)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)

    return inter_width > 0 and inter_height > 0

'''
actually deprecated for many problems.
'''
def contains(bbox_a, bbox_b, class_a, class_b, iou_threshold=0.98):
    #check if class are given and if are an invalid couple.
    x1a, y1a, x2a, y2a = bbox_a
    x1b, y1b, x2b, y2b = bbox_b

    xi1 = max(x1a, x1b)
    yi1 = max(y1a, y1b)
    xi2 = min(x2a, x2b)
    yi2 = min(y2a, y2b)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_b = max(1e-6, (x2b - x1b) * (y2b - y1b))

    return inter_area / area_b >= iou_threshold


def is_adjacent(bbox1, bbox2, threshold=75):
    c1 = centroid(bbox1)
    c2 = centroid(bbox2)
    dist = ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5
    return dist < threshold