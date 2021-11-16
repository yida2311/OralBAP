import numpy as np
from PIL import Image
from torch._C import dtype

CLASS_COLORS = (
    (0, 0, 0),
    (192, 0, 0), 
    (0, 192, 0), 
    (0, 0, 192)
)

def RGB_mapping_to_class(label):
    h, w = label.shape[0], label.shape[1]
    classmap = np.zeros(shape=(h, w))

    for i in range(4):
        indices = np.where(np.all(label == CLASS_COLORS[i], axis=-1))
        classmap[indices[0].tolist(), indices[1].tolist()] = i
    return classmap


def class_to_RGB(label):
    h, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(h, w, 3)).astype(np.uint8)
    
    for i in range(4):
        indices = np.where(label == i)
        colmap[indices[0].tolist(), indices[1].tolist(), :] = CLASS_COLORS[i]
    return colmap