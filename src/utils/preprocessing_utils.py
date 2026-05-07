import numpy as np


def multilabel_binarize(lists_of_labels):
    classes = sorted({label for lst in lists_of_labels for label in lst})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(lists_of_labels), len(classes)), dtype=np.uint8)
    for i, labels in enumerate(lists_of_labels):
        for label in labels:
            M[i, class_to_idx[label]] = 1
    return M, classes
