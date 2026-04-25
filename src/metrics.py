import numpy as np


def one_hot(y, num_classes=10):
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def accuracy_from_logits(logits, y_true):
    y_pred = np.argmax(logits, axis=1)
    return np.mean(y_pred == y_true)


def confusion_matrix_np(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_accuracy(cm):
    denom = np.maximum(cm.sum(axis=1), 1)
    return cm.diagonal() / denom
