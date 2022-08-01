import numpy as np


def generate_folds(L, k=5):
    return np.random.permutation(np.arange(L)).reshape(k, -1)


def compute_metrics(eval_pred):
    """
    Called at the end of validation - compute F1 score and/or accuracy.

    eval_pred: a tuple that contains:
        (a) logit scores output by model, and
        (b) ground truth labels
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    tp = np.argwhere((predictions == labels) * (predictions == 1)).sum()
    allp = (predictions == 1).sum()
    return {'F1':tp/allp, 'accuracy': np.mean(predictions==labels)}


def voting(logits, val_accuracy=None):
    labels, count = np.unique(logits, axis=0, return_counts=True)
    return labels[np.argmax(count)]


# def averaging(logits, val_accuracy, weighted=True):
#     assert len(logits) == len(val_accuracy)
#     if weighted:
#         weights = (np.argsort(val_accuracy).argsort()+1) / np.arange(1, len(val_accuracy)+1).sum()
    
#     ensemble_logits = np.array(logits) * np.expand_dims(weights, axis=1)
#     return np.argmax(ensemble_logits, axis=1)


def averaging(logits, val_accuracy, weighted=True):
    assert len(logits) == len(val_accuracy)
    if weighted:
        weights = (np.argsort(val_accuracy).argsort()+1) / np.arange(1, len(val_accuracy)+1).sum()
    else:
        weights = np.ones(len(logits)) / len(logits)
    ensemble_logits = (np.array(logits) * np.expand_dims(weights, axis=(1, 2))).sum(0)
    return np.argmax(ensemble_logits, axis=1)