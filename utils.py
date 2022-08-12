import numpy as np


def generate_folds(L, k=5):
    return np.random.permutation(np.arange(L)).reshape(k, -1)

def easy_ensenble_generate_kfolds(L, k, minority_idx):
    s_fold = L // k
    majority_idx = np.array(list(set(range(L)) - set(minority_idx)))
    folds = []
    for _ in range(k):
        minor_sample = np.random.choice(minority_idx, size=s_fold//2, replace=False)
        major_sample = np.random.choice(majority_idx, size=s_fold//2, replace=False)
        fold = np.concatenate((minor_sample, major_sample))
        folds.append(fold)
    return np.array(folds)

def compute_metrics(eval_pred):
    """
    Called at the end of validation - compute F1 score and/or accuracy.

    eval_pred: a tuple that contains:
        (a) logit scores output by model, and
        (b) ground truth labels
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions==labels).mean()

    tp = ((labels == 1) * (predictions == 1)).sum()
    allp = (predictions == 1).sum()
    fn = ((labels == 1) * (predictions == 0)).sum()
    precision = tp / allp
    recall = tp / (tp + fn)
    f1 = 2*precision*recall / (precision + recall)

    return {
        'F1': f1, 
        'precision': precision, 
        'recall': recall,
        # 'accuracy': accuracy, 
    }


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