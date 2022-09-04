from math import ceil
import numpy as np
import torch
import logging


def ntf(file="https://notificationsounds.com/storage/sounds/file-sounds-1228-so-proud.mp3"):
    # from pydub import AudioSegment
    # from pydub.playback import play
    # song = AudioSegment.from_mp3(file)
    # play(song)
    from playsound import playsound
    playsound(file)

def full_display(df):
    import pandas as pd
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, ):
        pd.options.display.max_colwidth = 200
        display(df)

def generate_folds(L, k=5):
    permuted = np.random.permutation(np.arange(L))
    return np.array_split(permuted, k)

def easy_ensenble_generate_kfolds(indices, k, minority_idx, fold_size=None, minor_major_ratio=1.):
    s_fold = fold_size if fold_size else len(indices) // k 
    majority_idx = np.array(list(set(indices) - set(minority_idx)))
    minor_proportion = minor_major_ratio / (1. + minor_major_ratio)
    num_minor_samples = int(s_fold*minor_proportion)
    minor_samples = np.random.choice(minority_idx, size=(k, num_minor_samples), replace=True)
    major_samples = np.random.choice(majority_idx, size=(k, s_fold-num_minor_samples), replace=False)
    folds = np.concatenate((minor_samples, major_samples), axis=1)
    folds = folds[:, np.random.permutation(range(s_fold))]
    print(f'Using easy ensemble - each fold has {minor_samples.shape[1]} negatives in the training set, paired with {major_samples.shape[1]} positives')
    return folds

# def easy_ensenble_generate_kfolds(L, k, minority_idx, fold_size=None):
#     """Wrong implementation - minority class examples are not all used in each fold."""
#     s_fold = len(minority_idx) * 2
#     majority_idx = np.array(list(set(range(L)) - set(minority_idx)))
#     folds = []
#     for _ in range(k):
#         minor_sample = np.random.choice(minority_idx, size=s_fold//2, replace=False)
#         major_sample = np.random.choice(majority_idx, size=s_fold//2, replace=False)
#         fold = np.concatenate((minor_sample, major_sample))
#         fold = np.random.permutation(fold)
#         folds.append(fold)
#     logging.info(f'Using easy ensemble - each fold has {len(minor_sample)} negatives in the training set, paired with {len(major_sample)} positives')
#     return np.array(folds)


def extract_predictions(logits):
    contain_bool = np.vectorize(lambda x: x in np.array([0, 1]))
    pass_label = contain_bool(np.unique(logits).flatten()).all()
    
    if logits.ndim > 2:  # get the logits pair with highest difference in logits (1 higher than)
        max_idx = (logits[..., 1] - logits[..., 0]).argmax(1)
        logits = logits[range(logits.shape[0]), max_idx]
    
    if pass_label:
        predictions = logits.any(axis=-1).astype(int)

    else:
        predictions = np.argmax(logits, axis=-1)
    return predictions

def compute_metrics(eval_pred):
    """
    Called at the end of validation - compute F1 score and/or accuracy.

    eval_pred: a tuple that contains:
        (a) logit scores output by model, and
        (b) ground truth labels
    """
    
    outputs, labels = eval_pred
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    predictions = extract_predictions(logits)
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
        'accuracy': accuracy, 
    }


def voting(logits:torch.Tensor, val_accuracy=None):
    agg_labels = []
    model_id = 0
    for single_examples_pred_labels in logits.argmax(-1).T:
        labels, counts = torch.unique(single_examples_pred_labels, return_counts=True)
        agg_labels.append(labels[torch.argmax(counts)])
        model_id += 1
    return torch.tensor(agg_labels).long()


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

def postprocess_logits(logits:torch.Tensor, attention_mask:torch.Tensor, calibration_temperature=1.) -> torch.Tensor:  # logits shape=(batch_size, 128, 2)
    x, y = torch.argwhere(attention_mask).T
    max_idx = torch.ones(logits.size(0), dtype=torch.int, device=attention_mask.device)
    for sample_idx in range(logits.size(0)):
        mask_is = torch.argwhere(x == sample_idx).flatten()
        xi, yi = x[mask_is], y[mask_is]
        assert torch.equal(xi, torch.ones(len(mask_is), dtype=xi.dtype, device=xi.device)*sample_idx)
        max_idx[sample_idx] = yi[(logits[sample_idx, yi, 1] - logits[sample_idx, yi, 0]).argmax(-1)]
    assert torch.all(max_idx < attention_mask.sum(1))
    logits = logits[range(logits.shape[0]), max_idx.long()]
    return logits / calibration_temperature