import numpy as np
import torch


def ntf(file="https://notificationsounds.com/storage/sounds/file-sounds-1228-so-proud.mp3"):
    # from pydub import AudioSegment
    # from pydub.playback import play
    # song = AudioSegment.from_mp3(file)
    # play(song)
    from playsound import playsound
    playsound(file)

def generate_folds(L, k=5):
    permuted = np.random.permutation(np.arange(L))
    return np.array_split(permuted, k)

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
    if logits.ndim > 2:  # get the logits pair with highest difference in logits (1 higher than)
        max_idx = (logits[..., 1] - logits[..., 0]).argmax(1)
        logits = logits[range(logits.shape[0]), max_idx]
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
        'accuracy': accuracy, 
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

def postprocess_logits(logits, attention_mask, calibration_temperature=1.):  # logits shape=(batch_size, 128, 2)
    x, y = torch.argwhere(attention_mask).T
    max_idx = torch.ones(logits.size(0), dtype=torch.int, device=attention_mask.device)
    for sample_idx in range(logits.size(0)):
        mask_is = torch.argwhere(x == sample_idx).flatten()
        xi, yi = x[mask_is], y[mask_is]
        assert torch.equal(xi, torch.ones(len(mask_is), dtype=xi.dtype, device=xi.device)*sample_idx)
        max_idx[sample_idx] = yi[(logits[sample_idx, yi, 1] - logits[sample_idx, yi, 0]).argmax(-1)]
    # logits = (logits[:, :(1+end_idx), 1] - logits[:, :(1+end_idx), 0]).argmax(1)
    logits = logits[range(logits.shape[0]), max_idx.long()]
    return logits / calibration_temperature