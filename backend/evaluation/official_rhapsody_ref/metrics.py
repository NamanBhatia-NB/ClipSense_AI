def hit(gt: list[int], pred: list[int]) -> float:
    if len(gt) == 0:
        return 1 if len(pred) == 0 else 0
    return int(len(set(pred).intersection(set(gt))) > 0)

def precision(gt: list[int], pred: list[int]) -> float:
    if len(pred) == 0:
        return 1 if len(gt) == 0 else 0
    return len(set(gt).intersection(set(pred))) / len(pred)

def recall(gt: list[int], pred: list[int]) -> float:
    if len(gt) == 0:
        return 1 if len(pred) == 0 else 0
    return len(set(gt).intersection(set(pred))) / len(gt)

def f1(gt: list[int], pred: list[int]) -> float:
    if len(gt) == 0 and len(pred) == 0:
        return 1
    if len(gt) > 0 and len(pred) == 0:
        return 0
    if len(gt) == 0 and len(pred) > 0:
        return 0

    # len(gt) > 0 and len(pred) > 0
    if len(set(gt).intersection(set(pred))) == 0:
        return 0
    return 2 * precision(gt, pred) * recall(gt, pred) / (precision(gt, pred) + recall(gt, pred))

def ap(gt: list[int], pred: list[int]) -> float:
    if len(gt) == 0:
        return 1 if len(pred) == 0 else 0
    ap = 0
    num_hits = 0
    for i, p in enumerate(pred):
        if p in gt:
            num_hits += 1
            ap += num_hits / (i + 1)
    return ap / len(gt)

def calculate_metrics(gts: list[list[int]], preds: list[list[int]]) -> dict:
    """
    Calculate evaluation metrics from predicted and ground truth highlight indices.
    
    Args:
        gts (list[list[int]]): Ground truth data from the dataset
        preds (list[list[int]]): list of predictions from the method
        
    Returns:
        dict: dictionary containing evaluation metrics
    """
    metrics = {metric: [] for metric in ['hit', 'precision', 'recall', 'f1', 'ap', 'n_gt', 'n_pred']}
    for gt, pred in zip(gts, preds):
        metrics['hit'].append(hit(gt, pred))
        metrics['precision'].append(precision(gt, pred))
        metrics['recall'].append(recall(gt, pred))
        metrics['f1'].append(f1(gt, pred))
        metrics['ap'].append(ap(gt, pred))
        metrics['n_gt'].append(len(gt))
        metrics['n_pred'].append(len(pred))

    # average over examples
    avg_metrics = {}
    for key in metrics.keys():
        avg_metrics[key] = sum(metrics[key]) / len(metrics[key])
    return avg_metrics
