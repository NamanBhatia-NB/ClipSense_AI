from methods.base import BaseMethod
from utils import graph_to_indices
from collections import Counter
import random

class RandomMethod(BaseMethod):
    """Random sampling baseline. The number of predicted highlights are determined from the distribution of the number of GT highlights."""
    
    @classmethod
    def add_cli_args(cls, parser):
        pass
    
    def __init__(self, args, train_dataset):
        # to determine the number of predicted highlights from the distribution of the number of GT highlights
        n_gt_counter = Counter[int]()
        for row in train_dataset:
            indices = graph_to_indices(row['gt'], args.max_n_gt, args.gt_threshold)
            n_gt_counter.update([len(indices)])
        self.n_gt_population = list[int](n_gt_counter.keys())
        self.n_gt_weights = [n_gt_counter[x] / len(train_dataset) for x in self.n_gt_population]

    def predict(self, row):
        n_predictions = random.choices(self.n_gt_population, self.n_gt_weights)[0]
        predictions = random.sample(range(0, 100), n_predictions)
        return predictions

class FrequencyMethod(RandomMethod):
    """Frequency-based sampling baseline. The number of predicted highlights are determined from the frequency of the highlight indices in the train split."""

    @classmethod
    def add_cli_args(cls, parser):
        pass
    
    def __init__(self, args, train_dataset):
        super().__init__(args, train_dataset)
        self.index_counter  = Counter[int]()
        for row in train_dataset:
            indices = graph_to_indices(row['gt'], args.max_n_gt, args.gt_threshold)
            self.index_counter.update(indices)
        
    def predict(self, row):
        n_predictions = random.choices(self.n_gt_population, self.n_gt_weights)[0]
        predictions = [item[0] for item in self.index_counter.most_common(n_predictions)]
        return predictions
