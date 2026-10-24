from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from datasets import Dataset

class BaseMethod(ABC):
    """Base class for all prediction methods."""
    
    @classmethod
    @abstractmethod
    def add_cli_args(cls, parser: ArgumentParser):
        """
        Add method-specific command-line arguments to the argument parser.

        Args:
            parser: Argument parser to which method-specific arguments are added.
        """
        pass
    
    @abstractmethod
    def __init__(self, args: Namespace, train_dataset: Dataset):
        """
        Initialize the prediction method.

        Args:
            args (Namespace): Command-line arguments.
            train_dataset (Dataset): Training dataset used for initialization.
        """
        # note that ome methods may not use all keys in `args` or `train_dataset`.
        pass

    @abstractmethod
    def predict(self, row: dict) -> list[int]:
        """
        Predict highlight indices for a given dataset row.

        Args:
            row (dict): A single dataset row. Expected to have the following keys: ['vid', 'cid', 'plid', 'gt', 'title', 'metadata', 'dva', 'hubert', 'segment_summaries', 'entire_summary', 'transcript']

        Returns:
            list[int]: List of indices corresponding to predicted highlight segments.
        """
        pass
