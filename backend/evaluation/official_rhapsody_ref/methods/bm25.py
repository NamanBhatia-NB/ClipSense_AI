from methods.base import BaseMethod
from rank_bm25 import BM25Okapi
import numpy as np

class BM25Method(BaseMethod):
    """BM25-based highlight detectined from the BM25 score of the highlight indices in the train split."""
    
    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument(
            "--bm25_threshold",
            type=float,
            default=3.0,
            help="threshold for the BM25 score to be considered as a highlight.",
        )
    
    def __init__(self, args, train_dataset):
        self.bm25_threshold = args.bm25_threshold
        self.max_n_predictions = args.max_n_gt
    
    def predict(self, row):
        corpus = row['segment_summaries']
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        query = row['title']
        tokenized_query = query.split()

        doc_scores = bm25.get_scores(tokenized_query)
        indices = np.nonzero(np.array(doc_scores) > self.bm25_threshold)[0].tolist()
        predictions = sorted(indices, key=lambda x: doc_scores[x], reverse=True)[:self.max_n_predictions] # preserve the order of BM25 scores
        return predictions
        