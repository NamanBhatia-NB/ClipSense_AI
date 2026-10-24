import argparse
from datasets import load_dataset
from methods import get_method, list_methods
from metrics import calculate_metrics
from utils import graph_to_indices
import numpy as np
from tqdm import tqdm

def gather_results(results):
    keys = results[0].keys()
    report = {}
    for key in keys:
        values = [r[key] for r in results]
        report[key] = {
            'avg': np.mean(values),
            'std': 0 if len(values) == 1 else np.std(values, ddof=1), # sample standard deviation
        }
    
    return report

def main(args):
    # prepare dataset
    dataset = load_dataset("yhpark/rhapsody")
    dataset.set_format("numpy")
    test_dataset = dataset['test']
    
    # prepare prediction method - some methods may inspect the distribution of the train split during initialization
    method = get_method(args.method, args, dataset['train'])
    
    # run evaluation n_trials times
    all_results = []
    for _ in tqdm(range(args.n_trials)):
        gts = []
        predictions = []
        # iterate over the test split
        for row in test_dataset:
            # row.keys(): ['vid', 'cid', 'plid', 'gt', 'title', 'metadata', 'dva', 'hubert', 'segment_summaries', 'entire_summary', 'transcript']
            gt = graph_to_indices(row['gt'], args.max_n_gt, args.gt_threshold)
            if not args.include_empty_gt and len(gt) == 0:
                continue
            gts.append(gt)
            predictions.append(method.predict(row))
        
        # calculate metrics
        result = calculate_metrics(gts, predictions)
        all_results.append(result)
    
    # average results across trials
    report = gather_results(all_results)
    
    # print results
    for key, value in report.items():
        print(f"{key}: {value['avg']:.3f} ({value['std']:.3f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a method on the rhapsody test set.")
    
    # evaluator arguments (method-agnostic)
    parser.add_argument(
        "--max_n_gt",
        type=int,
        default=20,
        help="maximum number of ground truth (GT) highlight segments to extract from the raw replay graphs."
    )
    parser.add_argument(
        "--gt_threshold",
        type=float,
        default=0.5,
        help="minimum replay score threshold for a segment to be considered as a highlight."
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=1,
        help="number of repeated evaluation runs. Final results are averaged over n_trials runs."
    )
    parser.add_argument(
        "--include_empty_gt",
        action="store_true",
        help="include test examples with 0 ground truth highlights. By default, such examples are skipped."
    )

    # add method-specific arguments
    subparsers = parser.add_subparsers(dest="method", help="method to evaluate", required=True)
    for method_name in list_methods():
        method_cls = get_method(method_name, return_cls=True)
        method_parser = subparsers.add_parser(method_name, help=method_cls.__doc__)
        method_cls.add_cli_args(method_parser)
    
    args = parser.parse_args()
    print(args)

    main(args)
