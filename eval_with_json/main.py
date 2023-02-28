import os
import json
import numpy as np
from eval_with_json.evaluator import InstanceSegEvaluator
from datasets.registration import register_seginw_instance

from collections import OrderedDict

def evaluate(submission_file, **kwargs):
    print("Starting SegInW Evaluation.....")

    output = {}
    all_leaderboard_keys = {
        # "model": ["# Vision Backbone Params [M]", "# Trainable Params [K]"],
        "datasets": ['Elephants', 'Hand-Metal', 'Watermelon', 'House-Parts', 'HouseHold-Items', 
                    'Strawberry', 'Fruits', 'Nutterfly-Squireel', 'Hand', 'Garbage', 'Chicken', 'Rail', 'Airplane-Parts', 
                    'Brain-Tumor', 'Poles', 'Electric-Shaver', 'Bottles', 'Toolkits', 'Trash', 'Salmon-Fillet', 'Puppies', 
                    'Tablets', 'Phones', 'Cows', 'Ginger-Garlic', 'ADE150-mIoU', 'ADE150-PQ', 'ADE150-mAP', 'ADE847-mIoU'],  # "Average Score"
    }

    results = {}
    ade_results = {}
    assert submission_file.endswith('.zip')
    import zipfile
    with zipfile.ZipFile(submission_file, 'r') as zf:
        for file_name in zf.namelist():
            if file_name.endswith('.json') and 'ade' in file_name:
                with zf.open(file_name) as json_file:
                    predictions = json.load(json_file)
                ade_results.update(predictions)
            elif file_name.endswith('.json'):
                with zf.open(file_name) as json_file:
                    predictions = json.load(json_file)
                dataset_name = file_name.split('/')[-1].split('.')[0]
                evaluator = InstanceSegEvaluator(dataset_name, output_dir="../../data/output/test")
                evaluator._results = OrderedDict()
                outputs = evaluator._eval_predictions(predictions)
                name = dataset_name.replace('seginw_','').replace('_val','')
                results[name] = outputs['AP']
            else:
                raise ValueError(f"Please provide json file annotation inside zip.")

        mean_scores = np.asarray(list(results.values()))
        results["Average Score"] = np.mean(mean_scores)
        results["Median Score"] = np.median(mean_scores)
        return results
    return output


if __name__ == "__main__":
    import sys
    prediction_file = sys.argv[1]
    # pharse cocodename: zeroshot, fewshot, fullshot
    print(evaluate(prediction_file))