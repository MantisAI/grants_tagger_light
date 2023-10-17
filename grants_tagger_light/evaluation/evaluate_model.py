"""
Evaluate model performance on test set
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional
from transformers import pipeline
from transformers.pipelines import PIPELINE_REGISTRY

import scipy.sparse as sp
import numpy as np
import typer
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from wasabi import row, table
from grants_tagger_light.utils import load_data, load_train_test_data
from grants_tagger_light.models.bert_mesh import BertMesh, BertMeshPipeline
from tqdm import tqdm

PIPELINE_REGISTRY.register_pipeline(
    "grants-tagging", pipeline_class=BertMeshPipeline, pt_model=BertMesh
)


def evaluate_model(
    model_path,
    data_path,
    threshold,
    split_data=True,
    results_path=None,
    full_report_path=None,
    batch_size=10,
):
    model = BertMesh.from_pretrained(model_path)

    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit([list(model.id2label.keys())])
    model.label2id = {value: key for key, value in model.id2label.items()}

    pipe = pipeline(
        "grants-tagging",
        model=model,
        tokenizer="Wellcome/WellcomeBertMesh",
    )

    if split_data:
        print(
            "Warning: Data will be split in the same way as train."
            " If you don't want that you set split_data=False"
        )
        _, X_test, _, Y_test = load_train_test_data(data_path, label_binarizer)
    else:
        X_test, Y_test, _ = load_data(data_path, label_binarizer, model_label2id=model.label2id)

    logging.info('data loaded')
    X_test = X_test[:100]
    Y_test = Y_test[:100]

    top_10_index = np.argsort(np.sum(Y_test, axis=0))[::-1][:10]
    print(top_10_index)
    print(np.sum(Y_test, axis=0)[top_10_index])

    logging.info(f'beginning of evaluation - {len(X_test)} items')
    
    outputs = pipe(['This grant is about malaria and HIV',
                    'My name is Arnault and I live in barcelona'], return_labels=False)
    print(outputs)
    for output in outputs:
        argmax = np.argmax(output[0])
        print(f'argmax: {argmax}')
        print(argmax, output[0][argmax])
    Y_pred_proba = pipe(X_test, return_labels=False)
    #Y_pred_proba = [torch.sigmoid(proba) for proba in Y_pred_proba]
    
    #Y_pred_proba = [torch.sigmoid(proba) for proba in Y_pred_proba]
    print(Y_pred_proba)
    Y_pred_proba = torch.vstack(Y_pred_proba)
    print('loss')
    print(F.binary_cross_entropy_with_logits(torch.tensor(Y_pred_proba), torch.tensor(Y_test).float()))
    print(Y_pred_proba.shape, Y_test.shape)
    Y_pred_proba = sp.csr_matrix(Y_pred_proba)

    if not isinstance(threshold, list):
        threshold = [threshold]

    widths = (12, 5, 5, 5)
    header = ["Threshold", "P", "R", "F1"]
    print(table([], header, divider=True, widths=widths))

    results = []
    for th in threshold:
        Y_pred = Y_pred_proba > th
        print(np.sum(Y_pred, axis=0)[:, top_10_index])
        print(np.sum(Y_pred))
        print(Y_pred_proba.shape)
        print(np.max(Y_pred_proba.toarray(), axis=0)[top_10_index])
        print(Y_pred.sum(axis=0))
        top_10_index = np.argsort(np.array(Y_pred.sum(axis=0))[0])[::-1][:10]
        print(top_10_index)
        print(np.array(Y_pred.sum(axis=0))[0, top_10_index])

        p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_pred.toarray(), average="micro")
        full_report = classification_report(Y_test, Y_pred.toarray(), output_dict=True)

        # Gets averages
        averages = {str(idx): report for idx, report in full_report.items() if "avg" in idx}
        # Gets class reports and converts index to class names for readability
        full_report = {
            str(label_binarizer.classes_[int(idx)]): report
            for idx, report in full_report.items()
            if "avg" not in idx
        }

        # Put the averages back
        full_report = {**averages, **full_report}

        result = {
            "threshold": f"{th:.2f}",
            "precision": f"{p:.2f}",
            "recall": f"{r:.2f}",
            "f1": f"{f1:.2f}",
        }
        results.append(result)

        row_data = (
            result["threshold"],
            result["precision"],
            result["recall"],
            result["f1"],
        )
        print(row(row_data, widths=widths))

    if results_path:
        with open(os.path.join(results_path, 'metrics.json'), "w") as f:
            f.write(json.dumps(results, indent=4))
    if full_report_path:
        with open(os.path.join(results_path, 'full_report_metrics.json'), "w") as f:
            f.write(json.dumps(full_report, indent=4))


evaluate_model_app = typer.Typer()


@evaluate_model_app.command()
def evaluate_model_cli(
    model_path: str = typer.Argument(
        ..., help="comma separated paths to pretrained models"
    ),
    data_path: Path = typer.Argument(
        ..., help="path to data that was used for training"
    ),
    threshold: Optional[str] = typer.Option(
        "0.5", help="threshold or comma separated thresholds used to assign tags"
    ),
    results_path: Optional[str] = typer.Option(None, help="path to save results"),
    full_report_path: Optional[bool] = typer.Option(
        False,
        help="Whether to save full report or not, i.e. "
        "more comprehensive results than the ones saved in results_path",
    ),
    split_data: bool = typer.Option(
        True, help="flag on whether to split data in same way as was done in train"
    ),
):
    if "," in threshold:
        threshold = [float(t) for t in threshold.split(",")]
    else:
        threshold = float(threshold)

    evaluate_model(
        model_path,
        data_path,
        threshold,
        split_data,
        results_path=results_path,
        full_report_path=full_report_path,
    )


if __name__ == "__main__":
    evaluate_model_app()
