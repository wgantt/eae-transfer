import click
import json
import logging
import numpy as np
import os
import sys
import torch

from collections import defaultdict
from datasets import Dataset, concatenate_datasets
from functools import partial
from pprint import pprint
from tqdm.auto import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from eae.dataset.dataset import (
    DatasetChoice,
    TaskChoice,
    DATASET_TO_SPLITS
)
from eae.eval.eval import (
    extract_qa_spans,
    METRIC_MAP,
    QA_SPAN_PERCENTILES,
    ceaf_ree_phi_subset_metrics,
)
from eae.eval.common import to_dataclass
from eae.preprocessing.preprocess_qa import (
    preprocess,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@click.command()
@click.argument("test_datasets", type=str)
@click.argument("model_path", type=str)
@click.option(
    "--device",
    type=click.INT,
    default=0,
    help="the device to on which to load the model",
)
@click.option("--batch-size", type=click.INT, default=1, help="the batch size to use")
@click.option(
    "--max-doc-len",
    type=click.INT,
    default=None,
    help="maximum number of tokens in the input document (all longer docs will be truncated)",
)
@click.option(
    "--top-k",
    type=click.INT,
    default=1,
    help="maximum number of spans to predict per example",
)
@click.option(
    "--span-thresh",
    type=click.FLOAT,
    default=float("-inf"),
    help="Any spans with scores below this value will be excluded from the predictions",
)
@click.option(
    "--results-dir",
    type=click.STRING,
    default="..",
    help="directory where results will be saved",
)
@click.option("--seed", type=click.INT, default=42, help="random seed")
@click.option("--split", type=click.STRING, default="test", help="split to evaluate on")
@click.option("--verbose-metrics", is_flag=True, help="print event type-level metrics")
def inference(
    test_datasets,
    model_path,
    device,
    batch_size,
    max_doc_len,
    top_k,
    span_thresh,
    results_dir,
    seed,
    split,
    verbose_metrics,
) -> None:
    """Run inference for QA-based event argument extraction

    :param test_datasets: name of the test datasets to run inference on
        (should be a "+"-separated list of dataset names)
    :param model_path: path to the model with which to run inference
    :param device: The GPU on which the model will be loaded
    :param batch_size: The batch size to use
    :param max_doc_len: maximum number of tokens in the input document
        (all longer docs will be truncated)
    :param top_k: maximum number of spans to predict per example
    :param span_thresh: Any spans with scores below this value will be excluded from the predictions
    :param results_dir: where model predictions and metrics will be written
    :param seed: random seed
    :param split: split to evaluate on
    :param verbose_metrics: if true, will print event type-level metrics
    :returns None:
    """
    torch.manual_seed(seed)

    # dump inference config and model parameters
    config = locals()
    config["model_path"] = os.path.abspath(config.pop("model_path"))
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "inference_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.warning(f"Loading model {model_path} for inference...")
    m = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    task = TaskChoice.QA
    test_dataset_names = [ds.upper() for ds in test_datasets.split("+")]
    test_datasets = []
    test_ref_files = {}
    test_coref_files = {}
    for ds in test_dataset_names:
        assert ds.upper() in DatasetChoice._member_names_, (
            f"Invalid test dataset {ds.upper()}. "
            f"Must be one of {DatasetChoice._member_names_}"
        )
        data = Dataset.from_generator(
            DATASET_TO_SPLITS[task][DatasetChoice[ds.upper()]][split]
        )
        preprocess_fn = partial(
            preprocess,
            tokenizer=tokenizer,
            max_length=max_doc_len,
            add_special_tokens=False,
            columns_to_keep=data.column_names,
        )
        test_datasets.append(data.map(preprocess_fn, batched=True))
        # if DatasetChoice[ds.upper()] == DatasetChoice.WIKIEVENTS:
        #     test_ref_files[ds.upper()] = WIKIEVENTS_DATA_FILES["test"]
        #     test_coref_files[ds.upper()] = WIKIEVENTS_COREF_FILES["test"]
        # elif DatasetChoice[ds.upper()] == DatasetChoice.RAMS:
        #     test_ref_files[ds.upper()] = RAMS_DATA_FILES["test"]

    model_type = m.config.model_type
    forward_columns = [
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
    ]
    if "bert" in model_type:
        forward_columns.append("token_type_ids")

    test_data = concatenate_datasets(test_datasets)
    test_data.set_format("pt", columns=forward_columns, output_all_columns=True)

    all_gold_start_toks = []
    gold_start_toks = []
    all_gold_end_toks = []
    gold_end_toks = []
    pred_start_logits = []
    pred_end_logits = []
    for batch_start_idx in tqdm(
        list(range(0, len(test_data), batch_size)), desc="Evaluating..."
    ):
        batch = test_data[batch_start_idx : batch_start_idx + batch_size]
        outputs = m(**{c: batch[c].to(device) for c in forward_columns})
        pred_start_logits.extend(outputs.start_logits.detach().cpu().numpy())
        pred_end_logits.extend(outputs.end_logits.detach().cpu().numpy())
        gold_start_toks.extend(batch["start_positions"].detach().tolist())
        all_gold_start_toks.extend(batch["all_start_positions"])
        gold_end_toks.extend(batch["end_positions"].detach().tolist())
        all_gold_end_toks.extend(batch["all_end_positions"])

    (
        ex_by_instance_id,
        pred_args_by_ex,
        gold_args_by_ex,
        all_pred_scores,
    ) = extract_qa_spans(
        pred_start_logits,
        pred_end_logits,
        all_gold_start_toks,
        all_gold_end_toks,
        test_data,
        top_k,
        span_thresh,
    )
    # if an explicit threshold is provided (which it always should be for test),
    # we use that instead of sweeping all thresholds
    if span_thresh > 0:
        thresholds = [span_thresh]
    else:
        thresholds = np.quantile(all_pred_scores, QA_SPAN_PERCENTILES)
    best_ret = {"f1": float("-inf")}
    ref_events = []
    ref_events_by_dataset = defaultdict(list)
    ref_events_by_type = defaultdict(list)
    for k, v in sorted(gold_args_by_ex.items()):
        dataset = k.split("__")[0]
        event_type = k.split("__")[1]
        args = {role: val.keys() for role, val in v.items()}
        dat = [to_dataclass(k, args)]
        ref_events.append(dat)
        ref_events_by_dataset[dataset].append(dat)
        ref_events_by_type[f"{dataset}::{event_type}"].append(dat)

    for thresh in thresholds:
        predictions = []
        for pred_k, pred_v in pred_args_by_ex.items():
            gold_v = gold_args_by_ex[pred_k]
            pred_v = {
                k: [
                    a
                    for a in sorted(v.items(), key=lambda x: x[1], reverse=True)
                    if a[1] >= thresh
                ]
                for k, v in pred_v.items()
            }
            gold_v = {
                k: sorted(v.items(), key=lambda x: x[1], reverse=True)
                for k, v in gold_v.items()
            }
            predictions.append(
                {
                    "instance_id": pred_k,
                    "prediction": pred_v,
                    "reference": gold_v,
                }
            )

        # TODO: should just call compute_qa_metrics here,
        #       but method signature is wrong, unfortunately
        result = {}
        pred_events = []
        pred_events_by_dataset = defaultdict(list)
        pred_events_by_type = defaultdict(list)
        for k, v in sorted(gold_args_by_ex.items()):
            dataset = k.split("__")[0]
            event_type = k.split("__")[1]
            pred_event = [
                to_dataclass(
                    k,
                    {
                        role: {v_k for v_k, v_v in val.items() if v_v >= thresh}
                        for role, val in pred_args_by_ex[k].items()
                    },
                )
            ]
            pred_events.append(pred_event)
            pred_events_by_dataset[dataset].append(pred_event)
            pred_events_by_type[f"{dataset}::{event_type}"].append(pred_event)

        ceaf_ree_phi_subset = ceaf_ree_phi_subset_metrics.new()
        ceaf_ree_phi_subset.update_batch(pred_events, ref_events)
        ret = ceaf_ree_phi_subset.compute()
        if ret["f1"] > best_ret["f1"]:
            best_ret = ret
            best_thresh = thresh
            best_preds = predictions
            best_preds_by_dataset = pred_events_by_dataset
            best_preds_by_type = pred_events_by_type

    for k, v in best_ret.items():
        result["ceaf_ree_phi_subset_" + METRIC_MAP[k]] = v

    for dataset, events in best_preds_by_dataset.items():
        metric = ceaf_ree_phi_subset_metrics.new()
        metric.update_batch(events, ref_events_by_dataset[dataset])
        ret = metric.compute()
        for k, v in ret.items():
            result[dataset + "_ceaf_ree_phi_subset_" + METRIC_MAP[k]] = v

    if verbose_metrics:
        for event_type, events in best_preds_by_type.items():
            metric = ceaf_ree_phi_subset_metrics.new()
            metric.update_batch(events, ref_events_by_type[event_type])
            ret = metric.compute()
            for k, v in ret.items():
                result[event_type + "_ceaf_ree_phi_subset_" + METRIC_MAP[k]] = v

    result["best_threshold"] = best_thresh
    pprint(f"Best threshold: {best_thresh}")
    pprint(result)
    results_path = os.path.join(model_path, results_dir)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    predictions_file = os.path.join(results_path, f"{split}_preds.jsonl")
    with open(predictions_file, "w") as f:
        f.write("\n".join(list(map(json.dumps, best_preds))))

    with open(os.path.join(results_path, f"{split}_preds_pretty.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    scores_file = os.path.join(results_path, f"{split}_metrics.json")
    with open(scores_file, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    inference()
