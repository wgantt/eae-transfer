import logging
import numpy as np
import os
import sys

from argparse import Namespace
from collections import defaultdict
from datasets import Dataset
from tempfile import NamedTemporaryFile
from transformers import EvalPrediction, PreTrainedTokenizer
from transformers.pipelines.question_answering import select_starts_ends
from typing import Dict, List, Mapping, Set, Tuple

from eae.dataset.common import DatasetChoice, TaskChoice
from eae.eval.common import (
    to_dataclass,
    extract_events_from_str,
    extract_events_from_reference,
    extract_event_from_infilling_template,
    trigger_metrics,
    ceaf_ree_phi_subset_metrics,
)
from eae.eval.wikievents.scorer import score_wikievents
from eae.eval.rams.predictions_to_rams_format import convert as convert_to_rams_format
from eae.eval.rams.scorer import run_evaluation as run_rams_evaluation

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


METRIC_MAP = {"precision": "p", "recall": "r", "f1": "f1"}
QA_SPAN_PERCENTILES = [0.05 * i for i in range(1, 20)]


def extract_qa_spans(
    pred_start_logits: np.ndarray,
    pred_end_logits: np.ndarray,
    gold_start_toks: List[List[int]],
    gold_end_toks: List[List[int]],
    eval_dataset: Dataset,
    top_k: int = 1,
    span_thresh: float = float("-inf"),
) -> Tuple[
    Mapping[str, Mapping[str, Set[str]]], Mapping[str, Mapping[str, Set[str]]], float
]:
    context_mask = ~np.array(eval_dataset["context_mask"], dtype=bool)
    attention_mask = np.array(eval_dataset["attention_mask"], dtype=bool)
    token_to_word_map = eval_dataset["token_to_word_map"]
    pred_args_by_ex = defaultdict(
        lambda: defaultdict(dict)
    )  # example -> role -> span -> score
    gold_args_by_ex = defaultdict(lambda: defaultdict(dict))
    ex_by_instance_id = {}
    all_pred_scores = []
    # Get top-k predicted spans for each example
    for i in range(len(eval_dataset)):
        ex_dataset = eval_dataset[i]["dataset"]
        start_logits = pred_start_logits[i][None]
        end_logits = pred_end_logits[i][None]
        attn_mask = attention_mask[i][None]
        p_mask = context_mask[i][None]
        tokens = eval_dataset[i]["tokens"]
        instance_id = (
            ex_dataset + "__" + eval_dataset[i]["instance_id"].split("__")[0]
        )  # remove role_type
        ex_by_instance_id[instance_id] = i
        role_type = eval_dataset[i]["role_type"]
        tok2word = token_to_word_map[i]
        starts, ends, scores, min_null_score = select_starts_ends(
            start_logits,
            end_logits,
            attention_mask=attn_mask,
            p_mask=p_mask,
            top_k=top_k,
            handle_impossible_answer=True,
        )
        # Accumulate predicted and gold spans for each event.
        # Only include a span if its score exceeds that of the null span
        curr_span_thresh = max(span_thresh, min_null_score)
        for start, end, score in sorted(zip(starts, ends, scores)):
            if score > curr_span_thresh:
                pred_toks = tokens[tok2word[start] : tok2word[end]]
                pred_span = " ".join(pred_toks)
                if pred_span == "":  # update span threshold
                    curr_span_thresh = max(curr_span_thresh, score)
                    continue  # drop the null (=no answer) span
                prev_score = pred_args_by_ex[instance_id][role_type].get(
                    pred_span, -1.0
                )
                pred_args_by_ex[instance_id][role_type][pred_span] = max(
                    prev_score, float(score)
                )
                all_pred_scores.append(score)
        if not pred_args_by_ex[instance_id][role_type]:
            pred_args_by_ex[instance_id][role_type] = {}

        for start, end in zip(gold_start_toks[i], gold_end_toks[i]):
            gold_toks = tokens[tok2word[start] : tok2word[end]]
            gold_span = " ".join(gold_toks)
            gold_args_by_ex[instance_id][role_type][
                gold_span
            ] = 1.0  # gold span always gets score of 1.0

        if not gold_args_by_ex[instance_id][role_type]:
            gold_args_by_ex[instance_id][role_type] = {}

    return ex_by_instance_id, pred_args_by_ex, gold_args_by_ex, sorted(all_pred_scores)


def compute_qa_metrics(
    eval_pred: EvalPrediction,
    eval_dataset_names: Set[str],
    eval_dataset: Dataset,
    eval_ref_files: Dict[str, str] = {},
    eval_coref_files: Dict[str, str] = {},
    span_thresh: float = float("-inf"),
    top_k: int = 1,
    verbose_metrics: bool = False,
) -> Dict[str, float]:
    pred_start_logits = eval_pred.predictions[0]
    pred_end_logits = eval_pred.predictions[1]
    gold_start_toks = eval_dataset["all_start_positions"]
    gold_end_toks = eval_dataset["all_end_positions"]
    (
        ex_by_instance_id,
        pred_args_by_ex,
        gold_args_by_ex,
        all_pred_scores,
    ) = extract_qa_spans(
        pred_start_logits,
        pred_end_logits,
        gold_start_toks,
        gold_end_toks,
        eval_dataset,
        top_k,
        span_thresh,
    )
    result = {}

    # CEAF-REE
    ref_events = []
    ref_events_by_dataset = defaultdict(list)
    ref_events_by_type = defaultdict(list)
    for k, v in sorted(gold_args_by_ex.items()):
        ref_event = to_dataclass(k, {role: val.keys() for role, val in v.items()})
        dataset = k.split("__")[0]
        event_type = k.split("__")[1]
        ref_events.append([ref_event])
        ref_events_by_dataset[dataset].append([ref_event])
        ref_events_by_type[f"{dataset}::{event_type}"].append([ref_event])

    if not all_pred_scores:
        logger.warning("No argument spans predicted!")
        return {"ALL_ceaf_ree_phi_subset_" + k: 0.0 for k in METRIC_MAP.values()}

    if span_thresh == float("-inf"):
        thresholds = np.quantile(all_pred_scores, QA_SPAN_PERCENTILES)
    else:
        thresholds = [span_thresh]

    best_ret = {"f1": float("-inf")}
    best_thresh = float("-inf")
    for thresh in thresholds:
        pred_events = [
            [
                to_dataclass(
                    k,
                    {
                        role: [v_k for v_k, v_v in val.items() if v_v >= thresh]
                        for role, val in pred_args_by_ex[k].items()
                    },
                )
            ]
            for k, v in sorted(gold_args_by_ex.items())
        ]
        ceaf_ree_phi_subset = ceaf_ree_phi_subset_metrics.new()
        ceaf_ree_phi_subset.update_batch(pred_events, ref_events)
        ret = ceaf_ree_phi_subset.compute()
        # TODO: should probably support early stopping based on
        # a single dataset, rather than just the aggregate score
        if ret["f1"] > best_ret["f1"]:
            best_ret = ret
            best_thresh = thresh
            best_pred_events = pred_events

    # group predicted events by dataset
    best_pred_events_by_dataset = defaultdict(list)
    best_pred_events_by_type = defaultdict(list)
    for pred_events in best_pred_events:
        # the dataset for this event is given before the __
        # in the trigger name
        dataset = pred_events[0].trigger.mention.split("__")[0]
        event_type = pred_events[0].trigger.mention.split("__")[1]
        best_pred_events_by_dataset[dataset].append(pred_events)
        best_pred_events_by_type[f"{dataset}::{event_type}"].append(pred_events)

    # metrics by dataset
    for dataset, events in best_pred_events_by_dataset.items():
        metric = ceaf_ree_phi_subset_metrics.new()
        metric.update_batch(events, ref_events_by_dataset[dataset])
        for k, v in metric.compute().items():
            result[dataset + "_ceaf_ree_phi_subset_" + METRIC_MAP[k]] = v

    if verbose_metrics:
        for event_type, events in best_pred_events_by_type.items():
            metric = ceaf_ree_phi_subset_metrics.new()
            metric.update_batch(events, ref_events_by_type[event_type])
            for k, v in metric.compute().items():
                result[event_type + "_ceaf_ree_phi_subset_" + METRIC_MAP[k]] = v

    # aggregate metrics across all datasets
    for k3, v in best_ret.items():
        result["ALL_ceaf_ree_phi_subset_" + METRIC_MAP[k3]] = v
    result["best_threshold"] = best_thresh
    return result


def compute_metrics(
    eval_pred: EvalPrediction,
    eval_dataset_names: Set[str],
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    task: TaskChoice,
    eval_ref_files: Dict[str, str] = {},
    eval_coref_files: Dict[str, str] = {},
    verbose_metrics: bool = False,
) -> Dict[str, float]:
    """Compute OTE metrics, including Trigger P/R/F, and argument P/R/F (CEAF-REE)

    :param eval_pred: the predictions to be evaluated
    :param eval_dataset_names: the names of all the datasets on which the predictions were made.
        Options are given by dataset.DatasetChoice.
    :param eval_dataset: the dataset on which the predictions were made
    :param tokenizer: the tokenizer associated with the model
    :param task: The type of task being evaluated. Options are given by dataset.TaskChoice.
    :param eval_ref_files: the gold reference file for WikiEvents or RAMS evaluations
    :param eval_coref_files: the gold coreference file for WikiEvents evaluation
    :param verbose_metrics: if true, will print per-event type metrics
    :return: a dictionary containing various metrics
    """
    # We allow the predictions either to be a Transformers
    # EvalPrediction object or else just a list of decoded strings.
    if isinstance(eval_pred, EvalPrediction):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
        ]
    else:
        assert isinstance(eval_pred, list), "Expected list of strings or EvalPrediction"
        assert isinstance(
            eval_pred[0], str
        ), "Expected list of strings or EvalPrediction"
        prediction_lens = [len(tokenizer.tokenize(p)) for p in eval_pred]
        decoded_preds = eval_pred

    total_pred_events = 0
    total_pred_errors = 0
    pred_events_by_dataset = defaultdict(list)
    pred_events_by_type = defaultdict(list)
    ref_events_by_dataset = defaultdict(list)
    ref_events_by_type = defaultdict(list)
    for r, p in zip(eval_dataset, decoded_preds):
        if task == TaskChoice.INFILLING:
            trigger = r["event"]["trigger"]["text"]
            role_types = set(r["role_types"])
            unfilled_template = r["template"]
            predicted_template = p
            pred_events = [
                extract_event_from_infilling_template(
                    "", role_types, unfilled_template, predicted_template
                )
            ]
            ref_events = extract_events_from_reference(r, do_eae=True)
            assert len(ref_events) == 1
            num_events = 1
            num_errors = 0  # NOTE: we don't have a good way of ID'ing errors here
        else:
            pred_events, num_events, num_errors = extract_events_from_str(
                p, task == TaskChoice.EAE
            )
            ref_events = extract_events_from_reference(r, task == TaskChoice.EAE)
        ref_events_by_dataset["ALL"].append(ref_events)
        ref_events_by_dataset[r["dataset"].upper()].append(ref_events)
        ref_events_by_type[f"{r['dataset']}::{r['event_type']}"].append(ref_events)
        total_pred_events += num_events
        total_pred_errors += num_errors
        pred_events_by_dataset["ALL"].append(pred_events)
        pred_events_by_dataset[r["dataset"].upper()].append(pred_events)
        pred_events_by_type[f"{r['dataset']}::{r['event_type']}"].append(pred_events)
    logger.warning(
        f"{total_pred_errors}/{total_pred_events} predicted events could not be decoded due to parsing errors"
    )
    result = {}

    # compute trigger p, r, f1
    for k1, pred_events in pred_events_by_dataset.items():
        ref_events = ref_events_by_dataset[k1]
        trigger = trigger_metrics.new()
        trigger.update_batch(pred_events, ref_events)
        ret = trigger.compute()
        for k2, v in ret.items():
            result[k1 + "_trigger_" + METRIC_MAP[k2]] = v

        # compute arg p, r, f1 (ceaf-ree)
        has_some_args = any([len(e.args) > 0 for es in pred_events for e in es])
        if has_some_args:
            ceaf_ree_phi_subset = ceaf_ree_phi_subset_metrics.new()
            ceaf_ree_phi_subset.update_batch(pred_events, ref_events)
            ret = ceaf_ree_phi_subset.compute()
            for k3, v in ret.items():
                result[k1 + "_ceaf_ree_phi_subset_" + METRIC_MAP[k3]] = v
        else:
            result[k1 + "_ceaf_ree_phi_subset_p"] = 0.0
            result[k1 + "_ceaf_ree_phi_subset_r"] = 0.0
            result[k1 + "_ceaf_ree_phi_subset_f1"] = 0.0

    if verbose_metrics:
        for k1, pred_events in pred_events_by_type.items():
            ref_events = ref_events_by_type[k1]
            trigger = trigger_metrics.new()
            trigger.update_batch(pred_events, ref_events)
            ret = trigger.compute()
            for k2, v in ret.items():
                result[k1 + "_trigger_" + METRIC_MAP[k2]] = v

            # compute arg p, r, f1 (ceaf-ree)
            has_some_args = any([len(e.args) > 0 for es in pred_events for e in es])
            if has_some_args:
                ceaf_ree_phi_subset = ceaf_ree_phi_subset_metrics.new()
                ceaf_ree_phi_subset.update_batch(pred_events, ref_events)
                ret = ceaf_ree_phi_subset.compute()
                for k3, v in ret.items():
                    result[k1 + "_ceaf_ree_phi_subset_" + METRIC_MAP[k3]] = v
            else:
                result[k1 + "_ceaf_ree_phi_subset_p"] = 0.0
                result[k1 + "_ceaf_ree_phi_subset_r"] = 0.0
                result[k1 + "_ceaf_ree_phi_subset_f1"] = 0.0

    # if DatasetChoice.WIKIEVENTS.name in eval_dataset_names:
    #     assert (
    #         DatasetChoice.WIKIEVENTS.name in eval_ref_files
    #     ), "Must provide eval_ref_file for WikiEvents scoring"
    #     wikievents_ref_file = eval_ref_files[DatasetChoice.WIKIEVENTS.name]
    #     assert os.path.isfile(
    #         wikievents_ref_file
    #     ), f"Could not find {wikievents_ref_file}"
    #     assert (
    #         DatasetChoice.WIKIEVENTS.name in eval_coref_files
    #     ), "Must provide eval_coref_file for WikiEvents scoring"
    #     wikievents_coref_file = eval_coref_files[DatasetChoice.WIKIEVENTS.name]
    #     assert os.path.isfile(
    #         wikievents_coref_file
    #     ), f"Could not find {wikievents_coref_file}"
    #     preds_for_wikievents_scoring = []
    #     for ex, pred in zip(eval_dataset, decoded_preds):
    #
    #
    #                 {
    #                     "instance_id": ex["instance_id"],
    #                     "prediction": pred,
    #                     "events": ex["event"] if do_eae else ex["events"],
    #                 }
    #             )
    #     wikievents_metrics = score_wikievents(
    #         preds_for_wikievents_scoring,
    #         wikievents_ref_file,
    #         wikievents_coref_file,
    #         coref=True,
    #         dataset="KAIROS",
    #     )
    #     result = result | wikievents_metrics

    # elif DatasetChoice.RAMS.name in eval_dataset_names:
    #     assert (
    #         DatasetChoice.RAMS.name in eval_ref_files
    #     ), "Must provide eval_ref_file for RAMS scoring"
    #     rams_ref_file = eval_ref_files[DatasetChoice.RAMS.name]
    #     assert os.path.isfile(rams_ref_file), f"Could not find {rams_ref_file}"

    #     preds_for_rams_scoring = []
    #     for ex, pred in zip(eval_dataset, decoded_preds):
    #         if ex["dataset"].upper() == DatasetChoice.RAMS.name:
    #             preds_for_rams_scoring.append(
    #                 {"instance_id": ex["instance_id"], "prediction": pred}
    #             )
    #     predictions_file = NamedTemporaryFile()
    #     convert_to_rams_format(
    #         predictions_file.name, rams_ref_file, preds_for_rams_scoring
    #     )
    #     # hard-coding these settings for now
    #     rams_eval_args = Namespace()
    #     rams_eval_args.gold_file = rams_ref_file
    #     rams_eval_args.pred_file = predictions_file.name
    #     rams_eval_args.ontology_file = None
    #     rams_eval_args.cd = False
    #     rams_eval_args.metrics = True
    #     rams_eval_args.do_all = False
    #     rams_eval_args.distance = False
    #     rams_eval_args.role_table = False
    #     rams_eval_args.confusion = False
    #     rams_eval_args.reuse_gold_format = True
    #     rams_metrics = run_rams_evaluation(rams_eval_args)
    #     out = {}
    #     metric_abbreviations = {
    #         "precision": "rams_p",
    #         "recall": "rams_r",
    #         "f1": "rams_f1",
    #     }
    #     for k, v in rams_metrics["metrics"].items():
    #         out[metric_abbreviations[k]] = v
    #     result = result | out

    result["mean_output_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}
