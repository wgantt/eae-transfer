import click
import json
import logging
import os
import sys
import torch

from dataclasses import asdict
from datasets import Dataset, concatenate_datasets
from functools import partial
from pprint import pprint
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from eae.dataset.dataset import (
    DatasetChoice,
    TaskChoice,
    DATASET_TO_SPLITS,
)
from eae.eval.eval import (
    compute_metrics,
    extract_events_from_reference,
    extract_event_from_infilling_template,
)
from eae.preprocessing.preprocess_infilling import (
    preprocess,
)
from eae.training.train_infilling import MIN_OUTPUT_LENGTH, MAX_OUTPUT_LENGTH

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
    "--beam-size",
    type=click.INT,
    default=5,
    help="the beam size for beam search decoding",
)
@click.option(
    "--num-return-sequences",
    type=click.INT,
    default=1,
    help="number of candidate outputs to generate",
)
@click.option(
    "--max-doc-len",
    type=click.INT,
    default=None,
    help="maximum number of tokens in the input document (all longer docs will be truncated)",
)
@click.option(
    "--min-new-tokens",
    type=click.INT,
    default=MIN_OUTPUT_LENGTH,
    help="the minimum number of tokens to generate in the output",
)
@click.option(
    "--max-new-tokens",
    type=click.INT,
    default=MAX_OUTPUT_LENGTH,
    help="maximum number of tokens to generate in the output",
)
@click.option(
    "--results-dir",
    type=click.STRING,
    default="..",
    help="directory where results will be saved",
)
@click.option("--seed", type=click.INT, default=42, help="random seed")
@click.option(
    "--verbose-metrics",
    is_flag=True,
    help="Will print per-event type results if True",
)
def inference(
    test_datasets,
    model_path,
    device,
    batch_size,
    beam_size,
    min_new_tokens,
    max_new_tokens,
    num_return_sequences,
    max_doc_len,
    results_dir,
    seed,
    verbose_metrics,
) -> None:
    """Run inference for infilling-based event argument extraction

    :param test_datasets: name of the test datasets to run inference on
        (should be a "+"-separated list of dataset names)
    :param model_path: path to the model with which to run inference
    :param device: The GPU on which the model will be loaded
    :param batch_size: The batch size to use
    :param beam_size: The number of beams to use for beam search
    :param min_new_tokens: minimum number of tokens to generate in the output
    :param max_new_tokens: maximum number of tokens to generate in the output
    :param num_return_sequences: The number of candidate outputs to generate
        for each input
    :param max_doc_len: maximum number of tokens in the input document
        (all longer docs will be truncated)
    :param results_dir: where model predictions and metrics will be written
    :param seed: random seed
    :param verbose_metrics: will print per-event type results if True
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
    m = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    preprocess_fn = partial(
        preprocess,
        tokenizer=tokenizer,
        max_doc_len=max_doc_len,
        # special tokens are already added
        add_special_tokens=False,
    )
    test_dataset_names = [ds.upper() for ds in test_datasets.split("+")]
    test_datasets = []
    test_ref_files = {}
    test_coref_files = {}
    for ds in test_dataset_names:
        assert ds.upper() in DatasetChoice._member_names_, (
            f"Invalid test dataset {ds.upper()}. "
            f"Must be one of {DatasetChoice._member_names_}"
        )
        test_datasets.append(
            Dataset.from_generator(
                DATASET_TO_SPLITS[TaskChoice.INFILLING][DatasetChoice[ds.upper()]][
                    "test"
                ]
            ).map(preprocess_fn, batched=True)
        )
        # if DatasetChoice[ds.upper()] == DatasetChoice.WIKIEVENTS:
        #     test_ref_files[ds.upper()] = WIKIEVENTS_DATA_FILES["test"]
        #     test_coref_files[ds.upper()] = WIKIEVENTS_COREF_FILES["test"]
        # elif DatasetChoice[ds.upper()] == DatasetChoice.RAMS:
        #     test_ref_files[ds.upper()] = RAMS_DATA_FILES["test"]

    test_data = concatenate_datasets(test_datasets)

    preds = []
    for batch_start_idx in tqdm(
        list(range(0, len(test_data), batch_size)), desc="Evaluating..."
    ):
        batch = test_data[batch_start_idx : batch_start_idx + batch_size]
        input_ids = tokenizer(
            text=batch["formatted_doc"],
            text_pair=batch["template"],
            padding="max_length",
            truncation="only_first",
            return_tensors="pt",
        )["input_ids"]

        input_ids = input_ids.to(device)
        outputs = m.generate(
            input_ids,
            num_beams=beam_size,
            num_return_sequences=num_return_sequences,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            suppress_tokens=None,
        )
        preds += list(
            map(lambda x: tokenizer.decode(x, skip_special_tokens=True), outputs)
        )

    scores_dict = compute_metrics(
        preds,
        test_dataset_names,
        test_data,
        tokenizer,
        TaskChoice.INFILLING,
        eval_ref_files=test_ref_files,
        eval_coref_files=test_coref_files,
        verbose_metrics=verbose_metrics,
    )
    pprint(scores_dict)

    predictions = []
    for ex, pred in zip(test_data, preds):
        trigger = ex["event"]["trigger"]["text"]
        role_types = set(ex["role_types"])
        unfilled_template = ex["template"]
        pred_event = extract_event_from_infilling_template(
            trigger, role_types, unfilled_template, pred
        )
        ref_event = extract_events_from_reference(ex, do_eae=True)
        if len(ref_event) > 1:
            logger.warning(
                f"Found multiple events in reference for instance {ex['instance_id']}"
            )
        ref_event = ref_event[0]
        pred_output = {
            "instance_id": ex["instance_id"],
            "document": ex["text"],
            "input_str": ex["input_str"],
            "prediction": pred,
            "pred_event": asdict(pred_event),
            "reference": ex["targets"],
            "ref_event": asdict(ref_event),
            "template": ex["template"],
            "role_types": ex["role_types"],
        }
        pred_output["events"] = ex["event"]
        predictions.append(pred_output)

    results_path = os.path.join(model_path, results_dir)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    predictions_file = os.path.join(results_path, "test_preds.jsonl")
    with open(predictions_file, "w") as f:
        f.write("\n".join(list(map(json.dumps, predictions))))

    with open(os.path.join(results_path, "test_preds_pretty.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    scores_file = os.path.join(results_path, "test_metrics.json")
    with open(scores_file, "w") as f:
        json.dump(scores_dict, f, indent=2)


if __name__ == "__main__":
    inference()
