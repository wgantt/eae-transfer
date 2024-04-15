import click
import datasets
import json
import logging
import numpy as np
import os
import sys
import transformers

from dataclasses import asdict
from datasets import Dataset
from functools import partial
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import PredictionOutput

from eae.dataset.dataset import *
from eae.dataset.common import TaskChoice, EVENT_SEP
from eae.eval.eval import compute_metrics, extract_event_from_infilling_template
from eae.preprocessing.preprocess_infilling import *

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

SUPPORTED_MODELS = {
    "facebook/bart-base",
    "facebook/bart-large",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/pegasus-large",
    "t5-large",
    "t5-base",
}
DEFAULT_MODEL = "facebook/bart-large"

MIN_OUTPUT_LENGTH = 0
MAX_OUTPUT_LENGTH = 128


@click.command()
@click.argument(
    "output_dir",
    type=str,
)
@click.argument("train_dataset", type=str)
@click.argument("dev_dataset", type=str)
@click.argument("test_dataset", type=str)
@click.option(
    "--model",
    "-m",
    type=click.Choice(SUPPORTED_MODELS),
    default=DEFAULT_MODEL,
    help="the pretrained model to fine-tune",
)
@click.option(
    "--num-epochs", type=click.INT, default=30, help="maximum training epochs"
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
    help="the minimum number of tokens to generate in the output (for inference only)",
)
@click.option(
    "--max-new-tokens",
    type=click.INT,
    default=MAX_OUTPUT_LENGTH,
    help="maximum number of tokens to generate in the output (for inference only)",
)
@click.option(
    "--num-beams",
    type=click.INT,
    default=5,
    help="number of beams to use for beam search (eval loop only)",
)
@click.option(
    "--patience",
    type=click.INT,
    default=10,
    help="number of epochs to wait for improvement before early stopping",
)
@click.option(
    "--gradient-checkpointing",
    is_flag=True,
    default=False,
    help="whether to use gradient checkpointing for training",
)
@click.option("--per-device-batch-size", type=click.INT, default=8)
@click.option("--seed", type=int, default=1337, help="the random seed for training")
@click.option("--fp16", is_flag=True, default=False, help="whether to use fp16")
def train(
    train_dataset,
    dev_dataset,
    test_dataset,
    output_dir,
    model,
    num_epochs,
    max_doc_len,
    min_new_tokens,
    max_new_tokens,
    num_beams,
    patience,
    gradient_checkpointing,
    per_device_batch_size,
    seed,
    fp16,
) -> None:
    """Train a seq-to-seq infilling model for EAE

    :param train_dataset: the name of the training dataset to use
    :param dev_dataset: the name of the dev dataset to use
    :param test_dataset: the name of the test dataset to use
    :param output_dir: the directory where checkpoints will be saved
    :param model: a string indicating the HuggingFace base model to be fine-tuned
    :param num_epochs: the number of epochs for which training will be run
    :param max_doc_len: the maximum length of an input document (documents longer
        than this will be truncated)
    :param min_new_tokens: minimum number of tokens to generate in the output (for inference only)
    :param max_new_tokens: maximum number of tokens to generate in the output (for inference only)
    :param num_beams: number of beams to use for beam search (eval loop only)
    :param patience: number of epochs to wait for improvement before early stopping
    :param gradient_checkpointing: whether to use gradient checkpointing for training
    :param per_device_batch_size: batch size to use (for training and for dev evaluation)
    :param seed: the random seed to use
    :param fp16: whether to use fp16
    :return: None
    """
    # dump training and model parameters
    config = locals()
    config["output_dir"] = os.path.abspath(config.pop("output_dir"))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    m = AutoModelForSeq2SeqLM.from_pretrained(model)
    if "t5" in model:
        # default behavior of `from_pretrained` here is apparently incorrect for T5; see below:
        if model in {"t5-small", "t5-base"}:
            model_max_length = 512
        else:
            model_max_length = 1024
        tokenizer = AutoTokenizer.from_pretrained(
            model, model_max_length=model_max_length
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)

    tokenizer.add_tokens([EVENT_SEP, ROLE_SEP])
    m.resize_token_embeddings(len(tokenizer))

    for split, dnames in zip(
        ["train", "dev", "test"], [train_dataset, dev_dataset, test_dataset]
    ):
        for dname in dnames.split("+"):
            assert (
                dname.upper() in DatasetChoice._member_names_
            ), f'Dataset {dname.upper()} for split "{split}" not found. Must be one of {DatasetChoice._member_names_}'

    preprocess_fn = partial(
        preprocess,
        tokenizer=tokenizer,
        max_doc_len=max_doc_len,
        # special tokens are already added
        add_special_tokens=False,
    )

    # we support training on multiple datasets at once
    train_datasets = []
    for ds in train_dataset.split("+"):
        train_datasets.append(
            Dataset.from_generator(
                DATASET_TO_SPLITS[TaskChoice.INFILLING][DatasetChoice[ds.upper()]][
                    "train"
                ]
            ).map(preprocess_fn, batched=True)
        )
    train_dataset = datasets.interleave_datasets(train_datasets, seed=seed)

    # we don't yet support evaluating or testing on multiple datasets
    dev_dataset_names = [ds.upper() for ds in dev_dataset.split("+")]
    dev_datasets = []
    eval_ref_files = {}
    eval_coref_files = {}
    metric_for_best_model = None
    for ds in dev_dataset_names:
        dev_datasets.append(
            Dataset.from_generator(
                DATASET_TO_SPLITS[TaskChoice.INFILLING][DatasetChoice[ds.upper()]][
                    "dev"
                ]
            ).map(preprocess_fn, batched=True)
        )
        # if DatasetChoice[ds.upper()] == DatasetChoice.WIKIEVENTS:
        #     eval_ref_files[ds.upper()] = WIKIEVENTS_DATA_FILES["dev"]
        #     eval_coref_files[ds.upper()] = WIKIEVENTS_COREF_FILES["dev"]
        #     metric_for_best_model = "wikievents_coref_arg-c_f1"
        # elif DatasetChoice[ds.upper()] == DatasetChoice.RAMS:
        #     eval_ref_files[ds.upper()] = RAMS_DATA_FILES["dev"]
        #     metric_for_best_model = "rams_f1"

    # default stopping metric is CEAF-REE F1;
    # this is equivalent to argument F1 when no coref is available
    if not metric_for_best_model or len(dev_datasets) > 1:
        metric_for_best_model = "ALL_ceaf_ree_phi_subset_f1"
    logger.warning(f"Using {metric_for_best_model} for early stopping")

    # eval datasets get interleaved
    # (and are disaggregated in `compute_metrics`)
    eval_dataset = datasets.interleave_datasets(dev_datasets, seed=seed)

    test_dataset_names = [ds.upper() for ds in test_dataset.split("+")]
    test_datasets = []
    test_ref_files = {}
    test_coref_files = {}
    for ds in test_dataset_names:
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

    # test datasets just get concatenated
    # (and are disaggregated in `compute_metrics`)
    test_dataset = datasets.concatenate_datasets(test_datasets)

    # Load model's default generation config, but
    # override with user-provided parameters
    assert m.generation_config is not None
    generation_config = m.generation_config
    generation_config.min_new_tokens = min_new_tokens
    generation_config.max_new_tokens = max_new_tokens
    generation_config.num_beams = num_beams

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=num_epochs,
        output_dir=output_dir,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=gradient_checkpointing,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        predict_with_generate=True,
        generation_config=generation_config,
        seed=seed,
        fp16=fp16,
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    logger.warning(f"(Min, Max) output length: ({min_new_tokens}, {max_new_tokens})")
    logger.warning(f"Using beam size = {num_beams}")
    eval_metrics = partial(
        compute_metrics,
        eval_dataset_names=set(dev_dataset_names),
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        task=TaskChoice.INFILLING,
        eval_ref_files=eval_ref_files,
        eval_coref_files=eval_coref_files,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=m)
    trainer = Seq2SeqTrainer(
        model=m,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=eval_metrics,
    )
    if patience > 0:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer.train()

    test_metrics = partial(
        compute_metrics,
        eval_dataset_names=set(test_dataset_names),
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        task=TaskChoice.INFILLING,
        eval_ref_files=test_ref_files,
        eval_coref_files=test_coref_files,
        verbose_metrics=True,  # only print verbose metrics at the end
    )
    trainer.compute_metrics = test_metrics
    prediction_output = trainer.predict(test_dataset)
    save_predictions(prediction_output, test_dataset, tokenizer, output_dir)


def save_predictions(
    prediction_output: PredictionOutput,
    test_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    model_path: str,
) -> None:
    predictions = prediction_output.predictions
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    predictions = []
    assert len(decoded_preds) == len(test_dataset)
    for i, pred in enumerate(decoded_preds):
        trigger = test_dataset[i]["event"]["trigger"]["text"]
        role_types = set(test_dataset[i]["role_types"])
        unfilled_template = test_dataset[i]["template"]
        pred_event = extract_event_from_infilling_template(
            trigger, role_types, unfilled_template, pred
        )
        ref_event = extract_event_from_infilling_template(
            trigger, role_types, unfilled_template, test_dataset[i]["targets"]
        )
        pred_output = {
            "instance_id": test_dataset[i]["instance_id"],
            "document": test_dataset[i]["text"],
            "input_str": test_dataset[i]["input_str"],
            "prediction": pred,
            "pred_event": asdict(pred_event),
            "reference": test_dataset[i]["targets"],
            "ref_event": asdict(ref_event),
            "template": test_dataset[i]["template"],
            "role_types": test_dataset[i]["role_types"],
        }
        pred_output["events"] = test_dataset[i]["event"]
        predictions.append(pred_output)
    with open(os.path.join(model_path, "test_preds.jsonl"), "w") as f:
        f.write("\n".join(list(map(json.dumps, predictions))))

    with open(os.path.join(model_path, "test_preds_pretty.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    metrics = prediction_output.metrics
    with open(os.path.join(model_path, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    train()
