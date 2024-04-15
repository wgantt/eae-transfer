import click
import datasets
import json
import logging
import os
import sys
import transformers

from datasets import Dataset
from functools import partial
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import PredictionOutput

from eae.dataset.dataset import *
from eae.eval.eval import compute_qa_metrics, extract_qa_spans
from eae.preprocessing.preprocess_qa import *
from pprint import pprint

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

SUPPORTED_MODELS = {
    "facebook/bart-base",
    "facebook/bart-large",
    "bert-base-cased",
    "bert-large-cased",
    "t5-base",
    "t5-large",
    "google/flan-t5-base",
    "google/flan-t5-large",
}
DEFAULT_MODEL = "bert-large-cased"


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
    "--top-k",
    type=click.INT,
    default=1,
    help="number of candidate spans to predict for each example",
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
@click.option("--seed", type=int, default=1337, help="the random seed for training")
@click.option("--fp16", is_flag=True, default=False, help="whether to use fp16")
@click.option("--per-device-train-batch-size", type=click.INT, default=8)
@click.option("--per-device-eval-batch-size", type=click.INT, default=8)
@click.option("--grad-accumulation-steps", type=click.INT, default=1)
def train(
    train_dataset,
    dev_dataset,
    test_dataset,
    output_dir,
    model,
    num_epochs,
    max_doc_len,
    top_k,
    patience,
    gradient_checkpointing,
    seed,
    fp16,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    grad_accumulation_steps,
) -> None:
    """Train a QA model for EAE

    :param train_dataset: the name of the training dataset to use
    :param dev_dataset: the name of the dev dataset to use
    :param test_dataset: the name of the test dataset to use
    :param output_dir: the directory where checkpoints will be saved
    :param model: a string indicating the HuggingFace base model to be fine-tuned
    :param num_epochs: the number of epochs for which training will be run
    :param max_doc_len: the maximum length of an input document (documents longer
        than this will be truncated)
    :param patience: number of epochs to wait for improvement before early stopping
    :param gradient_checkpointing: whether to use gradient checkpointing for training
    :param seed: the random seed to use
    :param fp16: whether to use fp16
    :param per_device_train_batch_size: the batch size per device for training
    :param per_device_eval_batch_size: the batch size per device for evaluation
    :param grad_accumulation_steps: determines the effective batch size
    :return: None
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # dump training and model parameters
    config = locals()
    config["output_dir"] = os.path.abspath(config.pop("output_dir"))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    m = AutoModelForQuestionAnswering.from_pretrained(model)
    if "t5" in model:
        # default behavior of `from_pretrained` here is apparently incorrect for T5; see below:
        if "t5-base" in model or "t5-small" in model:
            model_max_length = 512
        else:
            model_max_length = 1024
        tokenizer = AutoTokenizer.from_pretrained(
            model, model_max_length=model_max_length
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model, add_prefix_space="bart" in model
        )
    tokenizer.add_tokens(TOKENS_TO_ADD)
    m.resize_token_embeddings(len(tokenizer))

    for split, dnames in zip(
        ["train", "dev", "test"], [train_dataset, dev_dataset, test_dataset]
    ):
        for dname in dnames.split("+"):
            assert (
                dname.upper() in DatasetChoice._member_names_
            ), f'Dataset {dname.upper()} for split "{split}" not found. Must be one of {DatasetChoice._member_names_}'

    task = TaskChoice.QA

    # we support training on multiple datasets at once
    train_datasets = []
    train_dataset_names = []
    for ds in train_dataset.split("+"):
        train_dataset_names.append(ds.upper())
        data = Dataset.from_generator(
            DATASET_TO_SPLITS[task][DatasetChoice[ds.upper()]]["train"]
        )
        preprocess_fn = partial(
            preprocess,
            tokenizer=tokenizer,
            max_length=max_doc_len,
            # special tokens are already added
            add_special_tokens=False,
            columns_to_keep=data.column_names,
        )
        data = data.map(preprocess_fn, batched=True)
        train_datasets.append(data)
    train_dataset = datasets.interleave_datasets(train_datasets, seed=seed)

    # we don't yet support evaluating or testing on multiple datasets
    dev_dataset_names = [ds.upper() for ds in dev_dataset.split("+")]
    dev_datasets = []
    eval_ref_files = {}
    eval_coref_files = {}
    metric_for_best_model = None
    for ds in dev_dataset_names:
        data = Dataset.from_generator(
            DATASET_TO_SPLITS[task][DatasetChoice[ds.upper()]]["dev"]
        )
        preprocess_fn = partial(
            preprocess,
            tokenizer=tokenizer,
            max_length=max_doc_len,
            add_special_tokens=False,
            columns_to_keep=data.column_names,
        )
        data = data.map(preprocess_fn, batched=True)
        dev_datasets.append(data)

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
        data = Dataset.from_generator(
            DATASET_TO_SPLITS[task][DatasetChoice[ds.upper()]]["test"]
        )
        preprocess_fn = partial(
            preprocess,
            tokenizer=tokenizer,
            max_length=max_doc_len,
            add_special_tokens=False,
            columns_to_keep=data.column_names,
        )
        data = data.map(preprocess_fn, batched=True)
        test_datasets.append(data)
        if DatasetChoice[ds.upper()] == DatasetChoice.WIKIEVENTS:
            test_ref_files[ds.upper()] = WIKIEVENTS_DATA_FILES["test"]
            test_coref_files[ds.upper()] = WIKIEVENTS_COREF_FILES["test"]
        elif DatasetChoice[ds.upper()] == DatasetChoice.RAMS:
            test_ref_files[ds.upper()] = RAMS_DATA_FILES["test"]

    # test datasets just get concatenated
    # (and are disaggregated in `compute_metrics`)
    test_dataset = datasets.concatenate_datasets(test_datasets)

    training_args = TrainingArguments(
        num_train_epochs=num_epochs,
        output_dir=output_dir,
        load_best_model_at_end=True,
        include_inputs_for_metrics=True,
        metric_for_best_model=metric_for_best_model,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=gradient_checkpointing,
        gradient_accumulation_steps=grad_accumulation_steps,
        fp16=fp16,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        seed=seed,
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    eval_metrics = partial(
        compute_qa_metrics,
        eval_dataset_names=set(dev_dataset_names),
        eval_dataset=eval_dataset,
        eval_ref_files=eval_ref_files,
        eval_coref_files=eval_coref_files,
        top_k=top_k,
        verbose_metrics=False,
    )
    data_collator = DefaultDataCollator()
    trainer = Trainer(
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

    logger.warning(f"Starting QA training on the following datasets: {', '.join(train_dataset_names)} (random seed={seed})")
    trainer.train()

    # run evaluation one final time to get optimal dev threshold
    eval_ret = trainer.evaluate()
    # Since the `compute_metrics` function takes the relevant dataset as
    # input, we have to update it here (dev -> test). There's probably a
    # less janky way to do this, but leaving I'm leaving it for now...
    test_metrics = partial(
        compute_qa_metrics,
        eval_dataset_names=set(test_dataset_names),
        eval_dataset=test_dataset,
        eval_ref_files=test_ref_files,
        eval_coref_files=test_coref_files,
        top_k=top_k,
        span_thresh=eval_ret["eval_best_threshold"],
        verbose_metrics=True,
    )
    trainer.compute_metrics = test_metrics
    prediction_output = trainer.predict(test_dataset)
    pprint(prediction_output.metrics)
    save_predictions(
        prediction_output,
        test_dataset,
        output_dir,
        top_k=top_k,
        span_thresh=eval_ret["eval_best_threshold"],
    )


def save_predictions(
    prediction_output: PredictionOutput,
    test_dataset: Dataset,
    model_path: str,
    top_k: int = 1,
    span_thresh: float = float("-inf"),
) -> None:
    pred_start_logits = prediction_output.predictions[0]
    pred_end_logits = prediction_output.predictions[1]
    gold_start_toks = test_dataset["all_start_positions"]
    gold_end_toks = test_dataset["all_end_positions"]
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
        test_dataset,
        top_k,
        span_thresh,
    )
    predictions = []
    for pred_k, pred_v in pred_args_by_ex.items():
        gold_v = gold_args_by_ex[pred_k]
        pred_v = {
            k: sorted(
                {k_: v_ for k_, v_ in v.items() if v_ >= span_thresh}.items(),
                key=lambda x: x[1],
                reverse=True,
            )
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
    with open(os.path.join(model_path, "test_preds.jsonl"), "w") as f:
        f.write("\n".join(list(map(json.dumps, predictions))))

    with open(os.path.join(model_path, "test_preds_pretty.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    metrics = prediction_output.metrics
    metrics["threshold"] = span_thresh
    with open(os.path.join(model_path, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    train()
