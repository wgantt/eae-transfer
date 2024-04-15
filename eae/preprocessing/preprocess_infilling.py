import logging
import sys

from collections import defaultdict
from datasets.formatting.formatting import LazyBatch
from transformers import BatchEncoding, PreTrainedTokenizerBase
from typing import Optional

from eae.dataset.common import EVENT_SEP, ROLE_SEP, RAMS_ROLE_TO_UPPER
from eae.eval.common import normalize_text

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def preprocess(
    examples: LazyBatch,
    tokenizer: PreTrainedTokenizerBase,
    prefix: Optional[str] = None,
    max_doc_len: Optional[int] = None,
    add_special_tokens: bool = False,
) -> BatchEncoding:
    """Preprocess EAE data for template infilling

    :param examples: the examples to be preprocessed
    :param tokenizer: the tokenizer that will be used to tokenize each example
    :param prefix: an optional prefix to prepend to the document text (not used in paper experiments)
    :param max_doc_len: the maximum length of an input document (defaults to the
        maximum model length)
    :param add_special_tokens: if true, will add the special EVENT_SEP and ROLE_SEP tokens
    :return: the preprocessed data
    """
    if add_special_tokens:
        tokenizer.add_special_tokens([EVENT_SEP, ROLE_SEP])
    model_input_dim = tokenizer.model_max_length
    assert (
        not max_doc_len or max_doc_len <= model_input_dim
    ), f"Maximum document length ({max_doc_len}) > model input dimension ({model_input_dim})"

    # Always inform the user of the document length
    if max_doc_len:
        logger.warning(f"Maximum document length: {max_doc_len}")
    else:
        logger.warning(f"Maximum document length: {model_input_dim}")

    # We may want to prepend some prefix to the document text
    if prefix:
        docs = [prefix + doc for doc in examples["text"]]
    else:
        docs = examples["text"]

    def format_event_template(event, template, dataset) -> str:
        """Format an event as an infilled template"""
        assert len(event) == 1
        event = event[0]
        args_by_role = defaultdict(list)
        for arg in event["arguments"]:
            if dataset == "RAMS":
                role = RAMS_ROLE_TO_UPPER[arg["role"]]
            else:
                role = arg["role"]
            args_by_role[role].append(normalize_text(arg["text"]))

        for role, args in args_by_role.items():
            if role not in template:
                assert (
                    role.capitalize() in template
                ), f"{event['event_type']} does not have role {role} in template {template}"
                role = role.capitalize()
            arg_str = " AND ".join(args)
            template = template.replace(
                ROLE_SEP + role + ROLE_SEP, ROLE_SEP + arg_str + ROLE_SEP, 1
            )
        return template

    # Format output
    dataset = examples["dataset"][0].upper()
    event_inputs = [[e] for e in examples["event"]]
    targets = [
        format_event_template(event, template, dataset)
        for (event, template) in zip(event_inputs, examples["template"])
    ]
    model_inputs = tokenizer(
        text=docs,
        text_pair=examples["template"],
        padding="max_length",
        truncation="only_first",
    )

    # save the raw input format to a slot in the dataset (used during inference)
    input_str = tokenizer.batch_decode(model_inputs["input_ids"])
    labels = tokenizer(text_target=targets, truncation=True)

    model_inputs["targets"] = targets
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["formatted_doc"] = docs
    model_inputs["input_str"] = input_str
    return model_inputs