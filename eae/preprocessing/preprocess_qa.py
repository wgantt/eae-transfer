import logging
import sys

from datasets.formatting.formatting import LazyBatch
from transformers import BatchEncoding, PreTrainedTokenizerBase
from typing import List, Optional

from eae.dataset.common import EVENT_SEP, ROLE_SEP

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def preprocess(
    examples: LazyBatch,
    tokenizer: PreTrainedTokenizerBase,
    max_length: Optional[int] = None,
    doc_stride: Optional[int] = 256,
    columns_to_keep: List[str] = [],
    add_special_tokens: bool = True,
) -> BatchEncoding:
    """Preprocess EAE data for question answering 

    Substantial portions of this code are adapted from the 
    preprocessing code from the tutorial for HuggingFace QA models:
    https://github.com/huggingface/transformers/blob/8127f39624f587bdb04d55ab655df1753de7720a/examples/pytorch/question-answering/run_qa.py#L403

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

    # Currently, we only support models that add padding on the right
    pad_on_right = tokenizer.padding_side == "right"
    if not pad_on_right:
        raise NotImplementedError()

    max_length = max_length or tokenizer.model_max_length

    # Tokenize our examples with truncation and padding,
    # but keep the overflows using a stride. This results
    # in one example possible giving several features when
    # a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    if not tokenizer.cls_token and not tokenizer.bos_token:
        # If there is no CLS or BOS token by default, we
        # just add one, using
        cls_tok = tokenizer.additional_special_tokens[0]
        examples["role_question_tok"] = [
            [cls_tok] + q for q in examples["role_question_tok"]
        ]

    tokenized_examples = tokenizer(
        examples["role_question_tok" if pad_on_right else "tokens"],
        examples["tokens" if pad_on_right else "role_question_tok"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=tokenizer.max_len_sentences_pair,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        is_split_into_words=True,
        return_tensors="pt",
    )

    # Since one example might give us several features if it
    # has a long context, we need a map from a feature to its
    # corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["all_start_positions"] = []
    tokenized_examples["start_positions"] = []
    tokenized_examples["all_end_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["context_mask"] = []
    tokenized_examples["token_to_word_map"] = []

    for i in range(len(sample_mapping)):
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]

        # We will label impossible answers with the index of the CLS token.
        tokenized_ex = tokenized_examples[i]
        input_ids = tokenized_ex.ids
        if tokenizer.cls_token_id:
            cls_index = input_ids.index(tokenizer.cls_token_id)
        elif tokenizer.bos_token_id:
            cls_index = input_ids.index(tokenizer.bos_token_id)
        else:
            # this is the case where we've manually inserted
            # a CLS token at the beginning of the input
            cls_index = 0

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_ex.sequence_ids
        context_sequence_id = 1 if pad_on_right else 0
        context_mask = [s_id == context_sequence_id for s_id in sequence_ids]
        context_mask[0] = True  # CLS token is a valid answer
        tokenized_examples["context_mask"].append(context_mask)

        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != context_sequence_id:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != context_sequence_id:
            token_end_index -= 1
        token_end_index += 1  # exclusive

        token_to_word_map = []
        for i in range(len(input_ids)):
            token_to_word_map.append(tokenized_ex.token_to_word(i))
        tokenized_examples["token_to_word_map"].append(token_to_word_map)

        # Answers are arguments
        answer = examples["answer"][sample_index]

        # If no answers are given, set the cls_index as answer.
        if not answer:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["all_start_positions"].append([])
            tokenized_examples["end_positions"].append(cls_index)
            tokenized_examples["all_end_positions"].append([])
        else:
            if isinstance(answer, list):
                all_answer_start_toks = []
                all_answer_end_toks = []
                for a in answer:
                    answer_start_tok = tokenized_ex.word_to_tokens(
                        a["start_tok_idx"], sequence_index=context_sequence_id
                    )
                    answer_end_tok = tokenized_ex.word_to_tokens(
                        a["end_tok_idx"], sequence_index=context_sequence_id
                    )
                    if (
                        answer_start_tok
                        and answer_end_tok
                        and (
                            token_start_index <= answer_start_tok[0]
                            and token_end_index >= answer_end_tok[1]
                        )
                    ):
                        all_answer_start_toks.append(answer_start_tok[0])
                        all_answer_end_toks.append(answer_end_tok[1])

                if answer:
                    answer_start_tok = tokenized_ex.word_to_tokens(
                        answer[0]["start_tok_idx"], sequence_index=context_sequence_id
                    )
                    answer_end_tok = tokenized_ex.word_to_tokens(
                        answer[0]["end_tok_idx"], sequence_index=context_sequence_id
                    )
                else:
                    answer_start_tok = None
                    answer_end_tok = None
                tokenized_examples["all_start_positions"].append(all_answer_start_toks)
                tokenized_examples["all_end_positions"].append(all_answer_end_toks)
            else:
                answer_start_tok = tokenized_ex.word_to_tokens(
                    answer["start_tok_idx"], sequence_index=context_sequence_id
                )
                answer_end_tok = tokenized_ex.word_to_tokens(
                    answer["end_tok_idx"], sequence_index=context_sequence_id
                )
                tokenized_examples["all_start_positions"].append([answer_start_tok])
                tokenized_examples["all_end_positions"].append([answer_end_tok])
            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if (
                not answer_start_tok
                or not answer_end_tok
                or not (
                    token_start_index <= answer_start_tok[0]
                    and token_end_index >= answer_end_tok[1]
                )
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                tokenized_examples["start_positions"].append(answer_start_tok[0])
                tokenized_examples["end_positions"].append(answer_end_tok[1])

        # If we want to preserve any of the original columns, we need to
        # create new lists, since the number of output examples may differ
        # from the number of input examples for long contexts
        for c in columns_to_keep:
            if c not in tokenized_examples:
                tokenized_examples[c] = []
            tokenized_examples[c].append(examples[c][sample_index])

    return tokenized_examples
