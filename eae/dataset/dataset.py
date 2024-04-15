import json
import logging
import spacy
import spacy_alignments as tokenizations
import sys

from collections import defaultdict
from functools import partial
from itertools import groupby
from sacremoses import MosesDetokenizer, MosesTokenizer
from typing import Any, Dict, Tuple, Union

from eae.dataset.common import (
    ACE_DATA_FILES,
    ACE_ONTOLOGY_FILE,
    ACE_ROLE_MAPPINGS,
    ACE_IGNORED_ROLES,
    DatasetChoice,
    ERE_NUM_CONTEXT_SENTENCES,
    ERE_SUBTYPE_MAPPINGS,
    EVENT_SEP,
    FAMUS_DATA_FILES,
    FAMUS_ONTOLOGY_FILE,
    FRAMENET_DATA_FILES,
    FRAMENET_ONTOLOGY_FILE,
    LIGHT_ERE_FILES,
    LIGHT_ERE_ONTOLOGY_FILE,
    RAMS_DATA_FILES,
    RAMS_ONTOLOGY_FILE,
    RICH_ERE_FILES,
    RICH_ERE_ONTOLOGY_FILE,
    ROLE_SEP,
    TaskChoice,
    _validate_split,
    _validate_mode,
    WIKIEVENTS_COREF_FILES,
    WIKIEVENTS_DATA_FILES,
    WIKIEVENTS_IGNORED_TYPES,
    WIKIEVENTS_ONTOLOGY_FILE,
    WIKIEVENTS_MAX_CONTEXT_LEN,
    WIKIEVENTS_MERGED_TYPES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

nlp = spacy.load("en_core_web_sm")
config = {"punct_chars": None}
nlp.add_pipe("sentencizer", config=config)
tokenizer = nlp.tokenizer


MT = MosesTokenizer("en")
MD = MosesDetokenizer("en")


def construct_wikievents_context(
    ex: Dict[str, Any], trigger: Dict[str, Union[str, int]]
) -> str:
    """WikiEvents pre-processing code for constructing the context around a trigger.

    This is taken directly from the publicly available repo:
    https://github.com/raspberryice/gen-arg/blob/main/src/genie/KAIROS_data_module.py
    """
    offset = 0
    # trigger span does not include last index
    context_words = ex["tokens"]
    center_sent = trigger["sent_idx"]
    if len(context_words) > WIKIEVENTS_MAX_CONTEXT_LEN:
        cur_len = len(ex["sentences"][center_sent][0])
        context_words = [tup[0] for tup in ex["sentences"][center_sent][0]]
        if cur_len > WIKIEVENTS_MAX_CONTEXT_LEN:
            # one sentence is very long
            trigger_start = trigger["start_tok_idx"]
            start_idx = max(0, trigger_start - WIKIEVENTS_MAX_CONTEXT_LEN // 2)
            end_idx = min(
                len(context_words), trigger_start + WIKIEVENTS_MAX_CONTEXT_LEN // 2
            )
            if end_idx < trigger_start:
                end_idx = trigger_start + WIKIEVENTS_MAX_CONTEXT_LEN // 2
            context_words = ex["tokens"][start_idx:end_idx]
            offset = start_idx

        else:
            # take a sliding window
            left = center_sent - 1
            right = center_sent + 1

            total_sents = len(ex["sentences"])
            prev_len = 0
            while cur_len > prev_len:
                prev_len = cur_len
                # try expanding the sliding window
                if left >= 0:
                    left_sent_tokens = [tup[0] for tup in ex["sentences"][left][0]]
                    if cur_len + len(left_sent_tokens) <= WIKIEVENTS_MAX_CONTEXT_LEN:
                        context_words = left_sent_tokens + context_words
                        left -= 1
                        cur_len += len(left_sent_tokens)

                if right < total_sents:
                    right_sent_tokens = [tup[0] for tup in ex["sentences"][right][0]]
                    if cur_len + len(right_sent_tokens) <= WIKIEVENTS_MAX_CONTEXT_LEN:
                        context_words = context_words + right_sent_tokens
                        right += 1
                        cur_len += len(right_sent_tokens)
            # update trigger offset
            offset = sum([len(ex["sentences"][idx][0]) for idx in range(left + 1)])

    assert len(context_words) <= WIKIEVENTS_MAX_CONTEXT_LEN
    trigger_start = trigger["start_tok_idx"] - offset
    trigger_end = trigger["end_tok_idx"] - offset - 1  # make end inclusive
    return (offset, context_words, trigger_start, trigger_end)


def gen_wikievents(split: str, task: str):
    task = TaskChoice[task.upper()]
    is_train = split == "train"
    _validate_split(split)
    # read in ontology information
    with open(WIKIEVENTS_ONTOLOGY_FILE) as f:
        onto = json.load(f)

    total_events_without_args = 0
    total_dropped_args = 0
    total_read_args = 0
    total_cleaned_args = 0
    total_dropped_events = 0
    total_read_events = 0
    events_with_converted_type = 0
    with open(WIKIEVENTS_DATA_FILES[split]) as f:
        for line in f:
            l = json.loads(line)
            entity_mentions_by_id = {m["id"]: m for m in l["entity_mentions"]}
            sorted_event_mentions = sorted(
                l["event_mentions"], key=lambda x: x["event_type"]
            )
            for event_type, events in groupby(
                sorted_event_mentions, key=lambda x: x["event_type"]
            ):
                if event_type in WIKIEVENTS_IGNORED_TYPES:
                    events = [e for e in events]
                    total_dropped_events += len(events)
                    total_read_events += len(events)
                    continue
                elif event_type in WIKIEVENTS_MERGED_TYPES:
                    events_with_converted_type += len([e for e in events])
                    event_type = WIKIEVENTS_MERGED_TYPES[event_type]
                formatted_events = []
                for e in events:
                    filtered_args = []
                    for a in e["arguments"]:
                        a["start_tok_idx"] = entity_mentions_by_id[a["entity_id"]][
                            "start"
                        ]
                        a["end_tok_idx"] = entity_mentions_by_id[a["entity_id"]]["end"]
                        if a["role"] in onto[event_type]["roles"]:
                            a["role"] = a["role"]
                            filtered_args.append(a)
                        else:
                            total_dropped_args += 1
                        total_read_args += 1
                    e["trigger"]["start_tok_idx"] = e["trigger"].pop("start")
                    e["trigger"]["end_tok_idx"] = e["trigger"].pop("end")
                    e["arguments"] = filtered_args
                    if len(filtered_args) > 0 or split != "train":
                        # The original WikiEvents preprocessing code drops train instances that lack arguments
                        # https://github.com/raspberryice/gen-arg/blob/main/src/genie/KAIROS_data_module.py#L195
                        formatted_events.append(e)
                    else:
                        total_dropped_events += 1
                        total_events_without_args += 1

                role_types = []
                role_defs = []
                role_questions = []
                for k, v in sorted(onto[event_type]["roles"].items()):
                    role_types.append(k)
                    role_defs.append(v["definition"])
                    role_questions.append(v["question"])

                ret = {
                    "instance_id": f"{event_type}__{l['doc_id']}",
                    "doc_id": l["doc_id"],
                    "event_type": event_type,
                    "event_type_template": onto[event_type]["orig_template"],
                    "event_type_template_roles": onto[event_type]["template roles"],
                    "event_type_definition": onto[event_type]["definition"],
                    "split": split,
                    "dataset": DatasetChoice.WIKIEVENTS.name,
                }
                # if doing event argument extraction (EAE), each example
                # consists of a single event (rather than potentially multiple)
                # and we highlight the relevant trigger in the document text.
                if task == TaskChoice.EAE or task == TaskChoice.INFILLING:
                    for i, e in enumerate(formatted_events):
                        ex = ret.copy()
                        ex["instance_id"] = f"{ex['instance_id']}-{i}"
                        ex["role_types"] = role_types
                        ex["role_definitions"] = role_defs
                        (
                            offset,
                            context,
                            trigger_start_tok,
                            trigger_end_tok,
                        ) = construct_wikievents_context(l, e["trigger"])
                        e["trigger"]["start_tok_idx"] = trigger_start_tok
                        e["trigger"]["end_tok_idx"] = trigger_end_tok
                        ex["event"] = e
                        tokens = (
                            context[:trigger_start_tok]
                            + [EVENT_SEP]
                            + context[trigger_start_tok : trigger_end_tok + 1]
                            + [EVENT_SEP]
                            + context[trigger_end_tok + 1 :]
                        )
                        new_args = []
                        for a in e["arguments"]:
                            arg_start_tok, arg_end_tok = compute_arg_token_offset(
                                a["start_tok_idx"] - offset,
                                a["end_tok_idx"] - offset,
                                trigger_start_tok,
                                trigger_end_tok,
                                end_idx_is_inclusive=True,
                            )
                            a["start_tok_idx"] = arg_start_tok
                            a["end_tok_idx"] = arg_end_tok - 1  # make inclusive
                            if a["start_tok_idx"] < 0 or a["end_tok_idx"] >= len(
                                tokens
                            ):
                                total_dropped_args += 1
                                continue
                            else:
                                new_args.append(a)
                        e["arguments"] = new_args
                        e["trigger"].pop("sent_idx")
                        ex["tokens"] = tokens
                        ex["text"] = MD.detokenize(ex["tokens"])
                        if task == TaskChoice.INFILLING:
                            ex["template"] = " ".join(
                                [
                                    ROLE_SEP + tok.text + ROLE_SEP
                                    if tok.text[0].isupper()
                                    else tok.text
                                    for tok in tokenizer(
                                        onto[event_type]["new_template"]
                                    )
                                ]
                            )
                            # ex["template"] = onto[event_type]["new_template"]
                        yield ex
                elif task == TaskChoice.QA:
                    for i, e in enumerate(formatted_events):
                        (
                            offset,
                            context,
                            trigger_start_tok,
                            trigger_end_tok,
                        ) = construct_wikievents_context(l, e["trigger"])
                        e["trigger"]["start_tok_idx"] = trigger_start_tok
                        e["trigger"]["end_tok_idx"] = trigger_end_tok
                        e["trigger"].pop("sent_idx")
                        tokens = (
                            context[:trigger_start_tok]
                            + [EVENT_SEP]
                            + context[trigger_start_tok:trigger_end_tok]
                            + [EVENT_SEP]
                            + context[trigger_end_tok:]
                        )
                        ret["tokens"] = tokens
                        args_by_role = defaultdict(list)
                        for a in e["arguments"]:
                            arg_start_tok, arg_end_tok = compute_arg_token_offset(
                                a["start_tok_idx"] - offset,
                                a["end_tok_idx"] - offset,
                                trigger_start_tok,
                                trigger_end_tok,
                                end_idx_is_inclusive=True,
                            )
                            a["start_tok_idx"] = arg_start_tok
                            a["end_tok_idx"] = arg_end_tok - 1  # make inclusive
                            if a["start_tok_idx"] < 0 or a["end_tok_idx"] >= len(
                                tokens
                            ):
                                total_dropped_args += 1
                                continue
                            else:
                                args_by_role[a["role"]].append(a)
                        ret["event"] = e
                        for role_type, role_def, role_q in zip(
                            role_types, role_defs, role_questions
                        ):
                            args = args_by_role[role_type]
                            if args:
                                if is_train:
                                    for arg in args:
                                        qa_ex = ret.copy()
                                        qa_ex["instance_id"] = "__".join(
                                            [qa_ex["instance_id"], role_type]
                                        )
                                        qa_ex["role_type"] = role_type
                                        qa_ex["role_definition"] = role_def
                                        qa_ex["role_question"] = (
                                            role_q[:-1]
                                            + " in "
                                            + e["trigger"]["text"]
                                            + "?"
                                        )
                                        qa_ex["role_question_tok"] = MT.tokenize(
                                            qa_ex["role_question"]
                                        )
                                        qa_ex["answer"] = arg
                                        yield qa_ex
                                else:
                                    qa_ex = ret.copy()
                                    qa_ex["instance_id"] = "__".join(
                                        [qa_ex["instance_id"], role_type]
                                    )
                                    qa_ex["role_type"] = role_type
                                    qa_ex["role_definition"] = role_def
                                    qa_ex["role_question"] = (
                                        role_q[:-1]
                                        + " in "
                                        + e["trigger"]["text"]
                                        + "?"
                                    )
                                    qa_ex["role_question_tok"] = MT.tokenize(
                                        qa_ex["role_question"]
                                    )
                                    qa_ex["answer"] = args
                                    yield qa_ex
                            else:
                                qa_ex = ret.copy()
                                qa_ex["instance_id"] = "__".join(
                                    [qa_ex["instance_id"], role_type]
                                )
                                qa_ex["role_type"] = role_type
                                qa_ex["role_definition"] = role_def
                                qa_ex["role_question"] = (
                                    role_q[:-1] + " in " + e["trigger"]["text"] + "?"
                                )
                                qa_ex["role_question_tok"] = MT.tokenize(
                                    qa_ex["role_question"]
                                )
                                qa_ex["answer"] = None
                                yield qa_ex
                elif task == TaskChoice.OTE:
                    ret["events"] = formatted_events
                    ret["tokens"] = l["tokens"]
                    ret["text"] = l["text"]
                    yield ret

    logger.warning(f"Read {total_read_events} events")
    logger.warning(f"Dropped {total_events_without_args} events without arguments")
    logger.warning(f"Dropped {total_dropped_events} total events")
    logger.warning(f"Converted types of {events_with_converted_type} events")
    logger.warning(f"Read {total_read_args} arguments")
    logger.warning(f"Cleaned {total_cleaned_args} arguments")
    logger.warning(f"Dropped {total_dropped_args} arguments")


def gen_ace(split: str, task: str):
    is_train = split == "train"
    task = TaskChoice[task.upper()]
    _validate_split(split)
    with open(ACE_ONTOLOGY_FILE) as f:
        onto = json.load(f)

    with open(ACE_DATA_FILES[split]) as f:
        d = json.load(f)

    for i, ex in enumerate(d):
        mention_to_entity = {}
        for m in ex["golden-entity-mentions"]:
            mention_to_entity[(m["start"], m["end"])] = m["entity_id"]

        for j, m in enumerate(ex["golden-event-mentions"]):
            # ACE has exclusive end indices -> convert to inclusive
            # +1 to start index for EVENT_SEP
            trigger = {
                "start_tok_idx": m["trigger"]["start"] + 1,
                "end_tok_idx": m["trigger"]["end"],
                "text": m["trigger"]["text"],
            }
            event = {
                "id": f"{split}-{i}-{j}",
                "trigger": trigger,
                "event_type": m["event_type"],
                "arguments": [],
            }
            for arg in m["arguments"]:
                arg_start_tok, arg_end_tok = compute_arg_token_offset(
                    arg["start"],
                    arg["end"],
                    m["trigger"]["start"],
                    m["trigger"]["end"],
                    end_idx_is_inclusive=False,
                )
                arg_end_tok -= 1  # make end inclusive
                if (
                    event["event_type"] in ACE_ROLE_MAPPINGS
                    and arg["role"] in ACE_ROLE_MAPPINGS[event["event_type"]]
                ):
                    role = ACE_ROLE_MAPPINGS[event["event_type"]][arg["role"]]
                elif (
                    event["event_type"] in ACE_IGNORED_ROLES
                    and arg["role"] in ACE_IGNORED_ROLES[event["event_type"]]
                ) or (arg["role"].lower().startswith("time")):
                    # skipping time roles for now
                    continue
                else:
                    role = arg["role"]
                event["arguments"].append(
                    {
                        "entity_id": mention_to_entity[(arg["start"], arg["end"])],
                        "start_tok_idx": arg_start_tok,
                        "end_tok_idx": arg_end_tok,
                        "text": arg["text"],
                        "role": role.lower(),
                    }
                )

            role_types = []
            role_definitions = []
            role_questions = []
            for r, r_def in onto[m["event_type"]]["roles"].items():
                role_types.append(r.lower())
                role_definitions.append(r_def["definition"])
                role_questions.append(r_def["question"])

            tokens = (
                ex["words"][: m["trigger"]["start"]]
                + [EVENT_SEP]
                + ex["words"][m["trigger"]["start"] : m["trigger"]["end"]]
                + [EVENT_SEP]
                + ex["words"][m["trigger"]["end"] :]
            )

            out_ex = {
                "instance_id": f"{m['event_type']}__{split}-{i}-{j}",
                "doc_id": f"{split}-{i}",
                "text": MD.detokenize(tokens),  # ex["sentence"]
                "tokens": tokens,
                "event": event,
                "event_type": m["event_type"],
                "event_type_definition": onto[m["event_type"]]["definition"],
                "split": split,
                "dataset": DatasetChoice.ACE.name,
            }

            if task == TaskChoice.QA:
                args_by_role = defaultdict(list)
                for arg in event["arguments"]:
                    args_by_role[arg["role"]].append(arg)
                for role_type, role_def, role_q in zip(
                    role_types, role_definitions, role_questions
                ):
                    args = args_by_role[role_type.lower()]
                    if args:
                        if is_train:
                            # in training, each argument maps to its own example
                            for arg in args:
                                qa_ex = out_ex.copy()
                                qa_ex["instance_id"] = "__".join(
                                    [qa_ex["instance_id"], role_type]
                                )
                                qa_ex["role_type"] = role_type
                                qa_ex["role_definition"] = role_def
                                qa_ex["role_question"] = (
                                    role_q[:-1] + " in " + m["trigger"]["text"] + "?"
                                )
                                qa_ex["role_question_tok"] = MT.tokenize(
                                    qa_ex["role_question"]
                                )
                                qa_ex["answer"] = arg
                                yield qa_ex
                        else:
                            # at inference time, all arguments get mapped to
                            # the same example (we decode top-k answers and
                            # subset based on a threshold)
                            qa_ex = out_ex.copy()
                            qa_ex["instance_id"] = "__".join(
                                [qa_ex["instance_id"], role_type]
                            )
                            qa_ex["role_type"] = role_type
                            qa_ex["role_definition"] = role_def
                            qa_ex["role_question"] = (
                                role_q[:-1] + " in " + m["trigger"]["text"] + "?"
                            )
                            qa_ex["role_question_tok"] = MT.tokenize(
                                qa_ex["role_question"]
                            )
                            qa_ex["answer"] = args
                            yield qa_ex

                    else:
                        qa_ex = out_ex.copy()
                        qa_ex["instance_id"] = "__".join(
                            [qa_ex["instance_id"], role_type]
                        )
                        qa_ex["role_type"] = role_type
                        qa_ex["role_definition"] = role_def
                        qa_ex["role_question"] = (
                            role_q[:-1] + " in " + m["trigger"]["text"] + "?"
                        )
                        qa_ex["role_question_tok"] = MT.tokenize(qa_ex["role_question"])
                        qa_ex["answer"] = None
                        yield qa_ex
            else:  # INFILLING
                out_ex["role_types"] = role_types
                out_ex["role_definitions"] = role_definitions
                if task == TaskChoice.INFILLING:
                    out_ex["template"] = " ".join(
                        [
                            ROLE_SEP + tok.text + ROLE_SEP
                            if tok.text[0].isupper()
                            else tok.text
                            for tok in tokenizer(onto[m["event_type"]]["template"])
                        ]
                    )
                    # out_ex["template"] = onto[m["event_type"]]["template"]
                yield out_ex


def gen_ere(split: str, mode: str, task: str):
    is_train = split == "train"
    task = TaskChoice[task.upper()]
    if task == TaskChoice.OTE:
        raise NotImplementedError(f"OTE not yet supported for ERE")

    onto_file = (
        LIGHT_ERE_ONTOLOGY_FILE if mode == "light_ere" else RICH_ERE_ONTOLOGY_FILE
    )
    data_files = LIGHT_ERE_FILES if mode == "light_ere" else RICH_ERE_FILES
    _validate_split(split)
    _validate_mode(mode)
    with open(onto_file) as f:
        onto = json.load(f)

    n_dropped_args = 0
    n_total_args = 0
    with open(data_files[split]) as f:
        for line in f:
            l = json.loads(line)
            doc_id = l["doc_id"]
            text = l["text"]
            tokens = l["tokens"]
            char2tok, tok2char = tokenizations.get_alignments(list(text), tokens)
            sent_char_offsets = []
            sent_tok_offsets = []
            for s in nlp(
                text.lower()
            ).sents:  # lowercasing text makes for better sentence splits
                sent_char_offsets.append((s.start_char, s.end_char))
                start_char = s.start_char
                end_char = s.end_char - 1

                # sometimes we can have whitespace at the
                # beginning or end of a sentence; strip it
                while not char2tok[start_char]:
                    start_char += 1
                while not char2tok[end_char]:
                    end_char -= 1

                sent_tok_offsets.append(
                    (char2tok[start_char][0], char2tok[end_char][0])
                )

            for e, in_event in enumerate(l["events"]):
                event_type = in_event["trigger"]["type"]
                event_subtype = in_event["trigger"]["subtype"]
                if event_subtype in ERE_SUBTYPE_MAPPINGS:
                    event_subtype = ERE_SUBTYPE_MAPPINGS[event_subtype]
                else:
                    event_subtype = event_subtype.capitalize()
                event_type = event_type.capitalize() + ":" + event_subtype
                trigger = {
                    "text": in_event["trigger"]["text"],
                    "start_tok_idx": in_event["trigger"]["start_tok_idx"],
                    "end_tok_idx": in_event["trigger"]["end_tok_idx"],
                }
                # find trigger sentence
                trigger_sent_idx = -1
                for i, (start, end) in enumerate(sent_tok_offsets):
                    if (
                        trigger["start_tok_idx"] >= start
                        and trigger["end_tok_idx"] <= end
                    ):
                        trigger_sent_idx = i
                        break
                if trigger_sent_idx == -1:
                    raise ValueError("Trigger not found in any sentence")

                context_start_sent = max(
                    0, trigger_sent_idx - ERE_NUM_CONTEXT_SENTENCES
                )
                context_start_char = sent_char_offsets[context_start_sent][0]
                context_start_tok = sent_tok_offsets[context_start_sent][0]
                context_end_sent = min(
                    len(sent_tok_offsets) - 1,
                    trigger_sent_idx + ERE_NUM_CONTEXT_SENTENCES,
                )
                context_end_char = sent_char_offsets[context_end_sent][1]
                context_end_tok = sent_tok_offsets[context_end_sent][1]
                context_tokens = tokens[context_start_tok:context_end_tok]
                context_text = text[context_start_char:context_end_char]
                trigger["start_tok_idx"] = trigger["start_tok_idx"] - context_start_tok
                trigger_start_tok_idx = trigger["start_tok_idx"]
                trigger["end_tok_idx"] = trigger["end_tok_idx"] - context_start_tok
                trigger_end_tok_idx = trigger["end_tok_idx"]
                context_tokens = (
                    context_tokens[: trigger["start_tok_idx"]]
                    + [EVENT_SEP]
                    + context_tokens[
                        trigger["start_tok_idx"] : trigger["end_tok_idx"] + 1
                    ]
                    + [EVENT_SEP]
                    + context_tokens[trigger["end_tok_idx"] + 1 :]
                )
                # +1 here for event_sep token
                trigger["start_tok_idx"] = trigger_start_tok_idx + 1
                trigger["end_tok_idx"] = trigger_end_tok_idx + 1
                context_text = MD.detokenize(context_tokens)

                out_event = {
                    "id": f"{event_type}__{doc_id}-{e}",
                    "event_type": event_type,
                    "trigger": trigger,
                    "arguments": [],
                }

                role_types = []
                role_definitions = []
                role_questions = []
                for r, r_def in onto[event_type]["roles"].items():
                    role_types.append(r.lower())
                    role_definitions.append(r_def["definition"])
                    role_questions.append(r_def["question"])

                for arg in in_event["arguments"]:
                    n_total_args += 1
                    mentions_in_context = [
                        m
                        for m in arg["mentions"]
                        if m["start_tok_idx"] >= context_start_tok
                        and m["end_tok_idx"] <= context_end_tok
                    ]
                    if not mentions_in_context:
                        n_dropped_args += 1
                        continue

                    # select the most informative mention in context
                    arg_mention = sorted(
                        arg["mentions"], key=lambda x: x["mention_type"]
                    )[0]
                    assert (
                        arg["type"].capitalize() in onto[event_type]["roles"]
                    ), f"Role {arg['type']} not found in ontology for event type {event_type}"
                    arg_start_tok, arg_end_tok = compute_arg_token_offset(
                        arg_mention["start_tok_idx"] - context_start_tok,
                        arg_mention["end_tok_idx"] - context_start_tok,
                        trigger_start_tok_idx,
                        trigger_end_tok_idx,
                        end_idx_is_inclusive=True,
                    )
                    out_arg = {
                        "entity_id": arg["entity_id"],
                        "role": arg["type"],
                        "start_tok_idx": arg_start_tok,
                        "end_tok_idx": arg_end_tok,
                        "text": arg_mention["text"],
                    }
                    out_event["arguments"].append(out_arg)

                ex = {
                    "instance_id": out_event["id"],
                    "doc_id": doc_id,
                    "text": context_text,
                    "tokens": context_tokens,
                    "event": out_event,
                    "event_type": event_type,
                    "event_type_definition": onto[event_type]["definition"],
                    "split": split,
                    "dataset": mode,
                }
                if task == TaskChoice.EAE:
                    ex["role_types"] = role_types
                    ex["role_definitions"] = role_definitions
                    yield ex
                elif task == TaskChoice.QA:
                    for (
                        role_type,
                        role_def,
                        role_q,
                    ) in zip(role_types, role_definitions, role_questions):
                        args = [
                            a for a in out_event["arguments"] if a["role"] == role_type
                        ]
                        if args:
                            if is_train:
                                for arg in args:
                                    qa_ex = ex.copy()
                                    qa_ex["instance_id"] = "__".join(
                                        [qa_ex["instance_id"], role_type]
                                    )
                                    qa_ex["role_type"] = role_type
                                    qa_ex["role_definition"] = role_def
                                    qa_ex["role_question"] = (
                                        role_q[:-1] + " in " + trigger["text"] + "?"
                                    )
                                    qa_ex["role_question_tok"] = MT.tokenize(
                                        qa_ex["role_question"]
                                    )
                                    qa_ex["answer"] = arg
                                    yield qa_ex
                            else:
                                qa_ex = ex.copy()
                                qa_ex["instance_id"] = "__".join(
                                    [qa_ex["instance_id"], role_type]
                                )
                                qa_ex["role_type"] = role_type
                                qa_ex["role_definition"] = role_def
                                qa_ex["role_question"] = (
                                    role_q[:-1] + " in " + trigger["text"] + "?"
                                )
                                qa_ex["role_question_tok"] = MT.tokenize(
                                    qa_ex["role_question"]
                                )
                                qa_ex["answer"] = args
                                yield qa_ex
                        else:
                            qa_ex = ex.copy()
                            qa_ex["instance_id"] = "__".join(
                                [qa_ex["instance_id"], role_type]
                            )
                            qa_ex["role_type"] = role_type
                            qa_ex["role_definition"] = role_def
                            qa_ex["role_question"] = (
                                role_q[:-1] + " in " + trigger["text"] + "?"
                            )
                            qa_ex["role_question_tok"] = MT.tokenize(
                                qa_ex["role_question"]
                            )
                            qa_ex["answer"] = None
                            yield qa_ex
                else:
                    ex["role_types"] = role_types
                    ex["role_definitions"] = role_definitions
                    if task == TaskChoice.INFILLING:
                        ex["template"] = " ".join(
                            [
                                ROLE_SEP + tok.text + ROLE_SEP
                                if tok.text[0].isupper()
                                else tok.text
                                for tok in tokenizer(onto[ex["event_type"]]["template"])
                            ]
                        )
                        # ex["template"] = onto[ex["event_type"]]["template"]
                    yield ex

    logger.warning(f"Dropped {n_dropped_args}/{n_total_args} arguments")


def gen_rams(split: str, task: str):
    is_train = split == "train"
    task = TaskChoice[task.upper()]
    _validate_split(split)
    with open(RAMS_ONTOLOGY_FILE) as f:
        onto = json.load(f)

    with open(RAMS_DATA_FILES[split]) as f:
        for line in f:
            l = json.loads(line)
            doc_id = l["doc_key"]
            doc_tokens = [tok for s in l["sentences"] for tok in s]
            assert len(l["evt_triggers"]) == 1
            event = l["evt_triggers"][0]
            event_start_tok = event[0]
            event_end_tok = event[1] + 1
            event_type = event[2][0][0]
            trigger = {
                "start_tok_idx": event_start_tok + 1,
                "end_tok_idx": event_end_tok,  # make end token inclusive
                "text": " ".join(doc_tokens[event_start_tok:event_end_tok]),
            }
            event = {
                "id": doc_id,
                "event_type": event_type,
                "trigger": trigger,
                "arguments": [],
            }
            tokens = (
                doc_tokens[:event_start_tok]
                + [EVENT_SEP]
                + doc_tokens[event_start_tok:event_end_tok]
                + [EVENT_SEP]
                + doc_tokens[event_end_tok:]
            )
            args_by_role = defaultdict(list)
            for i, arg in enumerate(l["ent_spans"]):
                arg_start_tok = arg[0]
                arg_end_tok = arg[1] + 1
                role = arg[2][0][0][11:]  # strip AIDA specifier from role name
                new_arg_start_tok, new_arg_end_tok = compute_arg_token_offset(
                    arg_start_tok,
                    arg_end_tok,
                    event_start_tok,
                    event_end_tok,
                    end_idx_is_inclusive=False,
                )
                if task == TaskChoice.OTE:
                    text = " ".join(doc_tokens[arg_start_tok:arg_end_tok])
                else:
                    text = " ".join(tokens[new_arg_start_tok:new_arg_end_tok])
                arg = {
                    "entity_id": f"{event_type}-{trigger['text']}-{i}",
                    "start_tok_idx": new_arg_start_tok,
                    "end_tok_idx": new_arg_end_tok - 1,  # make end token inclusive
                    "role": role,
                    "text": text,
                }
                event["arguments"].append(arg)
                args_by_role[role].append(arg)
            role_types = []
            role_defs = []
            role_questions = []
            for k, v in sorted(onto[event_type]["roles"].items()):
                role_types.append(k)
                role_defs.append(v["definition"])
                role_questions.append(v["question"])

            ret = {
                "instance_id": f"{event_type}__{doc_id}",
                "doc_id": doc_id,
                "tokens": doc_tokens if task == TaskChoice.OTE else tokens,
                "event_type": event_type,
                "event_type_definition": onto[event_type]["definition"],
                "split": split,
                "dataset": DatasetChoice.RAMS.name,
            }
            if task == TaskChoice.EAE or task == TaskChoice.INFILLING:
                ex = ret.copy()
                ex["event"] = event
                ex["text"] = MD.detokenize(tokens)
                ex["role_types"] = role_types
                ex["role_definitions"] = role_defs
                if task == TaskChoice.INFILLING:
                    ex["template"] = " ".join(
                        [
                            ROLE_SEP + tok.text + ROLE_SEP
                            if tok.text[0].isupper()
                            else tok.text
                            for tok in tokenizer(onto[event_type]["new_template"])
                        ]
                    )
                    # ex["template"] = onto[event_type]["new_template"]
                yield ex
            elif task == TaskChoice.QA:
                ex = ret.copy()
                ex["event"] = event
                ex["text"] = MD.detokenize(tokens)
                for role, role_def, role_q in zip(
                    role_types, role_defs, role_questions
                ):
                    args = args_by_role[role]
                    if args:
                        if is_train:
                            for arg in args:
                                qa_ex = ex.copy()
                                qa_ex["instance_id"] = "__".join(
                                    [qa_ex["instance_id"], role]
                                )
                                qa_ex["role_type"] = role
                                qa_ex["role_definition"] = role_def
                                qa_ex["role_question"] = (
                                    role_q[:-1] + " in " + trigger["text"] + "?"
                                )
                                qa_ex["role_question_tok"] = MT.tokenize(
                                    qa_ex["role_question"]
                                )
                                qa_ex["answer"] = arg
                                yield qa_ex
                        else:
                            qa_ex = ex.copy()
                            qa_ex["instance_id"] = "__".join(
                                [qa_ex["instance_id"], role]
                            )
                            qa_ex["role_type"] = role
                            qa_ex["role_definition"] = role_def
                            qa_ex["role_question"] = (
                                role_q[:-1] + " in " + trigger["text"] + "?"
                            )
                            qa_ex["role_question_tok"] = MT.tokenize(
                                qa_ex["role_question"]
                            )
                            qa_ex["answer"] = args
                            yield qa_ex
                    else:
                        qa_ex = ex.copy()
                        qa_ex["instance_id"] = "__".join([qa_ex["instance_id"], role])
                        qa_ex["role_type"] = role
                        qa_ex["role_definition"] = role_def
                        qa_ex["role_question"] = (
                            role_q[:-1] + " in " + trigger["text"] + "?"
                        )
                        qa_ex["role_question_tok"] = MT.tokenize(qa_ex["role_question"])
                        qa_ex["answer"] = None
                        yield qa_ex
            else:
                ret["events"] = [event]
                # AFAIK, RAMS only has tokenized text, which is why we have to do this
                ret["text"] = MD.detokenize(doc_tokens)
                ex["role_types"] = role_types
                ex["role_definitions"] = role_defs
                yield ret


def gen_framenet(
    split: str,
    task: str,
    core_roles_only: bool = False,
    use_famus_ontology: bool = False,
):
    _validate_split(split)
    is_train = split == "train"
    task = TaskChoice[task.upper()]
    if use_famus_ontology:
        ontology_file = FAMUS_ONTOLOGY_FILE
    else:
        ontology_file = FRAMENET_ONTOLOGY_FILE

    with open(ontology_file) as f:
        onto = json.load(f)

    with open(FRAMENET_DATA_FILES[split]) as f:
        for line in f:
            l = json.loads(line)

            for i, anno in enumerate(l["annotations"]):
                if use_famus_ontology and anno["label"] not in onto:
                    continue
                instance_id = f"{l['meta']['sentence ID']}-{anno['label']}-{i}"
                event = {
                    "id": instance_id,
                    "event_type": anno["label"],
                    "trigger": {
                        "start_tok_idx": anno["span"][0],
                        "end_tok_idx": anno["span"][1],
                        "text": " ".join(
                            l["tokens"][anno["span"][0] : anno["span"][1] + 1]
                        ),
                    },
                    "arguments": [],
                }
                roles_key = (
                    "core roles" if core_roles_only or use_famus_ontology else "roles"
                )
                role_types = []
                role_questions = []
                role_defs = []
                for role, role_info in sorted(onto[anno["label"]][roles_key].items()):
                    role_types.append(role)
                    role_defs.append(role_info["definition"])
                    role_questions.append(role_info["question"])

                for j, arg in enumerate(anno["children"]):
                    if arg["label"] in role_types:
                        event["arguments"].append(
                            {
                                "entity_id": str(j),
                                "text": " ".join(
                                    l["tokens"][arg["span"][0] : arg["span"][1] + 1]
                                ),
                                "start_tok_idx": arg["span"][0],
                                "end_tok_idx": arg["span"][1],
                                "role": arg["label"],
                            }
                        )

                tokens = (
                    l["tokens"][: anno["span"][0]]
                    + [EVENT_SEP]
                    + l["tokens"][anno["span"][0] : anno["span"][1] + 1]
                    + [EVENT_SEP]
                    + l["tokens"][anno["span"][1] + 1 :]
                )
                text = " ".join(tokens)
                ex = {
                    "instance_id": instance_id,
                    "doc_id": l["meta"]["doc"],
                    "tokens": tokens,
                    "text": text,
                    "event": event,
                    "event_type": anno["label"],
                    "event_type_definition": onto[anno["label"]]["definition"],
                    "split": split,
                    "dataset": DatasetChoice.FRAMENET.name,
                }
                if task == TaskChoice.QA:
                    args_by_role = defaultdict(list)
                    for arg in ex["event"]["arguments"]:
                        args_by_role[arg["role"]].append(arg)
                    for role_type, role_def, role_q in zip(
                        role_types, role_defs, role_questions
                    ):
                        args = args_by_role[role_type]
                        role_q = (
                            role_q[:-1] + " in " + ex["event"]["trigger"]["text"] + "?"
                        )
                        if args:
                            if is_train:
                                for arg in args:
                                    qa_ex = ex.copy()
                                    qa_ex["instance_id"] = "__".join(
                                        [qa_ex["instance_id"], role_type]
                                    )
                                    qa_ex["role_type"] = role_type
                                    qa_ex["role_definition"] = role_def
                                    qa_ex["role_question"] = role_q
                                    qa_ex["role_question_tok"] = MT.tokenize(role_q)
                                    qa_ex["answer"] = arg
                                    yield qa_ex
                            else:
                                qa_ex = ex.copy()
                                qa_ex["instance_id"] = "__".join(
                                    [qa_ex["instance_id"], role_type]
                                )
                                qa_ex["role_type"] = role_type
                                qa_ex["role_definition"] = role_def
                                qa_ex["role_question"] = role_q
                                qa_ex["role_question_tok"] = MT.tokenize(role_q)
                                qa_ex["answer"] = args
                                yield qa_ex
                        else:
                            qa_ex = ex.copy()
                            qa_ex["instance_id"] = "__".join(
                                [qa_ex["instance_id"], role_type]
                            )
                            qa_ex["role_type"] = role_type
                            qa_ex["role_definition"] = role_def
                            qa_ex["role_question"] = role_q
                            qa_ex["role_question_tok"] = MT.tokenize(role_q)
                            qa_ex["answer"] = None
                            yield qa_ex

                elif task == TaskChoice.INFILLING:
                    ex["template"] = " ".join(
                        [
                            ROLE_SEP + tok.text + ROLE_SEP
                            if tok.text[0].isupper()
                            else tok.text
                            for tok in tokenizer(onto[anno["label"]]["template"])
                        ]
                    )
                    ex["role_types"] = (role_types,)
                    ex["role_definitions"] = (role_defs,)
                    yield ex
                else:
                    raise NotImplementedError(
                        "Only QA and INFILLING tasks are supported"
                    )


def gen_famus(
    split: str, task: str, do_source: bool = False, use_paraphrases: bool = False
):
    is_train = split == "train"
    task = TaskChoice[task.upper()]
    _validate_split(split)
    with open(FAMUS_ONTOLOGY_FILE) as f:
        onto = json.load(f)

    with open(FAMUS_DATA_FILES[split]) as f:
        for line in f:
            l = json.loads(line)

            event = {
                "id": l["instance_id"],
                "event_type": l["frame"],
                "arguments": [],
            }

            if do_source:
                annotations = l["source_dict"]
                raise NotImplementedError(
                    "FAMuS source documents are not yet supported."
                )
            else:
                annotations = l["report_dict"]
                trigger = annotations["frame-trigger-span"]
                trigger_start_tok = trigger[3]
                trigger_end_tok = trigger[4]
                event["trigger"] = {
                    "text": trigger[0],
                    "start_tok_idx": trigger_start_tok,
                    "end_tok_idx": trigger_end_tok,
                }  # inclusive boundaries

            entity_id = 0  # no actual entity IDs in FAMuS data, so we make them up
            for role, args in annotations["role_annotations"].items():
                if role == "role-spans-indices-in-all-spans":
                    continue
                for arg in args:
                    arg_start_tok, arg_end_tok = compute_arg_token_offset(
                        arg[3], arg[4], trigger_start_tok, trigger_end_tok
                    )
                    event["arguments"].append(
                        {
                            "entity_id": str(entity_id),
                            "text": arg[0],
                            "start_tok_idx": arg_start_tok,
                            "end_tok_idx": arg_end_tok,
                            "role": role,
                        }
                    )
                    entity_id += 1

            role_types = []
            role_defs = []
            role_questions = []
            question_is_paraphrase = []
            for role, role_info in sorted(onto[l["frame"]]["core roles"].items()):
                role_types.append(role)
                role_defs.append(role_info["definition"])
                role_questions.append(role_info["question"])
                question_is_paraphrase.append(False)
                assert role_info[
                    "question"
                ], f"empty question for role {role} in {l['frame']}!"
                if use_paraphrases:
                    for paraphrase in role_info["question_paraphrases"]:
                        role_types.append(role)
                        role_defs.append(role_info["definition"])
                        role_questions.append(paraphrase)
                        question_is_paraphrase.append(True)

            tokens = (
                annotations["doctext-tok"][:trigger_start_tok]
                + [EVENT_SEP]
                + annotations["doctext-tok"][trigger_start_tok : trigger_end_tok + 1]
                + [EVENT_SEP]
                + annotations["doctext-tok"][trigger_end_tok + 1 :]
            )
            event["trigger"]["start_tok_idx"] += 1
            event["trigger"]["end_tok_idx"] += 1
            text = MD.detokenize(tokens)
            ex = {
                "instance_id": f"{l['frame']}__{l['instance_id']}",
                "doc_id": l["instance_id"],  # same as instance_id
                "tokens": tokens,
                "text": text,
                "event": event,
                "event_type": l["frame"],
                "event_type_definition": onto[l["frame"]]["definition"],
                "split": split,
                "dataset": DatasetChoice.FAMUS_REPORTS.name,
            }
            if task == TaskChoice.QA:
                args_by_role = defaultdict(list)
                for arg in ex["event"]["arguments"]:
                    args_by_role[arg["role"]].append(arg)
                for role_type, role_def, role_q, is_paraphrase in zip(
                    role_types, role_defs, role_questions, question_is_paraphrase
                ):
                    args = args_by_role[role_type]
                    role_q = role_q[:-1] + " in " + ex["event"]["trigger"]["text"] + "?"
                    if args:
                        if is_train:
                            for arg in args:
                                qa_ex = ex.copy()
                                qa_ex["instance_id"] = "__".join(
                                    [qa_ex["instance_id"], role_type]
                                )
                                qa_ex["role_type"] = role_type
                                qa_ex["role_definition"] = role_def
                                qa_ex["role_question"] = role_q
                                qa_ex["role_question_tok"] = MT.tokenize(role_q)
                                qa_ex["answer"] = arg
                                yield qa_ex
                        elif not is_paraphrase:
                            qa_ex = ex.copy()
                            qa_ex["instance_id"] = "__".join(
                                [qa_ex["instance_id"], role_type]
                            )
                            qa_ex["role_type"] = role_type
                            qa_ex["role_definition"] = role_def
                            qa_ex["role_question"] = role_q
                            qa_ex["role_question_tok"] = MT.tokenize(role_q)
                            qa_ex["answer"] = args
                            yield qa_ex
                    else:
                        if not is_train and is_paraphrase:
                            # we skip paraphrases if this is not the train
                            # split and there are no arguments
                            continue
                        qa_ex = ex.copy()
                        qa_ex["instance_id"] = "__".join(
                            [qa_ex["instance_id"], role_type]
                        )
                        qa_ex["role_type"] = role_type
                        qa_ex["role_definition"] = role_def
                        qa_ex["role_question"] = role_q
                        qa_ex["role_question_tok"] = MT.tokenize(role_q)
                        qa_ex["answer"] = None
                        yield qa_ex
            else:
                ex["role_types"] = role_types
                ex["role_definitions"] = role_defs
                if task == TaskChoice.INFILLING:
                    ex["template"] = " ".join(
                        [
                            ROLE_SEP + tok.text + ROLE_SEP
                            if tok.text[0].isupper()
                            else tok.text
                            for tok in tokenizer(onto[l["frame"]]["template"])
                        ]
                    )
                    yield ex
                    # only use paraphrases during training
                    if use_paraphrases and is_train:
                        for paraphrase in onto[l["frame"]]["template_paraphrases"]:
                            paraphrase_ex = ex.copy()
                            paraphrase_ex["template"] = " ".join(
                                [
                                    ROLE_SEP + tok.text + ROLE_SEP
                                    if tok.text[0].isupper()
                                    else tok.text
                                    for tok in tokenizer(paraphrase)
                                ]
                            )
                            yield paraphrase_ex
                    # ex["template"] = onto[l["frame"]]["template"]
                else:
                    yield ex


def compute_arg_token_offset(
    arg_start_tok_idx: int,
    arg_end_tok_idx: int,
    trigger_start_tok_idx: int,
    trigger_end_tok_idx: int,
    end_idx_is_inclusive: bool = True,
) -> Tuple[int, int]:
    """Computes updated argument token offsets relative to trigger after surrounding it with EVENT_SEP tokens"""
    updated_arg_start_tok_idx = arg_start_tok_idx
    updated_arg_end_tok_idx = arg_end_tok_idx
    # special case where the argument and the trigger are the same
    if (
        arg_start_tok_idx == trigger_start_tok_idx
        and arg_end_tok_idx == trigger_end_tok_idx
    ):
        return updated_arg_start_tok_idx + 1, updated_arg_end_tok_idx + 1

    if end_idx_is_inclusive:
        if arg_start_tok_idx > trigger_end_tok_idx:
            updated_arg_start_tok_idx += 2
        elif arg_start_tok_idx > trigger_start_tok_idx:
            updated_arg_start_tok_idx += 1

        if arg_end_tok_idx > trigger_end_tok_idx:
            updated_arg_end_tok_idx += 2
        elif arg_end_tok_idx > trigger_start_tok_idx:
            updated_arg_end_tok_idx += 1
    else:
        if arg_start_tok_idx >= trigger_end_tok_idx:
            updated_arg_start_tok_idx += 2
        elif arg_start_tok_idx > trigger_start_tok_idx:
            updated_arg_start_tok_idx += 1

        if arg_end_tok_idx >= trigger_end_tok_idx:
            updated_arg_end_tok_idx += 2
        elif arg_start_tok_idx > trigger_start_tok_idx:
            updated_arg_end_tok_idx += 1

    return updated_arg_start_tok_idx, updated_arg_end_tok_idx


#
# WikiEvents
#

# WikiEvents for event argument extraction (EAE)
# (one target event per example; trigger is given)
WIKIEVENTS_EAE_TRAIN = partial(gen_wikievents, split="train", task="EAE")
WIKIEVENTS_EAE_DEV = partial(gen_wikievents, split="dev", task="EAE")
WIKIEVENTS_EAE_TEST = partial(gen_wikievents, split="test", task="EAE")

# WikiEvents for question answering (QA)
WIKIEVENTS_QA_TRAIN = partial(gen_wikievents, split="train", task="QA")
WIKIEVENTS_QA_DEV = partial(gen_wikievents, split="dev", task="QA")
WIKIEVENTS_QA_TEST = partial(gen_wikievents, split="test", task="QA")

# WikiEvents for infilling
WIKIEVENTS_INFILLING_TRAIN = partial(gen_wikievents, split="train", task="INFILLING")
WIKIEVENTS_INFILLING_DEV = partial(gen_wikievents, split="dev", task="INFILLING")
WIKIEVENTS_INFILLING_TEST = partial(gen_wikievents, split="test", task="INFILLING")


# WikiEvents for event argument extraction (EAE)
# (potentially multiple events per example; triggers are not given)
WIKIEVENTS_OTE_TRAIN = partial(gen_wikievents, split="train", task="OTE")
WIKIEVENTS_OTE_DEV = partial(gen_wikievents, split="dev", task="OTE")
WIKIEVENTS_OTE_TEST = partial(gen_wikievents, split="test", task="OTE")

#
# ACE
#

# ACE for EAE
ACE_EAE_TRAIN = partial(gen_ace, split="dev", task="EAE")
ACE_EAE_DEV = partial(gen_ace, split="dev", task="EAE")
ACE_EAE_TEST = partial(gen_ace, split="test", task="EAE")

# ACE for QA
ACE_QA_TRAIN = partial(gen_ace, split="train", task="QA")
ACE_QA_DEV = partial(gen_ace, split="dev", task="QA")
ACE_QA_TEST = partial(gen_ace, split="test", task="QA")

# ACE for Infilling
ACE_INFILLING_TRAIN = partial(gen_ace, split="train", task="INFILLING")
ACE_INFILLING_DEV = partial(gen_ace, split="dev", task="INFILLING")
ACE_INFILLING_TEST = partial(gen_ace, split="test", task="INFILLING")

#
# RAMS
#

# RAMS for QA
RAMS_QA_TRAIN = partial(gen_rams, split="train", task="QA")
RAMS_QA_DEV = partial(gen_rams, split="dev", task="QA")
RAMS_QA_TEST = partial(gen_rams, split="test", task="QA")

# RAMS for EAE
RAMS_EAE_TRAIN = partial(gen_rams, split="train", task="EAE")
RAMS_EAE_DEV = partial(gen_rams, split="dev", task="EAE")
RAMS_EAE_TEST = partial(gen_rams, split="test", task="EAE")

# RAMS for OTE
RAMS_OTE_TRAIN = partial(gen_rams, split="train", task="OTE")
RAMS_OTE_DEV = partial(gen_rams, split="dev", task="OTE")
RAMS_OTE_TEST = partial(gen_rams, split="test", task="OTE")

# RAMS for Infilling
RAMS_INFILLING_TRAIN = partial(gen_rams, split="train", task="INFILLING")
RAMS_INFILLING_DEV = partial(gen_rams, split="dev", task="INFILLING")
RAMS_INFILLING_TEST = partial(gen_rams, split="test", task="INFILLING")

#
# ERE
#
ERE_LIGHT_EAE_TRAIN = partial(gen_ere, split="train", mode="light_ere", task="EAE")
ERE_LIGHT_EAE_DEV = partial(gen_ere, split="dev", mode="light_ere", task="EAE")
ERE_LIGHT_EAE_TEST = partial(gen_ere, split="test", mode="light_ere", task="EAE")

ERE_LIGHT_QA_TRAIN = partial(gen_ere, split="train", mode="light_ere", task="QA")
ERE_LIGHT_QA_DEV = partial(gen_ere, split="dev", mode="light_ere", task="QA")
ERE_LIGHT_QA_TEST = partial(gen_ere, split="test", mode="light_ere", task="QA")

ERE_LIGHT_INFILLING_TRAIN = partial(
    gen_ere, split="train", mode="light_ere", task="INFILLING"
)
ERE_LIGHT_INFILLING_DEV = partial(
    gen_ere, split="dev", mode="light_ere", task="INFILLING"
)
ERE_LIGHT_INFILLING_TEST = partial(
    gen_ere, split="test", mode="light_ere", task="INFILLING"
)

ERE_RICH_EAE_TRAIN = partial(gen_ere, split="train", mode="rich_ere", task="EAE")
ERE_RICH_EAE_DEV = partial(gen_ere, split="dev", mode="rich_ere", task="EAE")
ERE_RICH_EAE_TEST = partial(gen_ere, split="test", mode="rich_ere", task="EAE")

ERE_RICH_QA_TRAIN = partial(gen_ere, split="train", mode="rich_ere", task="QA")
ERE_RICH_QA_DEV = partial(gen_ere, split="dev", mode="rich_ere", task="QA")
ERE_RICH_QA_TEST = partial(gen_ere, split="test", mode="rich_ere", task="QA")

ERE_RICH_INFILLING_TRAIN = partial(
    gen_ere, split="train", mode="rich_ere", task="INFILLING"
)
ERE_RICH_INFILLING_DEV = partial(
    gen_ere, split="dev", mode="rich_ere", task="INFILLING"
)
ERE_RICH_INFILLING_TEST = partial(
    gen_ere, split="test", mode="rich_ere", task="INFILLING"
)

#
# FrameNet
#

# FrameNet for EAE (all roles)
FRAMENET_EAE_TRAIN = partial(gen_framenet, split="train")
FRAMENET_EAE_DEV = partial(gen_framenet, split="dev")
FRAMENET_EAE_TEST = partial(gen_framenet, split="test")

# FrameNet for EAE (core roles only)
FRAMENET_EAE_CORE_ROLES_ONLY_TRAIN = partial(
    gen_framenet, split="train", core_roles_only=True
)
FRAMENET_EAE_CORE_ROLES_ONLY_DEV = partial(
    gen_framenet, split="dev", core_roles_only=True
)
FRAMENET_EAE_CORE_ROLES_ONLY_TEST = partial(
    gen_framenet, split="test", core_roles_only=True
)

# FrameNet for FAMuS
# (FrameNet examples with FAMuS ontology)
FRAMENET_FOR_FAMUS_INFILLING_TRAIN = partial(
    gen_framenet, split="train", use_famus_ontology=True, task="INFILLING"
)
FRAMENET_FOR_FAMUS_INFILLING_DEV = partial(
    gen_framenet, split="dev", use_famus_ontology=True, task="INFILLING"
)
FRAMENET_FOR_FAMUS_INFILLING_TEST = partial(
    gen_framenet, split="test", use_famus_ontology=True, task="INFILLING"
)

FRAMENET_FOR_FAMUS_QA_TRAIN = partial(
    gen_framenet, split="train", use_famus_ontology=True, task="QA"
)
FRAMENET_FOR_FAMUS_QA_DEV = partial(
    gen_framenet, split="dev", use_famus_ontology=True, task="QA"
)
FRAMENET_FOR_FAMUS_QA_TEST = partial(
    gen_framenet, split="test", use_famus_ontology=True, task="QA"
)


#
# FAMUS
#

# FAMuS Reports
FAMUS_REPORTS_TRAIN = partial(gen_famus, split="train", do_source=False, task="EAE")
FAMUS_REPORTS_DEV = partial(gen_famus, split="dev", do_source=False, task="EAE")
FAMUS_REPORTS_TEST = partial(gen_famus, split="test", do_source=False, task="EAE")

FAMUS_REPORTS_QA_TRAIN = partial(gen_famus, split="train", do_source=False, task="QA")
FAMUS_REPORTS_QA_DEV = partial(gen_famus, split="dev", do_source=False, task="QA")
FAMUS_REPORTS_QA_TEST = partial(gen_famus, split="test", do_source=False, task="QA")

FAMUS_REPORTS_INFILLING_TRAIN = partial(
    gen_famus, split="train", do_source=False, task="INFILLING"
)
FAMUS_REPORTS_INFILLING_DEV = partial(
    gen_famus, split="dev", do_source=False, task="INFILLING"
)
FAMUS_REPORTS_INFILLING_TEST = partial(
    gen_famus, split="test", do_source=False, task="INFILLING"
)


DATASET_TO_SPLITS = {
    TaskChoice.OTE: {
        DatasetChoice.RAMS: {
            "train": RAMS_OTE_TRAIN,
            "dev": RAMS_OTE_DEV,
            "test": RAMS_OTE_TEST,
        },
        DatasetChoice.WIKIEVENTS: {
            "train": WIKIEVENTS_OTE_TRAIN,
            "dev": WIKIEVENTS_OTE_DEV,
            "test": WIKIEVENTS_OTE_TEST,
        },
    },
    TaskChoice.EAE: {
        DatasetChoice.ACE: {
            "train": ACE_EAE_TRAIN,
            "dev": ACE_EAE_DEV,
            "test": ACE_EAE_TEST,
        },
        DatasetChoice.ERE_LIGHT: {
            "train": ERE_LIGHT_EAE_TRAIN,
            "dev": ERE_LIGHT_EAE_DEV,
            "test": ERE_LIGHT_EAE_TEST,
        },
        DatasetChoice.ERE_RICH: {
            "train": ERE_RICH_EAE_TRAIN,
            "dev": ERE_RICH_EAE_DEV,
            "test": ERE_RICH_EAE_TEST,
        },
        DatasetChoice.FRAMENET: {
            "train": FRAMENET_EAE_TRAIN,
            "dev": FRAMENET_EAE_DEV,
            "test": FRAMENET_EAE_TEST,
        },
        DatasetChoice.FRAMENET_CORE_ROLES: {
            "train": FRAMENET_EAE_CORE_ROLES_ONLY_TRAIN,
            "dev": FRAMENET_EAE_CORE_ROLES_ONLY_DEV,
            "test": FRAMENET_EAE_CORE_ROLES_ONLY_TEST,
        },
        DatasetChoice.FAMUS_REPORTS: {
            "train": FAMUS_REPORTS_TRAIN,
            "dev": FAMUS_REPORTS_DEV,
            "test": FAMUS_REPORTS_TEST,
        },
        DatasetChoice.RAMS: {
            "train": RAMS_EAE_TRAIN,
            "dev": RAMS_EAE_DEV,
            "test": RAMS_EAE_TEST,
        },
        DatasetChoice.WIKIEVENTS: {
            "train": WIKIEVENTS_EAE_TRAIN,
            "dev": WIKIEVENTS_EAE_DEV,
            "test": WIKIEVENTS_EAE_TEST,
        },
    },
    TaskChoice.QA: {
        DatasetChoice.ACE: {
            "train": ACE_QA_TRAIN,
            "dev": ACE_QA_DEV,
            "test": ACE_QA_TEST,
        },
        DatasetChoice.ERE_LIGHT: {
            "train": ERE_LIGHT_QA_TRAIN,
            "dev": ERE_LIGHT_QA_DEV,
            "test": ERE_LIGHT_QA_TEST,
        },
        DatasetChoice.ERE_RICH: {
            "train": ERE_RICH_QA_TRAIN,
            "dev": ERE_RICH_QA_DEV,
            "test": ERE_RICH_QA_TEST,
        },
        DatasetChoice.FAMUS_REPORTS: {
            "train": FAMUS_REPORTS_QA_TRAIN,
            "dev": FAMUS_REPORTS_QA_DEV,
            "test": FAMUS_REPORTS_QA_TEST,
        },
        DatasetChoice.FRAMENET_FOR_FAMUS: {
            "train": FRAMENET_FOR_FAMUS_QA_TRAIN,
            "dev": FRAMENET_FOR_FAMUS_QA_DEV,
            "test": FRAMENET_FOR_FAMUS_QA_TEST,
        },
        DatasetChoice.RAMS: {
            "train": RAMS_QA_TRAIN,
            "dev": RAMS_QA_DEV,
            "test": RAMS_QA_TEST,
        },
        DatasetChoice.WIKIEVENTS: {
            "train": WIKIEVENTS_QA_TRAIN,
            "dev": WIKIEVENTS_QA_DEV,
            "test": WIKIEVENTS_QA_TEST,
        },
    },
    TaskChoice.INFILLING: {
        DatasetChoice.ACE: {
            "train": ACE_INFILLING_TRAIN,
            "dev": ACE_INFILLING_DEV,
            "test": ACE_INFILLING_TEST,
        },
        DatasetChoice.ERE_LIGHT: {
            "train": ERE_LIGHT_INFILLING_TRAIN,
            "dev": ERE_LIGHT_INFILLING_DEV,
            "test": ERE_LIGHT_INFILLING_TEST,
        },
        DatasetChoice.ERE_RICH: {
            "train": ERE_RICH_INFILLING_TRAIN,
            "dev": ERE_RICH_INFILLING_DEV,
            "test": ERE_RICH_INFILLING_TEST,
        },
        DatasetChoice.FAMUS_REPORTS: {
            "train": FAMUS_REPORTS_INFILLING_TRAIN,
            "dev": FAMUS_REPORTS_INFILLING_DEV,
            "test": FAMUS_REPORTS_INFILLING_TEST,
        },
        DatasetChoice.FRAMENET_FOR_FAMUS: {
            "train": FRAMENET_FOR_FAMUS_INFILLING_TRAIN,
            "dev": FRAMENET_FOR_FAMUS_INFILLING_DEV,
            "test": FRAMENET_FOR_FAMUS_INFILLING_TEST,
        },
        DatasetChoice.RAMS: {
            "train": RAMS_INFILLING_TRAIN,
            "dev": RAMS_INFILLING_DEV,
            "test": RAMS_INFILLING_TEST,
        },
        DatasetChoice.WIKIEVENTS: {
            "train": WIKIEVENTS_INFILLING_TRAIN,
            "dev": WIKIEVENTS_INFILLING_DEV,
            "test": WIKIEVENTS_INFILLING_TEST,
        },
    },
}


if __name__ == "__main__":
    from tqdm import tqdm

    for ex in tqdm(gen_framenet("test", task="QA", use_famus_ontology=True)):
        pass
