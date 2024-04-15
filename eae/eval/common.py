"""Metrics for EAE evaluation

Existing implementations of mentions in the Metametric library rely on offset-based
mentions, whereas here we want to use lexical matching. As such, we have to define new
dataclasses based on a lexical notion of a mention.
"""
import metametric.dsl as mm
import re

from collections import defaultdict
from dataclasses import dataclass
from metametric.core.decorator import metametric
from metametric.core.metric_suite import MetricFamily
from metametric.core.reduction import MicroAverage
from metametric.core.normalizers import Precision, Recall, FScore
from spacy.lang.en import English
from typing import Any, Collection, Dict, List, Set, Tuple

from eae.dataset.common import EVENT_SEP, ARG_SEP, ROLE_SEP

nlp = English()
tokenizer = nlp.tokenizer

SINGLE_QUOTE_RE = re.compile(r"' (.*?) '")
DOUBLE_QUOTE_RE = re.compile(r"\" (.*?) \"")


@metametric()
@dataclass(frozen=True, eq=True)
class Mention:
    """A mention, consisting of a single string

    :param mention: the string that defines the mention
    """

    mention: str


@metametric()
@dataclass
class Entity:
    """An entity, consisting of multiple mentions

    :param mentions: the set of mentions that define the entity
    """

    mentions: Collection[Mention]


@metametric()
@dataclass(frozen=True, eq=True)
class EntityArgument:
    """An argument entity commonly used in event extraction.

    :param role: The role satisfied by the argument.
    :param entity: The entity participant satisfying the role.
    """

    role: str
    entity: Entity


@metametric()
@dataclass
class Event:
    """An event commonly used in event extraction.

    :param trigger: The lexical trigger of the event.
    :param args: The arguments of the event.
    """

    trigger: Mention
    args: Collection[EntityArgument]


# Trigger precision, recall, and F1
trigger = mm.normalize["none"](
    mm.from_func(lambda e1, e2: 1.0 if e1.trigger == e2.trigger else 0.0)
)
trigger_metrics = MetricFamily(
    mm.set_matching[Event, "<->", "none"](trigger),
    MicroAverage([Precision(), Recall(), FScore()]),
)

# the phi-subset similarity function for entities
phi_subset = mm.normalize["none"](
    mm.from_func(
        lambda e1, e2: 1.0 if set(e1.mentions).issubset(set(e2.mentions)) else 0.0
    )
)
# the phi-subset similarity function, generalized to arguments
entity_argument_phi_subset = mm.normalize["none"](
    mm.dataclass[EntityArgument]({"entity": phi_subset, "role": mm.auto[str]})
)

# The CEAF-REE similarity metric, using phi-subset as the entity/argument similarity
ceaf_ree_phi_subset = mm.dataclass[Event](
    {
        "trigger": mm.auto[Mention],
        "args": mm.set_matching[EntityArgument, "<->", "none"](
            entity_argument_phi_subset
        ),
    }
)

# CEAF-REE w/ subset similarity for *multiple* events
ceaf_ree_phi_subset_set_match = mm.set_matching[Event, "<->", "none"](
    ceaf_ree_phi_subset
)
ceaf_ree_phi_subset_metrics = MetricFamily(
    ceaf_ree_phi_subset_set_match, MicroAverage([Precision(), Recall(), FScore()])
)

# The CEAF-REE similarity metric, using phi4 as the entity similarity
ceaf_ree_phi4_set_match = mm.set_matching[Event, "<->", "none"](mm.auto[Event])
ceaf_ree_phi4_metrics = MetricFamily(
    ceaf_ree_phi4_set_match, MicroAverage([Precision(), Recall(), FScore()])
)


def normalize_text(text: str) -> str:
    """Normalize mentions for evaluation"""
    if text == "i m":
        return "im"
    text = text.replace(" -", "-")
    text = text.replace("- ", "-")
    text = text.replace(" 'd ", "'d ")
    text = text.replace(" 's", "'s")
    text = text.replace(" 'S", "'S")
    text = text.replace(" ’s", "’s")
    text = text.replace(" ,", ",")
    text = text.replace(" / ", "/")
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    text = text.replace(" n't", "n't")
    text = text.replace(" 've", "'ve")
    text = re.sub(SINGLE_QUOTE_RE, r"'\1'", text)
    text = re.sub(DOUBLE_QUOTE_RE, r'"\1"', text)
    text = text.replace(' " ', '" ')
    text = text.replace(" ' ", "' ")
    text = text.replace("$ ", "$")
    if text.endswith(" ."):
        text = text[:-2] + "."
    return text.strip()


def extract_events_from_reference(
    example: Dict[str, Any], do_eae: bool = False
) -> List[Event]:
    """Extract events from the 'events' field of an OTE example

    :param example: the example from which to extract events
    :param do_eae: whether this is an EAE example (or else an OTE example)
    :return: a list of extracted reference events
    """
    if do_eae:
        in_events = [example["event"]]
    else:
        in_events = example["events"]
    out_events = []
    for e in in_events:
        if do_eae:
            # we don't care about the trigger in EAE
            trigger = ""
        else:
            trigger = e["trigger"]["text"]
        entities_by_role = defaultdict(lambda: defaultdict(set))
        for a in e["arguments"]:
            entities_by_role[a["role"].lower()][a["entity_id"]].add(a["text"])
        args = []
        for r, entities in entities_by_role.items():
            for mentions in entities.values():
                # Assume one entity per mention
                for m in mentions:
                    args.append(
                        EntityArgument(
                            role=r, entity=Entity(mentions=[Mention(normalize_text(m))])
                        )
                    )
        out_events.append(Event(trigger=Mention(trigger), args=args))
    return out_events


def to_dataclass(trigger: str, args_by_role: Dict[str, List[str]]) -> Event:
    return Event(
        trigger=Mention(trigger),
        args=[
            # we assume a single mention is predicted per argument
            EntityArgument(
                role=role.lower(),
                entity=Entity(mentions=[Mention(arg)]),
            )
            for role, args in args_by_role.items()
            for arg in args
            if arg  # ignore empty args
        ],
    )


def extract_event_from_infilling_template(
    trigger: str, role_types: Set[str], unfilled_template: str, filled_template: str
) -> Event:
    """Extract an event from an infilled template
    
    :param trigger: the trigger of the event
    :param role_types: the set of role types
    :param unfilled_template: the unfilled template
    :param filled_template: the filled template
    :return: the extracted event
    """
    def toks_to_str(toks: List[str]) -> str:
        arg_str = " ".join(toks)
        return normalize_text(arg_str)

    # normalize role types by lowercasing
    role_types_norm = {r.lower() for r in role_types}
    args_by_role = defaultdict(list)
    unfilled_template_tok = [tok.text for tok in tokenizer(unfilled_template)]
    filled_template_tok = [tok.text for tok in tokenizer(filled_template)]
    i = 0
    j = 0
    while i < len(unfilled_template_tok) and j < len(filled_template_tok):
        i_tok = unfilled_template_tok[i]
        j_tok = filled_template_tok[j]
        if i_tok.lower() in role_types_norm:
            if i_tok.lower() == j_tok.lower():
                i += 1
                j += 1
            else:
                arg_start_tok = j
                while (j < len(filled_template_tok)) and (
                    (i == len(unfilled_template_tok) - 1)
                    or (filled_template_tok[j] != unfilled_template_tok[i + 1])
                ):
                    j += 1
                curr_arg = []
                for tok in filled_template_tok[arg_start_tok:j]:
                    if tok != "AND":
                        curr_arg.append(tok)
                    else:
                        args_by_role[i_tok.lower()].append(toks_to_str(curr_arg))
                        curr_arg = []
                args_by_role[i_tok.lower()].append(toks_to_str(curr_arg))
                i += 1
        else:
            i += 1
            j += 1

    return to_dataclass(trigger, args_by_role)


def extract_events_from_str(
    event_str: str, do_eae: bool = False
) -> Tuple[List[Event], int, int]:
    """Extract events from a linearized representation of a set of events

    :param event_str: the linearized representation of the events
    :param do_eae: whether this is an EAE example (or else an OTE example)
    :return: a list of events, where each event is a tuple of (trigger, args_by_role)
        also returns the number of errors encountered in processing the events, as
        well as the total number of events
    """

    # events are separated by EVENT_SEP
    events = event_str.split(EVENT_SEP)
    out = []
    errors = 0
    args_by_role = defaultdict(list)
    for e in events:
        # this means no roles were predicted for this event
        if not do_eae and len(e.split(ROLE_SEP)) < 2:
            errors += 1
            continue
        # format for OTE: trigger<role_sep>role1<arg_sep>role1_arg1<arg_sep>role1_arg2...<role_sep>role2<arg_sep>...
        # format for EAE: role1<arg_sep>role1_arg1<arg_sep>role1_arg2...<role_sep>role2<arg_sep>...
        if do_eae:
            trigger = ""
            all_args = e.split(ROLE_SEP)
        else:
            trigger, *all_args = e.split(ROLE_SEP)

        for args in all_args:
            role, *args = args.split(ARG_SEP)
            args_by_role[role.lower().strip()].extend([a.strip() for a in args])
        out.append(to_dataclass(trigger.strip(), args_by_role))
    return out, len(events), errors


if __name__ == "__main__":
    # Testing the implementation of the CEAF-REE (subset) metric
    # on examples from the appendix of the following paper:
    # https://aclanthology.org/2021.eacl-main.52/
    m1 = Mention("Pilmai telephone company building")
    m2 = Mention("telephone company building")
    m3 = Mention("telephone company offices")
    m4 = Mention("water pipes")
    m5 = Mention("public telephone booth")

    g_arg1 = EntityArgument("r", Entity([m1, m2, m3]))
    g_arg2 = EntityArgument("r", Entity([m4]))
    g_arg3 = EntityArgument("r", Entity([m5]))
    g_event = Event(Mention("foo"), [g_arg1, g_arg2, g_arg3])

    p_arg1 = EntityArgument("r", Entity([m4]))
    p_arg2 = EntityArgument("r", Entity([m1]))
    p_arg3 = EntityArgument("r", Entity([m5]))
    p_arg4 = EntityArgument("r", Entity([m3]))

    subset = ceaf_ree_phi_subset_metrics.new()
    phi4 = ceaf_ree_phi4_metrics.new()

    # case 1
    p_event_case1 = Event(Mention("foo"), [p_arg1, p_arg2, p_arg3, p_arg4])
    subset.update_single([p_event_case1], [g_event])
    phi4.update_single([p_event_case1], [g_event])

    print("-------")
    print("case 1:")
    print("-------")
    print("-----------")
    print("phi-subset:")
    print("-----------")
    print(subset.compute())
    print("------")
    print("phi-4:")
    print("------")
    print(phi4.compute())

    # case 2:
    p_event_case2 = Event(Mention("foo"), [p_arg2, p_arg1, p_arg3])
    subset.reset()
    subset.update_single([p_event_case2], [g_event])
    phi4.reset()
    phi4.update_single([p_event_case2], [g_event])

    print("-------")
    print("case 2:")
    print("-------")
    print("-----------")
    print("phi-subset:")
    print("-----------")
    print(subset.compute())
    print("------")
    print("phi-4:")
    print("------")
    print(phi4.compute())

    # case 3:
    p_event_case3 = Event(Mention("foo"), [p_arg2, p_arg3])
    subset.reset()
    subset.update_single([p_event_case3], [g_event])
    phi4.reset()
    phi4.update_single([p_event_case3], [g_event])

    print("-------")
    print("case 3:")
    print("-------")
    print("-----------")
    print("phi-subset:")
    print("-----------")
    print(subset.compute())
    print("------")
    print("phi-4:")
    print("------")
    print(phi4.compute())
