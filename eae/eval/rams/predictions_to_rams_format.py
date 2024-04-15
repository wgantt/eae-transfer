"""Script for converting EAE predictions to RAMS format.

To score the file output by this script ('output-file'), run the official
RAMS evaluation script as follows (we assume you are scoring the test split):

python scripts/eval/rams/scorer.py \
    --gold_file data/rams/test.jsonl \
    --pred_file <your_output_file> \
    --reuse_gold_format \
    --do_all
    
Constrained decoding with the scorer is not yet supported,
owing to a bug that seems to prevent us from setting both
--reuse_gold_format and -cd.
"""
import click
import json

from collections import defaultdict
from sacremoses import MosesDetokenizer
from typing import Any, Dict, List

from eae.eval.common import extract_events_from_str

MD = MosesDetokenizer("en")


@click.command()
@click.argument("output-file", type=str)
@click.argument("pred-file", type=str)
@click.option(
    "--gold-file",
    type=str,
    help="the gold RAMS file",
    default="data/rams/test.jsonl",
)
def convert_wrapper(output_file, gold_file, pred_file) -> None:
    """Wrapper for convert function that allows it to be used as a CLI command."""
    convert(output_file, gold_file, pred_file)


def convert(
    output_file: str,
    gold_file: str,
    predictions: str | List[Dict[str, Any]],
) -> None:
    """Convert EAE predictions to RAMS format.

    :param output_file: the JSONL-formatted output file where
        the converted predictions will be written to
    :param predictions: the JSONL-formatted predictions file (or else a list of these predictions)
    :param gold_file: the JSONL-formatted gold file
    """
    if isinstance(predictions, str):
        with open(predictions, "r") as f:
            preds = [json.loads(line) for line in f]
    else:
        preds = predictions

    with open(gold_file, "r") as f:
        golds = [json.loads(line) for line in f]
    total_predicted_events = 0
    total_errors = 0
    for p, r in zip(preds, golds):
        pred_doc_id, pred_event_type = p["instance_id"].split("-")
        gold_doc_id = r["doc_key"]

        # sanity check that the doc ids match
        assert (
            pred_doc_id == gold_doc_id
        ), f"Mismatched doc ids: {pred_doc_id} != {gold_doc_id}"

        # prediction and reference should each only ever contain one event
        assert (
            len(r["evt_triggers"]) == 1
        ), f"Expected exactly one event in document {gold_doc_id}, but got {len(r['evt_triggers'])}"
        predicted_event, n_events, n_errors = extract_events_from_str(
            p["prediction"], do_eae=True
        )

        # the types of these events should match
        gold_event_type = r["evt_triggers"][0][2][0][0]
        assert (
            pred_event_type == gold_event_type
        ), f"Mismatched event types: {pred_event_type} != {gold_event_type}"

        # bookkeeping
        total_predicted_events += n_events
        total_errors = n_errors

        assert (
            len(predicted_event) <= 1
        ), f"Expected <= 1 predicted event for document {gold_doc_id} but got {len(predicted_event)}"

        pred_args_by_role = defaultdict(list)
        if len(predicted_event) == 1:
            predicted_event = predicted_event[0]
            # get all predicted arguments
            for arg in predicted_event.args:
                assert (
                    len(arg.entity.mentions) == 1
                ), f"Expected exactly one mention for arg {arg} in document {gold_doc_id}"
                mention = arg.entity.mentions[0].mention
                pred_args_by_role[arg.role].append(mention)

        # construct the data structures for the arguments to be output
        out_evt_links = []
        if len(pred_args_by_role) > 0:
            doc_toks = [tok for s in r["sentences"] for tok in s]
            trigger_tok_start = r["evt_triggers"][0][0]
            trigger_tok_end = r["evt_triggers"][0][1]

            # add all predicted arguments that match some gold argument;
            # i.e., role must match and the predicted argument string must
            # exactly match some gold argument string with the same role
            for arg in r["gold_evt_links"]:
                role = arg[2][11:]
                if role not in pred_args_by_role:
                    continue
                arg_str = MD.detokenize(doc_toks[arg[1][0] : arg[1][1] + 1])
                if arg_str in pred_args_by_role[role]:
                    out_evt_links.append(arg)
                    pred_args_by_role[role].remove(arg_str)

            # any remaining predicted arguments are spurious ones; add these
            for role, args in pred_args_by_role.items():
                for arg in args:
                    out_evt_links.append(
                        [
                            (trigger_tok_start, trigger_tok_end),
                            # we just use (-1, -1) for the span, since these are necessarily spurious arguments
                            (
                                -1,
                                -1,
                            ),
                            # this is a hack to make sure the role format is parsed correctly by the scorer
                            "0" * 11 + role,
                        ]
                    )
        r["gold_evt_links"] = out_evt_links
    print(
        f"{total_errors}/{total_predicted_events} predicted events could not be parsed due to errors"
    )
    with open(output_file, "w") as f:
        f.write("\n".join(list(map(json.dumps, golds))))


if __name__ == "__main__":
    convert()
