import io
import shlex
from contextlib import redirect_stdout

import pytest

from cli.keyword_search_cli import main

commands = [
    ["search 'brave'", "Madrasapattinam"],
    ["search 'nonsensetoken assault'", "Klansman"],
    ["tf 424 bear", "1"],
    ["tf 424 trapper", "4"],
    ["idf grizzly", "5.52"],
    ["idf actor", "3.29"],
    ["idf man", "0.76"],
    ["tfidf 424 trapper", "24.13"],
    ["tfidf 424 push", "2.14"],
    ["bm25idf grizzly", "5.55"],
    ["bm25idf actor", "3.29"],
    ["bm25idf love", "0.95"],
    ["bm25tf 1 anbuselvan", "2.31"],
    ["bm25tf 1 maya", "2.17"],
    ["bm25tf 1 police", "2.00"],
]


@pytest.mark.parametrize("command", commands, ids=lambda cmd_pair: cmd_pair[0])
def test(command: list[str]):
    cmd, output = command
    args = shlex.split(cmd)
    f = io.StringIO()
    with redirect_stdout(f):
        main(args)
    captured_output = f.getvalue()
    if output:
        print(captured_output)
        assert output in captured_output
