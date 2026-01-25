import io
import shlex
from contextlib import redirect_stdout

import pytest

from cli.keyword_search_cli import main as key_word_search
from cli.semantic_search_cli import main as semantic_search

keyword_search_commands = [
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
    ["bm25tf 1 anbuselvan", "2.35"],
    ["bm25tf 1 police", "2.09"],
    ["bm25tf 1 maya", "2.24"],
    ["bm25tf 1 cheese", "0.00"],
    # ["bm25search 'love story'", "The Inner Life of Martin Frost\n4.60\nDefinitely, Maybe\n4.51\nCama de Gato\n4.40"],
    # ["bm25search 'animated family'", "Gakuen Alice\n7.35\nDay of the Animals\n7.14\nFantastic Mr. Fox\n6.91"],
]


@pytest.mark.parametrize("command", keyword_search_commands, ids=lambda cmd_pair: cmd_pair[0])
def test_search_cli(command: list[str]):
    cmd, output = command
    args = shlex.split(cmd)
    f = io.StringIO()
    with redirect_stdout(f):
        key_word_search(args)
    captured_output = f.getvalue()
    if output:
        print(captured_output)
        if "\n" not in output:
            assert output in captured_output
        else:
            for line in output.split("\n"):
                assert line in captured_output


semantic_search_commands = [["verify", "Model loaded: SentenceTransformer\nMax sequence length: 256"]]


@pytest.mark.parametrize("command", semantic_search_commands, ids=lambda cmd_pair: cmd_pair[0])
def test_semantic_search_cli(command: list[str]):
    cmd, output = command
    args = shlex.split(cmd)
    f = io.StringIO()
    with redirect_stdout(f):
        semantic_search(args)
    captured_output = f.getvalue()
    if output:
        print(captured_output)
        if "\n" not in output:
            assert output in captured_output
        else:
            for line in output.split("\n"):
                assert line in captured_output
