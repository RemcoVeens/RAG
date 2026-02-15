import io
import shlex
from contextlib import redirect_stdout

import pytest

from cli.augmented_generation_cli import main

commands = [
    ["rag 'movies about action and dinosaurs'", "Jurassic Park"],
    ["summarize 'movies about action and dinosaurs'", "Jurassic Park"],
    ["citations 'action movie with lasers'", "Eliminators"],
]


@pytest.mark.parametrize("command", commands, ids=lambda cmd_pair: cmd_pair[0])
def test_search(command: list[str]):
    cmd, output = command
    args = shlex.split(cmd)
    f = io.StringIO()
    with redirect_stdout(f):
        main(args)
    captured_output = f.getvalue()
    if output:
        print(captured_output)
        if "\n" not in output:
            assert output in captured_output
        else:
            for line in output.split("\n"):
                assert line in captured_output
