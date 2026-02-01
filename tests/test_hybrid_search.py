import io
import shlex
from contextlib import redirect_stdout

import pytest

from cli.hybrid_search_cli import main as hybrid_search

hybrid_search_commands = [
    ["normalize 0.5\n2.3\n1.2\n0.5\n0.1", "0.1818\n1.0\n0.5\n0.1818\n0.0"],
    ["normalize 5 5 5", "1.0"],
    ["weighted-search 'British Bear' --alpha 0.5 --limit 25", "Paddington\nThe Country Bears\nLegends of the Fall"],
    ["weighted-search 'British Bear' --alpha 0.2 --limit 25", "Paddington\nLegends of the Fall\nThe Edge"],
    ["weighted-search 'British Bear' --alpha 0.8 --limit 25", "Paddington\nThe Duchess\nThe Great Bear"],
]


@pytest.mark.parametrize("command", hybrid_search_commands, ids=lambda cmd_pair: cmd_pair[0])
def test_hybrid(command: list[str]):
    if len(command) == 2:
        cmd, output = command
    else:
        cmd = command[0]
        output = "".join(command[1:])
    args = shlex.split(cmd)
    f = io.StringIO()
    with redirect_stdout(f):
        hybrid_search(args)
    captured_output = f.getvalue()
    if output:
        print(captured_output)
        if "\n" not in output:
            assert output in captured_output
        else:
            for line in output.split("\n"):
                assert line in captured_output
