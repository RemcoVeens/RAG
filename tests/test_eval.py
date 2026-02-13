import io
import shlex
from contextlib import redirect_stdout

import pytest

from cli.evaluation_cli import main as evaluation_cli

eval_commands = [
    ["--limit 4", "children's animated bear adventure\nPrecision@4: 0.2500\nRecall@4: 0.0769\nF1 Score: 0.1176\n"],
    ["--limit 8", "friendship transformation magic with bears\nPrecision@8: 0.2500\nRecall@8: 0.6667\nF1 Score: 0.3636\n"],
]


@pytest.mark.parametrize("command", eval_commands, ids=lambda cmd_pair: cmd_pair[0])
def test_semantic(command: list[str]):
    if len(command) == 2:
        cmd, output = command
    else:
        cmd = command[0]
        output = "".join(command[1:])
    args = shlex.split(cmd)
    f = io.StringIO()
    with redirect_stdout(f):
        evaluation_cli(args)
    captured_output = f.getvalue()
    if output:
        print(captured_output)
        if "\n" not in output:
            assert output in captured_output
        else:
            for line in output.split("\n"):
                assert line in captured_output
