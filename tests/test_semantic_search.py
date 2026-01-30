import io
import shlex
from contextlib import redirect_stdout

import pytest

from cli.semantic_search_cli import main as semantic_search

semantic_search_commands = [
    ["verify", "Model loaded: SentenceTransformer\nMax sequence length: 256"],
    ["embed_text 'Luke, I am your father'", "0.035\n-0.016\n0.043\nDimensions: 384"],
    ["verify_embeddings", "5000 vectors in 384 dimensions"],
    ["embedquery 'funny bear movies'", "-0.072\n-0.014\n0.001\n384"],
    ["embedquery 'FUNNY bear MOVIES'", "-0.072\n-0.014\n0.001\n384"],
    ["embedquery 'scary dinosaur'", "-0.101\n0.038\n-0.028\n384"],
    ["search 'funny bear movies'", "Bear\nThe Great Bear\nA Bear for Punishment"],
    ["search 'space adventure' ", "Spaceflight\nAdventureland\nOdyssey 5"],
]


@pytest.mark.parametrize("command", semantic_search_commands, ids=lambda cmd_pair: cmd_pair[0])
def test_semantic(command: list[str]):
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
