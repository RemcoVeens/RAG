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
    [
        "chunk 'This is a test text with more than ten words to see how chunking works' --chunk-size 5",
        "Chunking 70 characters\n1. This is a test text\n2. with more than ten words\n3. to see how chunking works",
    ],
    [
        "chunk 'The quick brown fox jumps over the lazy dog and then runs through the forest quickly' --chunk-size 10",
        "Chunking 84 characters\n1. The quick brown fox jumps over the lazy dog and\n",
        "2. then runs through the forest quickly",
    ],
    ["chunk 'Short text'", "Chunking 10 characters\n1. Short text"],
    [
        "chunk 'This is a test text with two chunks' --chunk-size 5 --overlap 2",
        "Chunking 35 characters\n1. This is a test text\n2. test text with two chunks",
    ],
    [
        "chunk 'The bear attack was very terrifying.' --chunk-size 4 --overlap 1",
        "Chunking 36 characters\n1. The bear attack was\n2. was very terrifying.",
    ],
    [
        "chunk 'Zero overlap means no shared words' --chunk-size 3 --overlap 0",
        "Chunking 34 characters\n1. Zero overlap means\n2. no shared words",
    ],
    [
        "semantic_chunk 'This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth sentence. This is the fifth sentence.' --max-chunk-size 3",  # noqa
        "Semantically chunking 141 characters\n1. This is the first sentence. This is the second sentence. ",
        "This is the third sentence.\n2. This is the fourth sentence. This is the fifth sentence.",
    ],
    [
        "semantic_chunk 'First sentence here. Second sentence here. Third sentence here. Fourth sentence here.' --max-chunk-size 2 --overlap 1",  # noqa
        "Semantically chunking 85 characters\n1. First sentence here. Second sentence here.\n2. Second sentence here. ",
        "Third sentence here.\n3. Third sentence here. Fourth sentence here.",
    ],
    [
        "semantic_chunk 'Only one sentence here.' --max-chunk-size 3",
        "Semantically chunking 23 characters\nOnly one sentence here.",
    ],
    [
        "semantic_chunk 'Sentence one. Sentence two. Sentence three.' --max-chunk-size 3 --overlap 1",
        "Semantically chunking 43 characters\nSentence one. Sentence two.\nSentence two. Sentence three.",
    ],
    ["embed_chunks", "Generated 72909 chunked embeddings"],
    ["search_chunked 'superhero action movie' --limit 25", "Kick-Ass\nThe Incredibles\nLogan"],
    ["search_chunked 'romantic comedy' --limit 25", "Austenland\nL'amant\nYou, Me and Dupree"],
]


@pytest.mark.parametrize("command", semantic_search_commands, ids=lambda cmd_pair: cmd_pair[0])
def test_semantic(command: list[str]):
    if len(command) == 2:
        cmd, output = command
    else:
        cmd = command[0]
        output = "".join(command[1:])
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
