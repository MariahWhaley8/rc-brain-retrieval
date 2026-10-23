import os

import voyageai.error

import indexer


class VoyageResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


def test_transient_voyage_errors_are_retried():
    os.environ["VOYAGE_MIN_INTERVAL_SECONDS"] = "0"
    os.environ["VOYAGE_EMBED_BATCH_SIZE"] = "32"
    sleeps = []

    class Client:
        def __init__(self):
            self.calls = 0

        def embed(self, texts, model, input_type):
            self.calls += 1
            if self.calls == 1:
                raise voyageai.error.RateLimitError("slow down", headers={"retry-after": "0"})
            if self.calls == 2:
                raise voyageai.error.ServiceUnavailableError("unavailable")
            return VoyageResult([[float(i)] for i in range(len(texts))])

    client = Client()
    vectors = indexer.embed_texts(
        ["one", "two"],
        client=client,
        sleep=sleeps.append,
        rand=lambda: 0,
        max_attempts=4,
    )
    assert vectors == [[0.0], [1.0]]
    assert client.calls == 3
    assert sleeps == [0.0, 2.0]


def test_token_limit_splits_the_batch():
    os.environ["VOYAGE_MIN_INTERVAL_SECONDS"] = "0"
    os.environ["VOYAGE_EMBED_BATCH_SIZE"] = "64"

    class Client:
        def __init__(self):
            self.sizes = []

        def embed(self, texts, model, input_type):
            self.sizes.append(len(texts))
            if len(texts) > 1:
                raise voyageai.error.InvalidRequestError(
                    "max tokens per request exceeded"
                )
            return VoyageResult([[1.0]])

    client = Client()
    vectors = indexer.embed_texts(
        ["a", "b", "c", "d"],
        client=client,
        sleep=lambda _seconds: None,
        rand=lambda: 0,
    )
    assert vectors == [[1.0], [1.0], [1.0], [1.0]]
    assert client.sizes[0] == 4
    assert client.sizes.count(1) == 4
