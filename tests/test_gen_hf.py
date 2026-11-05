"""msd.gen.hf_backend against fake `torch` and `transformers` modules injected through sys.modules.

Nothing here loads a model. The fakes are small but behave like the real calls where it matters:
the fake tokenizer refuses to pad without a pad token, honours padding_side, and the fake model
returns prompt-plus-continuation rows the way `generate` does.
"""

import contextlib
import sys
import types

import pytest

from msd import gen
from msd.gen import GenOut

PAD, EOS, END_RESPONSE = 0, 1, 2  # special ids; text ids are 1000 + ord(char)
OFFSET = 1000
TOKENS = {"<pad>": PAD, "<eos>": EOS}


def enc(text):
    return [OFFSET + ord(c) for c in text]


def dec(ids, skip_special_tokens):
    out = []
    for i in ids:
        if i >= OFFSET:
            out.append(chr(i - OFFSET))
        elif not skip_special_tokens:
            out.append(f"<{i}>")
    return "".join(out)


class FakeTensor:
    def __init__(self, rows):
        self.rows = [list(r) for r in rows]
        self.device = None

    def to(self, device):
        self.device = device
        return self

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def __getitem__(self, key):
        rows, cols = key
        assert rows == slice(None) and isinstance(cols, slice), key
        return FakeTensor([r[cols] for r in self.rows])

    def tolist(self):
        return [list(r) for r in self.rows]


class FakeBatch(dict):
    def to(self, device):
        for tensor in self.values():
            tensor.to(device)
        return self


class FakeTokenizer:
    def __init__(self, with_pad):
        self.eos_token = "<eos>"
        self.eos_token_id = EOS
        self.pad_token = "<pad>" if with_pad else None
        self.padding_side = "right"  # the usual default: the backend must change it
        self.template_calls = []
        self.call_kwargs = []
        self.encoded = []
        self.decode_calls = []

    @property
    def pad_token_id(self):
        return None if self.pad_token is None else TOKENS[self.pad_token]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        self.template_calls.append({"messages": messages, "tokenize": tokenize,
                                    "add_generation_prompt": add_generation_prompt, **kwargs})
        assert tokenize is False  # the backend tokenizes itself, in one padded batch
        return "".join(f"[{m['role']}]{m['content']}" for m in messages) + "[assistant]"

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=True, **kwargs):
        self.call_kwargs.append({"return_tensors": return_tensors, "padding": padding,
                                 "add_special_tokens": add_special_tokens, **kwargs})
        rows = [enc(t) for t in texts]
        width = max(len(r) for r in rows)
        if padding:
            if self.pad_token is None:
                raise ValueError("Asking to pad but the tokenizer does not have a padding token.")
            pads = lambda r: [self.pad_token_id] * (width - len(r))  # noqa: E731
            if self.padding_side == "left":
                ids = [pads(r) + r for r in rows]
                mask = [[0] * (width - len(r)) + [1] * len(r) for r in rows]
            else:
                ids = [r + pads(r) for r in rows]
                mask = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
        else:
            ids, mask = rows, [[1] * len(r) for r in rows]
        self.encoded.append(ids)
        return FakeBatch(input_ids=FakeTensor(ids), attention_mask=FakeTensor(mask))

    def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
        self.decode_calls.append({"skip_special_tokens": skip_special_tokens, **kwargs})
        seqs = sequences.tolist() if hasattr(sequences, "tolist") else sequences
        return [dec(s, skip_special_tokens) for s in seqs]

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        self.decode_calls.append({"skip_special_tokens": skip_special_tokens, **kwargs})
        return dec(ids, skip_special_tokens)


class FakeModel:
    def __init__(self, eos_token_id=EOS):
        self.generation_config = types.SimpleNamespace(eos_token_id=eos_token_id)
        self.device = None
        self.evaluated = False
        self.generate_calls = []
        self.script = []  # one entry per generate() call: a list of continuations, one per batch row

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        continuations = self.script.pop(0)
        ids = kwargs["input_ids"].tolist()
        assert len(continuations) == len(ids)
        longest = min(max(len(c) for c in continuations), kwargs["max_new_tokens"])
        pad = kwargs["pad_token_id"]
        rows = [row + c[:longest] + [pad] * (longest - len(c[:longest])) for row, c in zip(ids, continuations)]
        return FakeTensor(rows)


class World:
    """The fake libraries and the objects they hand out, so tests can look inside."""

    def __init__(self, *, with_pad=True, cuda=False, transformers_version="5.22.1", eos_token_id=EOS):
        self.tokenizer = FakeTokenizer(with_pad)
        self.model = FakeModel(eos_token_id)
        self.tokenizer_loads = []
        self.model_loads = []
        self.cuda = cuda
        world = self

        torch = types.ModuleType("torch")
        torch.__version__ = "2.9.0"
        torch.float16, torch.bfloat16, torch.float32 = "torch.float16", "torch.bfloat16", "torch.float32"
        torch.cuda = types.SimpleNamespace(is_available=lambda: world.cuda)
        torch.no_grad = lambda: contextlib.nullcontext()
        self.torch = torch

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                world.tokenizer_loads.append((path, kwargs))
                return world.tokenizer

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(path, **kwargs):
                world.model_loads.append((path, kwargs))
                return world.model

        transformers = types.ModuleType("transformers")
        transformers.__version__ = transformers_version
        transformers.AutoTokenizer = AutoTokenizer
        transformers.AutoModelForCausalLM = AutoModelForCausalLM
        self.transformers = transformers

    def install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", self.torch)
        monkeypatch.setitem(sys.modules, "transformers", self.transformers)
        return self


@pytest.fixture
def world(monkeypatch):
    return World().install(monkeypatch)


# --- chat template and tokenisation ----------------------------------------------------------------------


def test_each_prompt_goes_through_the_chat_template_as_one_user_turn(world):
    world.model.script = [[enc("A"), enc("B")]]
    gen.hf_backend("some/model")(["first prompt", "second prompt"])
    calls = world.tokenizer.template_calls
    assert [c["messages"] for c in calls] == [
        [{"role": "user", "content": "first prompt"}],
        [{"role": "user", "content": "second prompt"}],
    ]
    assert all(c["add_generation_prompt"] is True and c["tokenize"] is False for c in calls)


def test_templated_text_is_tokenised_left_padded_without_a_second_bos(world):
    world.model.script = [[enc("A"), enc("B")]]
    gen.hf_backend("some/model")(["short", "a much longer prompt"])
    assert world.tokenizer.padding_side == "left"
    kwargs = world.tokenizer.call_kwargs[0]
    assert kwargs["padding"] is True and kwargs["add_special_tokens"] is False  # the template already adds BOS
    first_row, second_row = world.tokenizer.encoded[0]
    assert first_row[0] == PAD and second_row[0] != PAD  # padding sits on the left, so the prompts end flush
    assert len(first_row) == len(second_row)
    assert first_row[-1] == second_row[-1] == OFFSET + ord("]")  # both end with the generation prompt


def test_generate_gets_greedy_arguments_and_nothing_sampling_related(world):
    world.model.script = [[enc("A")]]
    gen.hf_backend("some/model", max_new_tokens=77)(["p"])
    kwargs = world.model.generate_calls[0]
    assert kwargs["do_sample"] is False
    assert kwargs["max_new_tokens"] == 77
    assert kwargs["num_beams"] == 1
    assert kwargs["pad_token_id"] == PAD
    assert set(kwargs["eos_token_id"]) == {EOS}
    assert kwargs["input_ids"].shape[0] == 1 and kwargs["attention_mask"].shape == kwargs["input_ids"].shape
    for sampling_arg in ("temperature", "top_p", "top_k", "typical_p", "seed"):
        assert sampling_arg not in kwargs  # greedy ignores them; "temperature 0" means do_sample=False


def test_default_max_new_tokens_is_256(world):
    world.model.script = [[enc("A")]]
    gen.hf_backend("some/model")(["p"])
    assert world.model.generate_calls[0]["max_new_tokens"] == 256


# --- outputs ------------------------------------------------------------------------------------------------


def test_prompt_tokens_are_stripped_and_one_genout_comes_back_per_prompt_in_order(world):
    world.model.script = [[enc("one") + [EOS], enc("two!") + [EOS], enc("three") + [EOS]]]
    outs = gen.hf_backend("some/model")(["a", "a longer prompt here", "mid length"])
    assert outs == [GenOut("one", "stop"), GenOut("two!", "stop"), GenOut("three", "stop")]
    # the decoded text holds only the new tokens, never the templated prompt
    assert all("[user]" not in o.text and "[assistant]" not in o.text for o in outs)


def test_special_tokens_are_skipped_when_decoding(world):
    world.model.script = [[enc("hi") + [END_RESPONSE] + enc("!") + [EOS, PAD, PAD]]]
    outs = gen.hf_backend("some/model")(["p"])
    assert outs[0].text == "hi!"
    assert all(call["skip_special_tokens"] is True for call in world.tokenizer.decode_calls)


def test_finish_reason_stop_versus_length(world):
    limit = 4
    world.model.script = [[
        enc("ab") + [EOS],  # ended on its own, early
        enc("abcd"),  # no EOS and exactly max_new_tokens new tokens: cut off
        enc("abc") + [EOS],  # EOS in the very last slot still counts as a normal stop
    ]]
    outs = gen.hf_backend("some/model", max_new_tokens=limit)(["x", "y", "z"])
    assert [o.finish_reason for o in outs] == ["stop", "length", "stop"]
    assert [o.text for o in outs] == ["ab", "abcd", "abc"]


def test_all_rows_stopping_early_is_stop_not_length(world):
    world.model.script = [[enc("a") + [EOS], enc("bc") + [EOS]]]
    outs = gen.hf_backend("some/model", max_new_tokens=50)(["x", "y"])
    assert [o.finish_reason for o in outs] == ["stop", "stop"]


def test_an_immediate_eos_gives_empty_text_with_stop(world):
    world.model.script = [[[EOS], enc("fine") + [EOS]]]
    outs = gen.hf_backend("some/model")(["x", "y"])
    assert outs[0] == GenOut("", "stop")  # kept as an empty completion (downstream: no_response)


def test_only_the_stop_and_length_reasons_are_ever_produced(world):
    world.model.script = [[enc("abcd"), enc("a") + [EOS]]]
    outs = gen.hf_backend("some/model", max_new_tokens=4)(["x", "y"])
    assert {o.finish_reason for o in outs} <= gen.FINISH_REASONS


def test_the_generation_config_eos_ids_count_as_stops_too(monkeypatch):
    END_OF_TURN = 3
    world = World(eos_token_id=[END_OF_TURN, EOS]).install(monkeypatch)
    world.model.script = [[enc("ab") + [END_OF_TURN], enc("abcd")]]
    outs = gen.hf_backend("some/model", max_new_tokens=4)(["x", "y"])
    assert [o.finish_reason for o in outs] == ["stop", "length"]
    assert set(world.model.generate_calls[0]["eos_token_id"]) == {END_OF_TURN, EOS}


def test_an_empty_prompt_list_returns_an_empty_list_without_calling_the_model(world):
    assert gen.hf_backend("some/model")([]) == []
    assert world.model.generate_calls == []


def test_the_model_is_loaded_once_not_per_batch(world):
    world.model.script = [[enc("a") + [EOS]], [enc("b") + [EOS]]]
    backend = gen.hf_backend("some/model")
    backend(["p"])
    backend(["q"])
    assert len(world.model_loads) == len(world.tokenizer_loads) == 1
    assert len(world.model.generate_calls) == 2


def test_backend_plugs_into_generate_end_to_end(world, tmp_path):
    from msd.schema import Generation, Prompt, read_jsonl

    prompts = [Prompt(f"p{i}", "en", "harmful", f"q{i}", "dev", "s", "l") for i in range(3)]
    world.model.script = [[enc("x") + [EOS], enc("yyy")], [enc("z") + [EOS]]]
    out = tmp_path / "g.jsonl"
    n = gen.generate(prompts, gen.hf_backend("some/model", max_new_tokens=3), out, arm="A0", alpha=None, batch_size=2)
    assert n == 3
    assert [(r.text, r.finish_reason) for r in read_jsonl(out, Generation)] == [
        ("x", "stop"), ("yyy", "length"), ("z", "stop")]


# --- loading: dtype, pad token, device, secrets ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["float16", "bfloat16", "float32"])
def test_dtype_names_map_to_the_torch_dtype(world, name):
    gen.hf_backend("some/model", dtype=name)
    _, kwargs = world.model_loads[0]
    assert kwargs["dtype"] == f"torch.{name}"


def test_default_dtype_is_float16(world):
    gen.hf_backend("some/model")
    assert world.model_loads[0][1]["dtype"] == "torch.float16"  # a T4 has no native bf16


@pytest.mark.parametrize("bad", ["fp16", "half", "float64", "FLOAT16", "", "auto", None])
def test_unknown_dtype_is_refused_before_anything_is_imported_or_loaded(monkeypatch, bad):
    monkeypatch.setitem(sys.modules, "torch", None)  # would raise ImportError if touched first
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ValueError, match="dtype"):
        gen.hf_backend("some/model", dtype=bad)


def test_the_dtype_keyword_follows_the_transformers_version(monkeypatch):
    old = World(transformers_version="4.45.0").install(monkeypatch)
    gen.hf_backend("some/model")
    assert old.model_loads[0][1].get("torch_dtype") == "torch.float16" and "dtype" not in old.model_loads[0][1]
    new = World(transformers_version="5.22.1").install(monkeypatch)
    gen.hf_backend("some/model")
    assert new.model_loads[0][1].get("dtype") == "torch.float16" and "torch_dtype" not in new.model_loads[0][1]
    odd = World(transformers_version="weird").install(monkeypatch)  # an unparsable version: use the new name
    gen.hf_backend("some/model")
    assert odd.model_loads[0][1].get("dtype") == "torch.float16"


def test_missing_pad_token_falls_back_to_eos(monkeypatch):
    world = World(with_pad=False).install(monkeypatch)
    world.model.script = [[enc("a") + [EOS], enc("b") + [EOS]]]
    outs = gen.hf_backend("some/model")(["x", "longer prompt"])  # the fake tokenizer raises if it cannot pad
    assert world.tokenizer.pad_token == world.tokenizer.eos_token == "<eos>"
    assert world.model.generate_calls[0]["pad_token_id"] == EOS
    assert [o.text for o in outs] == ["a", "b"]  # trailing pads (== EOS) do not leak into the text


def test_an_existing_pad_token_is_left_alone(world):
    gen.hf_backend("some/model")
    assert world.tokenizer.pad_token == "<pad>"


@pytest.mark.parametrize("cuda, device", [(True, "cuda"), (False, "cpu")])
def test_device_is_cuda_when_available_else_cpu(monkeypatch, cuda, device):
    world = World(cuda=cuda).install(monkeypatch)
    world.model.script = [[enc("a") + [EOS]]]
    gen.hf_backend("some/model")(["p"])
    assert world.model.device == device
    assert world.model.generate_calls[0]["input_ids"].device == device
    assert world.model.generate_calls[0]["attention_mask"].device == device
    assert world.model.evaluated is True


@pytest.mark.parametrize("path", ["CohereLabs/tiny-aya-global", "/kaggle/working/merged/A2-0.25"])
def test_repo_ids_and_local_directories_are_passed_through_unchanged(world, path):
    gen.hf_backend(path)
    assert world.tokenizer_loads[0][0] == path and world.model_loads[0][0] == path


def test_no_token_and_no_remote_code_are_passed_and_nothing_is_printed(world, monkeypatch, capsys):
    token = "hf_" + "Q9w8E7r6T5" * 4
    monkeypatch.setenv("HF_TOKEN", token)  # transformers reads it from the environment itself
    world.model.script = [[enc("a") + [EOS]]]
    gen.hf_backend("some/model")(["p"])
    for _, kwargs in (*world.tokenizer_loads, *world.model_loads):
        assert not {"token", "use_auth_token", "trust_remote_code"} & set(kwargs)
        assert token not in repr(kwargs)
    shown = capsys.readouterr()
    assert shown.out == "" and shown.err == "" and token not in shown.out + shown.err


@pytest.mark.parametrize("bad", [0, -5, 1.5, True, None])
def test_bad_max_new_tokens_is_refused(world, bad):
    with pytest.raises(ValueError, match="max_new_tokens"):
        gen.hf_backend("some/model", max_new_tokens=bad)


def test_missing_libraries_give_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="torch.*transformers|transformers.*torch"):
        gen.hf_backend("some/model")
