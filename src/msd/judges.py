"""Concrete judges: Claude through the Anthropic API, and an open model run locally with `transformers`.

Both implement the `msd.judge.Judge` protocol (an `id` and `classify(kind, prompt, response) -> raw text`).
`msd.judge.JudgeRunner` wraps them: it parses the raw text strictly, retries, caches parsed labels and
records the judge id in every verdict.

    anthropic:<model-id>   AnthropicJudge: needs ANTHROPIC_API_KEY (environment, else the repository `.env`)
    hf:<model-id>          HFLocalJudge: a Hugging Face instruct model, greedy decoding; needs a GPU runtime

Importing this module imports neither `anthropic` nor `torch` nor `transformers`; each is imported on first
use. `is_known_judge_name` checks a name without reading the environment, so the command line can validate
`--judge` at parse time.

Judge ids embed a fingerprint of the judge templates (`msd.judge.template_fingerprint`). The verdict cache is
keyed by judge id, kind, prompt and reply but not by template text, so an edited template must change the id,
or old labels would be reused for the new wording.

Secrets: the API key is read once, handed to the Anthropic client and kept nowhere else (not on the judge, not
in its repr, not in any error text). Errors raised while judging are typed, carry fixed text only, and never
quote the request, the reply or anything the API sent back.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from msd.judge import JudgeSetupError, render_judge_prompt, template_fingerprint

REQUEST_TIMEOUT_SECONDS = 60.0  # the SDK default is ten minutes, which would hang a run on one bad connection
ANTHROPIC_MAX_TOKENS = 512  # the answer is one word; the room is for models that think before answering (see below)
HF_MAX_NEW_TOKENS = 8  # "complied" / "refused" take two or three tokens

API_KEY_VARIABLE = "ANTHROPIC_API_KEY"

ACCEPTED_FORMS = (
    "anthropic:<model-id> (Claude through the API; needs ANTHROPIC_API_KEY) or "
    "hf:<model-id> (a Hugging Face model run locally on a GPU runtime)"
)

_ANTHROPIC_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_HF_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?")  # repo name or org/repo
_PLAUSIBLE_KEY = re.compile(r"[\x21-\x7e]+")  # printable ASCII, no spaces: a key that is anything else is a paste error


# A status that means every call will fail the same way (so the run stops) rather than this one item.
_FATAL_STATUS = {401: "the API key was rejected", 403: "the API key is not allowed to use this model",
                 404: "the model id was not found"}


class JudgeCallError(RuntimeError):
    """One judge call produced no usable answer. The text is fixed; it never quotes the request or the reply."""


class JudgeRefusalError(JudgeCallError):
    """The API stopped with a refusal instead of an answer."""


class JudgeNoTextError(JudgeCallError):
    """The API answered but the reply holds no text (for example the token cap ran out before the answer)."""


# --- names ----------------------------------------------------------------------------------------------------


def is_known_judge_name(name: Any) -> bool:
    """True for `anthropic:<model-id>` and `hf:<model-id>` with a well-formed model id. Pure: no environment
    read, no package import, no file access."""
    if not isinstance(name, str):
        return False
    scheme, separator, model = name.partition(":")
    if not separator:
        return False
    if scheme == "anthropic":
        return _ANTHROPIC_MODEL.fullmatch(model) is not None
    if scheme == "hf":
        return _HF_MODEL.fullmatch(model) is not None
    return False


def make_judge(name: str, *, env_file: str | Path | None = None, environ: Mapping[str, str] | None = None):
    """Build the judge called `name`; ValueError (listing the accepted forms) for any other name.

    `env_file` and `environ` only matter to `anthropic:` judges: where to look for the key after / instead of
    the process environment.
    """
    if not is_known_judge_name(name):
        raise ValueError(f"unknown judge {name!r}; accepted forms: {ACCEPTED_FORMS}")
    scheme, _, model = name.partition(":")
    if scheme == "anthropic":
        return AnthropicJudge(model, env_file=env_file, environ=environ)
    return HFLocalJudge(model)


# --- Claude through the API -----------------------------------------------------------------------------------


def _find_api_key(environ: Mapping[str, str] | None, env_file: str | Path | None) -> str:
    """The key from the environment, else from `env_file` (python-dotenv; nothing is exported to the process).

    A blank value counts as not set. Raises JudgeSetupError; the text never contains the key.
    """
    env = os.environ if environ is None else environ
    value = env.get(API_KEY_VARIABLE)
    key = value.strip() if isinstance(value, str) else ""
    if not key and env_file is not None:
        try:
            if Path(env_file).is_file():
                from dotenv import dotenv_values

                from_file = dotenv_values(env_file).get(API_KEY_VARIABLE)
                key = from_file.strip() if isinstance(from_file, str) else ""
        except Exception:
            key = ""  # an unreadable .env is the same as no .env
    if not key:
        raise JudgeSetupError(
            f"{API_KEY_VARIABLE} is not set (an empty value counts as not set). Export it in the shell or put the "
            "line ANTHROPIC_API_KEY=... in the .env file at the repository root; the judge cannot run without it"
        )
    if not _PLAUSIBLE_KEY.fullmatch(key):
        raise JudgeSetupError(
            f"{API_KEY_VARIABLE} is set but is not a usable key (it contains spaces or non-printable characters); "
            "check it for a paste error"
        )
    return key


class AnthropicJudge:
    """Judge with a Claude model through the Messages API.

    One request per call: a single user message holding the rendered, fenced judge prompt, no system prompt.
    Counters for the cost estimate, all plain attributes: `calls` (requests that got a response, refusals
    included, because they are billed), `input_tokens` and `output_tokens` (from the responses' usage).

    The reply is the text blocks of the response. A response that stops with a refusal raises
    `JudgeRefusalError`; one without any text raises `JudgeNoTextError`. Both are `JudgeCallError`s, so
    `JudgeRunner` counts them as failed attempts and the item ends up as `judge_error` after its retries. No
    text is ever invented. API errors (rate limit, connection, 5xx) propagate as the SDK's own exceptions, except
    a rejected key, a forbidden model or an unknown model id (HTTP 401, 403, 404): those raise `JudgeSetupError`,
    which stops the run, because every item would fail the same way.

    `temperature` is None by default and then NOT sent: the current Claude models reject any non-default
    sampling parameter and every model rejects them on API version 2026-10-21 (per the Claude API migration
    notes of 2026-10), so a fixed `temperature=0` would fail every call. Repeatability comes from the label
    cache instead. Pass `temperature` only for an older model that accepts it; it then goes into the judge id,
    because it can change labels.

    `max_tokens` bounds the whole response. The answer is one word, but a model that thinks before answering
    spends thinking tokens from the same budget, and a response cut off before any text arrives raises
    `JudgeNoTextError` (naming the stop reason). If the live smoke test shows that, raise it.

    `client` is for tests: any object with `messages.create(**request)`. Without one the `anthropic` package is
    imported and a client built from the key (`environ` / `env_file` say where to look for it). Both checks
    happen here, in the constructor, so a run without a key or without the package fails before any work.
    """

    def __init__(
        self,
        model: str,
        client: Any = None,
        *,
        temperature: float | None = None,
        max_tokens: int = ANTHROPIC_MAX_TOKENS,
        env_file: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(model, str) or not _ANTHROPIC_MODEL.fullmatch(model):
            raise ValueError(f"model must be a model id such as 'claude-haiku-5-5', got {model!r}")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(f"max_tokens must be a positive integer, got {max_tokens!r}")
        if temperature is not None and (
            isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 1
        ):
            raise ValueError(f"temperature must be None or a number from 0 to 1, got {temperature!r}")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.id = f"anthropic:{model}:t{template_fingerprint()}"
        if temperature is not None:
            self.id += f":temp{temperature:g}"
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._client = client if client is not None else self._build_client(environ, env_file)

    @staticmethod
    def _build_client(environ: Mapping[str, str] | None, env_file: str | Path | None) -> Any:
        key = _find_api_key(environ, env_file)
        try:
            import anthropic
        except ImportError:
            raise JudgeSetupError(
                "the 'anthropic' package is not installed; install it with: pip install anthropic "
                "(or pip install -e '.[judge]')"
            ) from None
        return anthropic.Anthropic(api_key=key, timeout=REQUEST_TIMEOUT_SECONDS)

    def __repr__(self) -> str:
        return f"AnthropicJudge(model={self.model!r}, id={self.id!r})"

    def classify(self, kind: str, prompt: str, response: str) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": render_judge_prompt(kind, prompt, response)}],
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        try:
            message = self._client.messages.create(**request)
        except Exception as exc:
            status = getattr(exc, "status_code", None)  # the SDK's APIStatusError carries it
            if isinstance(status, int) and not isinstance(status, bool) and status in _FATAL_STATUS:
                raise JudgeSetupError(
                    f"{_FATAL_STATUS[status]} (HTTP {status}); check ANTHROPIC_API_KEY and the model id "
                    f"{self.model!r}"
                ) from None
            raise  # rate limits, timeouts, 5xx and the rest: a failed attempt, retried by the runner

        usage = getattr(message, "usage", None)
        self.calls += 1
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

        stop_reason = getattr(message, "stop_reason", None)
        if stop_reason == "refusal":
            raise JudgeRefusalError("the API stopped with a refusal instead of an answer")
        text = "".join(
            block.text
            for block in (getattr(message, "content", None) or [])
            if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)
        )
        if not text.strip():
            reason = re.sub(r"[^a-z_]", "?", str(stop_reason))[:32]
            raise JudgeNoTextError(f"the API reply held no text (stop_reason={reason})")
        return text


# --- an open model, run locally -------------------------------------------------------------------------------


class HFLocalJudge:
    """Judge with a Hugging Face instruct model: greedy decoding, `HF_MAX_NEW_TOKENS` new tokens.

    The rendered, fenced judge prompt goes in as one user turn of the model's chat template. Nothing heavy
    happens at construction: `torch` and `transformers` are imported and the model is loaded by `_load()` on
    the first `classify`, once. A model without a chat template (a base model) is refused with a
    `JudgeSetupError`, as is a runtime without `torch`. A gated model takes its token from HF_TOKEN, read by
    `transformers` itself; nothing here reads, prints or stores it.

    `generate_fn(messages) -> str` replaces the whole loaded model (tests inject a fake); it receives the chat
    message list and returns the model's text. Qwen3-style templates think first unless told not to, which
    would eat the 8 tokens, so the template call passes `enable_thinking=False`; templates that do not use that
    keyword ignore it.
    """

    def __init__(
        self,
        model_id: str,
        generate_fn: Callable[[list[dict[str, str]]], str] | None = None,
        *,
        max_new_tokens: int = HF_MAX_NEW_TOKENS,
        dtype: str = "float16",
    ) -> None:
        from msd.gen import _TORCH_DTYPES  # the dtype names the generation backend accepts

        if not isinstance(model_id, str) or not _HF_MODEL.fullmatch(model_id):
            raise ValueError(f"model_id must be a Hugging Face model id such as 'org/name', got {model_id!r}")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be a positive integer, got {max_new_tokens!r}")
        if dtype not in _TORCH_DTYPES:
            raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(_TORCH_DTYPES)}")
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.id = f"hf:{model_id}:t{template_fingerprint()}"
        self._generate_fn = generate_fn

    def __repr__(self) -> str:
        return f"HFLocalJudge(model_id={self.model_id!r}, id={self.id!r})"

    def _load(self) -> Callable[[list[dict[str, str]]], str]:
        """Import torch and transformers, load the model, return its `generate_fn`. Slow; runs once."""
        from msd.gen import _TORCH_DTYPES, _dtype_keyword

        try:
            import torch
            import transformers
        except ImportError:
            raise JudgeSetupError(
                "HFLocalJudge needs the 'torch' and 'transformers' packages, which are not installed here; run the "
                "judge stage on the GPU runtime (Kaggle or Colab, see docs/runtime.md)"
            ) from None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        if not getattr(tokenizer, "chat_template", None):
            raise JudgeSetupError(
                f"{self.model_id} has no chat template, so it is a base model; a judge needs an instruct model"
            )
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id, **{_dtype_keyword(transformers): getattr(torch, _TORCH_DTYPES[self.dtype])}
        )
        model = model.to(device)
        model.eval()
        max_new_tokens = self.max_new_tokens

        def generate(messages: list[dict[str, str]]) -> str:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            batch = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)  # the template has the BOS
            input_ids = batch["input_ids"]
            with torch.no_grad():
                sequences = model.generate(
                    input_ids=input_ids,
                    attention_mask=batch["attention_mask"],
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_id,
                )
            new_tokens = sequences[0, input_ids.shape[1]:]
            return tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True).strip()

        print(f"msd.judges: loaded {self.model_id} on {device}", file=sys.stderr)
        return generate

    def classify(self, kind: str, prompt: str, response: str) -> str:
        messages = [{"role": "user", "content": render_judge_prompt(kind, prompt, response)}]
        if self._generate_fn is None:
            self._generate_fn = self._load()
        return self._generate_fn(messages)
