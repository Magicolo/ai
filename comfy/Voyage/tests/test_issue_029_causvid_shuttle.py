"""Issue 029: CausVid encodes the whole segment on a single T5 shuttle.

`generate_blocks` mapped `_encode_conditional` over the prompts while each
call did its own `to("cuda") / to("cpu") + empty_cache()` cycle — N
alloc/free roundtrips of ~11 GiB per segment, the exact fragmentation
pattern the code comment warns about. `_encode_conditionals` moves the
model once, encodes every prompt, and parks it once.

CPU-only: the session is built without `__init__` (CUDA) with a fake
torch/pipeline recording every shuttle move.
"""

from __future__ import annotations

from typing import Any

import pytest

from voyage.workers.video_causvid import CausvidSession, _stub_encoder_class, _stub_encoder_classes


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()


class _FakeTextEncoder:
    """T5 stub recording every device shuttle and every encode call."""

    def __init__(self) -> None:
        self.devices: list[str] = []
        self.prompts: list[str] = []

    def to(self, device: str) -> _FakeTextEncoder:
        self.devices.append(device)
        return self

    def __call__(self, prompts: list[str]) -> dict[str, Any]:
        self.prompts.extend(prompts)
        return {f"conditional-for-{prompts[0]}": True}


class _FakePipeline:
    def __init__(self) -> None:
        self.text_encoder = _FakeTextEncoder()


class _ModuleBase:
    """Minimal `nn.Module` surface: call dispatches to `forward`."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise NotImplementedError


def _encode_session() -> tuple[CausvidSession, _FakeTorch, _FakeTextEncoder]:
    torch = _FakeTorch()
    pipeline = _FakePipeline()
    session = CausvidSession.__new__(CausvidSession)
    session._torch = torch  # type: ignore[attr-defined]
    session._pipeline = pipeline  # type: ignore[attr-defined]
    session._device = "cuda:0"  # type: ignore[attr-defined]
    return session, torch, pipeline.text_encoder


def test_segment_encodes_on_a_single_shuttle() -> None:
    session, torch, encoder = _encode_session()
    prompts = ["lantern valley", "paper river", "brass mountain"]
    conditionals = session._encode_conditionals(prompts)
    assert len(conditionals) == len(prompts)
    assert encoder.prompts == prompts
    # One CUDA roundtrip total: up once, encode all, park once.
    assert encoder.devices == ["cuda", "cpu"]
    assert torch.cuda.empty_cache_calls == 2


def test_single_prompt_uses_a_single_shuttle() -> None:
    session, torch, encoder = _encode_session()
    conditional = session._encode_conditional("lantern valley")
    assert conditional == {"conditional-for-lantern valley": True}
    assert encoder.devices == ["cuda", "cpu"]
    assert torch.cuda.empty_cache_calls == 2


def test_shuttle_restores_cpu_encoder_on_encode_failure() -> None:
    session, _torch, _encoder = _encode_session()

    class _FailingEncoder(_FakeTextEncoder):
        def __call__(self, prompts: list[str]) -> dict[str, Any]:
            del prompts
            raise RuntimeError("synthetic T5 failure")

    failing = _FailingEncoder()
    session._pipeline.text_encoder = failing  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="synthetic T5 failure"):
        session._encode_conditionals(["lantern valley"])
    assert failing.devices == ["cuda", "cpu"]


def test_empty_prompt_list_is_rejected() -> None:
    session, _torch, _encoder = _encode_session()
    with pytest.raises(ValueError, match="at least one prompt"):
        session._encode_conditionals([])


def test_stub_encoder_class_is_cached_per_module_base() -> None:
    _stub_encoder_classes.clear()
    first = _stub_encoder_class(_ModuleBase)
    second = _stub_encoder_class(_ModuleBase)
    assert first is second
    other_base = type("OtherBase", (_ModuleBase,), {})
    assert _stub_encoder_class(other_base) is not first


def test_cached_stub_returns_the_precomputed_conditional() -> None:
    _stub_encoder_classes.clear()
    stub_class = _stub_encoder_class(_ModuleBase)
    precomputed = {"prompt_embeds": object()}
    assert stub_class(precomputed)(["any prompt"]) is precomputed
