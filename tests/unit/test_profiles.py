"""Unit tests for desk/profiles.py: the pre-registered profile set and its leave-one-out structure."""

from __future__ import annotations

import dataclasses

import pytest

from desk.profiles import (
    CACHE_RAM_BOUNDED_MIB,
    CACHE_RAM_ENGINE_DEFAULT_MIB,
    MODEL_SHA256,
    PROFILES,
    Q4_FILE,
    Q8_FILE,
    Profile,
    get_profile,
    pinned_engine_flags,
)

SWITCHES = ("model_file", "protocol", "prefetch", "structured_report", "cache_ram_mib")


def differences(a: Profile, b: Profile) -> set[str]:
    return {f for f in SWITCHES if getattr(a, f) != getattr(b, f)}


def test_profile_names_are_the_pre_registered_set():
    assert list(PROFILES) == ["B", "F", "F-q8", "F-native", "F-noprefetch", "F-freetext",
                              "F-cacheram-default", "B-q4", "B-repeat"]


def test_baseline_and_final_settings():
    b, f = get_profile("B"), get_profile("F")
    assert (b.model_file, b.quantization, b.protocol, b.prefetch, b.structured_report, b.cache_ram_mib) == (
        Q8_FILE, "Q8_0", "native", False, False, 8192)
    assert (f.model_file, f.quantization, f.protocol, f.prefetch, f.structured_report, f.cache_ram_mib) == (
        Q4_FILE, "Q4_K_M", "constrained", True, True, CACHE_RAM_BOUNDED_MIB)
    assert CACHE_RAM_BOUNDED_MIB < CACHE_RAM_ENGINE_DEFAULT_MIB == 8192
    assert differences(b, f) == set(SWITCHES)


@pytest.mark.parametrize("name, changed", [
    ("F-q8", "model_file"), ("F-native", "protocol"), ("F-noprefetch", "prefetch"),
    ("F-freetext", "structured_report"), ("F-cacheram-default", "cache_ram_mib"),
])
def test_leave_one_out_reverts_exactly_one_switch_to_baseline(name, changed):
    variant, final, base = get_profile(name), get_profile("F"), get_profile("B")
    assert differences(variant, final) == {changed}
    assert getattr(variant, changed) == getattr(base, changed)


def test_same_quant_and_repeat_baselines():
    b = get_profile("B")
    assert differences(get_profile("B-q4"), b) == {"model_file"} and get_profile("B-q4").quantization == "Q4_K_M"
    repeat = get_profile("B-repeat")
    assert dataclasses.replace(repeat, name="B", description=b.description) == b


@pytest.mark.parametrize("profile", list(PROFILES.values()), ids=list(PROFILES))
def test_every_profile_pins_every_engine_flag(profile):
    flags = profile.engine_flags
    pairs = dict(zip(flags, flags[1:]))  # flag -> the token after it
    for flag, value in [("-ngl", "99"), ("-fa", "on"), ("-c", "8192"), ("-np", "1"), ("--reasoning", "off"),
                        ("--cache-ram", str(profile.cache_ram_mib)), ("--ctx-checkpoints", str(profile.ctx_checkpoints)),
                        ("-t", "5"), ("--host", "127.0.0.1"), ("--seed", "42")]:
        assert pairs[flag] == value, flag
    for switch in ("--jinja", "--metrics", "--no-webui"):
        assert switch in flags
    assert profile.model_sha256 == MODEL_SHA256[profile.model_file] and len(profile.model_sha256) == 64
    assert profile.description


def test_engine_argv_app_switches_and_caches():
    f = get_profile("F")
    argv = f.engine_argv("/opt/homebrew/bin/llama-server", "/m/model.gguf", 8401)
    assert argv[:5] == ["/opt/homebrew/bin/llama-server", "-m", "/m/model.gguf", "--port", "8401"]
    assert tuple(argv[5:]) == f.engine_flags
    assert f.app_switches() == {"protocol": "constrained", "prefetch_account_card": True,
                                "structured_ops_report": True}
    assert f.caches() == {"cache_prompt": True, "cache_ram_mib": CACHE_RAM_BOUNDED_MIB, "ctx_checkpoints": 32,
                          "persistence": "within a run across cases; cleared by server restart between runs"}


def test_engine_argv_with_more_slots_keeps_each_slot_at_the_pinned_context():
    f = get_profile("F")
    argv = f.engine_argv("/bin/llama-server", "/m/model.gguf", 8401, slots=4)
    assert argv[argv.index("-np") + 1] == "4" and argv[argv.index("-c") + 1] == str(4 * 8192)
    rest = [a for i, a in enumerate(argv[5:]) if a not in ("-np", "-c") and argv[5:][i - 1] not in ("-np", "-c")]
    pinned = [a for i, a in enumerate(f.engine_flags) if a not in ("-np", "-c") and f.engine_flags[i - 1] not in ("-np", "-c")]
    assert rest == pinned  # nothing else changes
    assert f.engine_argv("/bin/llama-server", "/m/model.gguf", 8401, slots=1)[5:] == list(f.engine_flags)


def test_aliases_and_unknown_names():
    assert get_profile("final") is get_profile("F")
    assert get_profile("baseline") is get_profile("B")
    with pytest.raises(KeyError, match="unknown profile 'f'.*known: B, F"):
        get_profile("f")


@pytest.mark.parametrize("changes, message", [
    ({"protocol": "xml"}, "unknown protocol"),
    ({"quantization": "Q8_0"}, "is not Q8_0"),
    ({"cache_ram_mib": 4096}, "engine_flags disagree"),
    ({"engine_flags": pinned_engine_flags(1024, 8)}, "engine_flags disagree"),
])
def test_profile_validation(changes, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(get_profile("F"), **changes)
