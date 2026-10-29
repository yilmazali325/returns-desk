"""Named run profiles: the engine flags and app switches for each measured configuration.

A profile is everything that may differ between benchmark runs, written down in one place so that
`configuration` in results.json, the llama-server argv and the orchestrator switches cannot drift apart.

Why every engine flag is pinned explicitly (SPEC §9): llama-server defaults are not stable across
builds and some depend on the machine (`-c 0` loads the model's 262k context, `-ngl`/`-fa` are "auto",
`--fit` resizes unset values). A profile that relies on a default silently changes when the engine or
the hardware changes, so each profile spells out the full flag set, including values that equal today's
default (`--cache-ram 8192`, `--ctx-checkpoints 32`, `-t 5`).

Profiles (leave-one-out design around F, SPEC §9):
  B                   baseline: Q8_0, native tool calls, no prefetch, free-text ops summary, cache-ram 8192
  F                   final candidate: Q4_K_M, constrained JSON, prefetch, structured report, cache-ram bounded
  F-q8, F-native, F-noprefetch, F-freetext, F-cacheram-default   F with exactly one change reverted to B
  B-q4                B with Q4_K_M (same-quantization comparison of the engineering changes)
  B-repeat            identical to B under another name (A/A noise floor)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

MODEL_REPO = "unsloth/Qwen3.5-4B-GGUF"
MODEL_REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
MODEL_NAME = "Qwen3.5-4B"
Q4_FILE = "Qwen3.5-4B-Q4_K_M.gguf"
Q8_FILE = "Qwen3.5-4B-Q8_0.gguf"
MODEL_SHA256 = {
    Q4_FILE: "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
    Q8_FILE: "10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1",
}
QUANTIZATION = {Q4_FILE: "Q4_K_M", Q8_FILE: "Q8_0"}

ENGINE = "llama.cpp llama-server"
ENGINE_VERSION = "0.7.0 (build 11669, commit 620d2225c)"

PROTOCOLS = ("native", "constrained")

# Shared engine settings. Context is per slot (one slot), so every profile gets the same 8192 tokens.
CTX_SIZE = 8192
SLOTS = 1
# Threads only drive the CPU side (sampling, grammar); all layers run on the GPU (-ngl 99). 5 equals
# the number of "Super" cores on the M5 Pro and llama.cpp's own auto choice on this machine.
THREADS = 5
SEED = 42

# Host prompt cache (llama-server --cache-ram). It saves the idle role's slot state (e.g. Resolution
# while Operations runs) and restores it, which is why one slot serves both agents without re-prefill.
# Measured (docs/notes/llm_infra.md, -lv 4 logs): an entry = prompt state (68-80 MiB) + the slot's
# context checkpoints (50.25 MiB each; Qwen3.5 is a hybrid recurrent model), 228-371 MiB in the agent
# workload, and the cache never held more than 1 entry because a restored entry leaves the cache.
# The engine default (8192 MiB) only matters for prompts that are never restored; the bound caps that
# worst case while still fitting one worst-case entry: 32 checkpoints x 50.25 + 80 = 1688 MiB < 2048.
# Measured cost of the bound in the 8-case workload: none (same cached tokens, same latency).
CACHE_RAM_ENGINE_DEFAULT_MIB = 8192
CACHE_RAM_BOUNDED_MIB = 2048
# Engine default (32 per slot), pinned so a future default change cannot alter cache behaviour.
CTX_CHECKPOINTS = 32


def pinned_engine_flags(cache_ram_mib: int, ctx_checkpoints: int) -> tuple[str, ...]:
    """The full llama-server flag set shared by all profiles (model path and port are added at launch)."""
    return (
        "-ngl", "99",
        "-fa", "on",
        "-c", str(CTX_SIZE),
        "-np", str(SLOTS),
        "--jinja",
        "--reasoning", "off",
        "--cache-ram", str(cache_ram_mib),
        "--ctx-checkpoints", str(ctx_checkpoints),
        "-t", str(THREADS),
        "--host", "127.0.0.1",
        "--metrics",
        "--no-webui",
        "--seed", str(SEED),
    )


@dataclass(frozen=True)
class Profile:
    """One measured configuration: model + engine flags + application switches."""

    name: str
    model_file: str
    quantization: str
    protocol: str  # "native" | "constrained"
    engine_flags: tuple[str, ...]
    prefetch: bool  # orchestrator injects the account card instead of Resolution calling find_order_lines
    structured_report: bool  # Ops reports only {outcome}; code builds the report (else free-text summary)
    cache_ram_mib: int
    ctx_checkpoints: int
    description: str

    def __post_init__(self) -> None:
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"profile {self.name}: unknown protocol {self.protocol!r}")
        if QUANTIZATION.get(self.model_file) != self.quantization:
            raise ValueError(f"profile {self.name}: {self.model_file} is not {self.quantization}")
        # The cache values are stored twice (fields for reporting, flags for the engine): keep them equal.
        if self.engine_flags != pinned_engine_flags(self.cache_ram_mib, self.ctx_checkpoints):
            raise ValueError(f"profile {self.name}: engine_flags disagree with cache_ram/ctx_checkpoints")

    @property
    def model_sha256(self) -> str:
        return MODEL_SHA256[self.model_file]

    def engine_argv(self, binary: str, model_path: str, port: int) -> list[str]:
        """Exact argv used to launch llama-server for this profile."""
        return [binary, "-m", model_path, "--port", str(port), *self.engine_flags]

    def app_switches(self) -> dict[str, object]:
        """The application-level switches, as reported under configuration.app_switches."""
        return {
            "protocol": self.protocol,
            "prefetch_account_card": self.prefetch,
            "structured_ops_report": self.structured_report,
        }

    def caches(self) -> dict[str, object]:
        """Cache declaration for configuration.caches (SPEC §8)."""
        return {
            "cache_prompt": True,
            "cache_ram_mib": self.cache_ram_mib,
            "ctx_checkpoints": self.ctx_checkpoints,
            "persistence": "within a run across cases; cleared by server restart between runs",
        }


def _profile(name: str, model_file: str, protocol: str, prefetch: bool, structured_report: bool,
             cache_ram_mib: int, description: str) -> Profile:
    return Profile(
        name=name,
        model_file=model_file,
        quantization=QUANTIZATION[model_file],
        protocol=protocol,
        engine_flags=pinned_engine_flags(cache_ram_mib, CTX_CHECKPOINTS),
        prefetch=prefetch,
        structured_report=structured_report,
        cache_ram_mib=cache_ram_mib,
        ctx_checkpoints=CTX_CHECKPOINTS,
        description=description,
    )


BASELINE = _profile(
    "B", Q8_FILE, "native", prefetch=False, structured_report=False,
    cache_ram_mib=CACHE_RAM_ENGINE_DEFAULT_MIB,
    description="Baseline: Q8_0, native tool calls, Resolution looks lines up itself, free-text ops "
                "summary, engine-default host prompt cache.",
)
FINAL = _profile(
    "F", Q4_FILE, "constrained", prefetch=True, structured_report=True,
    cache_ram_mib=CACHE_RAM_BOUNDED_MIB,
    description="Final candidate: Q4_K_M, schema-constrained JSON steps, prefetched account card, "
                "code-built ops report, bounded host prompt cache.",
)


def variant(base: Profile, name: str, description: str, **changes: object) -> Profile:
    """Copy `base` with some fields changed, keeping engine_flags consistent with the cache fields."""
    cache_ram = int(changes.get("cache_ram_mib", base.cache_ram_mib))
    model_file = str(changes.get("model_file", base.model_file))
    return replace(
        base, name=name, description=description,
        quantization=QUANTIZATION[model_file],
        engine_flags=pinned_engine_flags(cache_ram, base.ctx_checkpoints),
        **changes,
    )


PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (
        BASELINE,
        FINAL,
        variant(FINAL, "F-q8", "F with B's quantization (Q8_0).", model_file=Q8_FILE),
        variant(FINAL, "F-native", "F with B's protocol (native tool calls).", protocol="native"),
        variant(FINAL, "F-noprefetch", "F without the prefetched account card.", prefetch=False),
        variant(FINAL, "F-freetext", "F with B's free-text ops summary.", structured_report=False),
        variant(FINAL, "F-cacheram-default", "F with the engine-default host prompt cache (8192 MiB).",
                 cache_ram_mib=CACHE_RAM_ENGINE_DEFAULT_MIB),
        variant(BASELINE, "B-q4", "B with Q4_K_M: same-quantization comparison.", model_file=Q4_FILE),
        variant(BASELINE, "B-repeat", "Identical to B; A/A noise floor."),
    )
}

ALIASES = {"final": "F", "baseline": "B"}


def get_profile(name: str) -> Profile:
    """Look a profile up by name (case-sensitive) or alias ("final", "baseline")."""
    key = ALIASES.get(name, name)
    if key not in PROFILES:
        known = ", ".join([*PROFILES, *ALIASES])
        raise KeyError(f"unknown profile {name!r}; known: {known}")
    return PROFILES[key]
