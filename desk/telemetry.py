"""Memory, hardware and run-condition telemetry for Apple Silicon (macOS).

Why these sources (evidence in docs/notes/spec_critique.json and docs/notes/llm_infra.md):
- RAM: the kernel's phys_footprint, not RSS. On unified memory, RSS of llama-server counts GBs of
  clean mmapped weight pages that are also the GPU-resident memory, so it overcounts and cannot be added
  to a GPU figure. `proc_pid_rusage(pid, RUSAGE_INFO_V4)` exposes `ri_interval_max_phys_footprint`, a
  kernel-maintained high-water mark since the last `proc_reset_footprint_interval(pid)`. That catches
  short allocation spikes that any sampler would miss and works on our own child processes without sudo.
- GPU ("VRAM"): Apple Silicon has no separate VRAM. The AGX IOAccelerator node in the IORegistry
  publishes "In use system memory" (bytes of unified memory the GPU driver has wired). It is
  system-wide, so we subtract an idle baseline taken before llama-server starts and sample it at 5 Hz
  plus at every turn boundary. It overlaps peak_ram_mb; the two must never be added.
- hardware: only static identifiers, so the validator's `--baseline` equality check holds across runs.
  Live readings (power source, Low Power Mode, thermal state, load, competing processes) are taken
  with pmset/ps at run start and end and belong in run_conditions(); hardware.power points there.
- Privacy: other processes are recorded by executable name, pid, %CPU and %MEM only, never by their
  full command line (arguments can hold tokens and private paths).
- Linux/Windows (e.g. an NVIDIA workstation): libproc, ioreg, sysctl and pmset are missing, so footprints
  read None (callers fall back to RSS), GPU busy and dedicated VRAM come from `nvidia-smi`, and hardware
  names from lscpu/nproc/free/uname/nvidia-smi.

All readers return None instead of raising when a source is unavailable (non-macOS, process gone),
so a telemetry gap can never fail a case; the report then says why a value is null.
Units: "mb" values are MiB (2**20 bytes), rounded to 0.1.
"""

from __future__ import annotations

import ctypes
import functools
import os
import re
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import psutil

MIB = 1024 * 1024
RUSAGE_INFO_V4 = 4
LIBPROC_PATH = "/usr/lib/libproc.dylib"
SAMPLE_INTERVAL_S = 0.2  # 5 Hz


# --------------------------------------------------------------------------------------------------
# Per-process footprint via libproc (ctypes)

class _RusageInfoV4(ctypes.Structure):
    """`struct rusage_info_v4` from <sys/resource.h>; field order matters, all fields are uint64."""

    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ri_uuid_0", "ri_uuid_1",  # uint8_t ri_uuid[16]
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
        "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
        "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time",
        "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
        "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten",
        "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background",
        "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy", "ri_cpu_time_qos_user_initiated",
        "ri_cpu_time_qos_user_interactive", "ri_billed_system_time", "ri_serviced_system_time",
        "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
        "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
        "ri_runnable_time",
    )]


@functools.lru_cache(maxsize=1)
def _libproc() -> ctypes.CDLL | None:
    try:
        return ctypes.CDLL(LIBPROC_PATH, use_errno=True)
    except OSError:
        return None


@dataclass(frozen=True)
class Footprint:
    """One kernel reading for one process, in bytes."""

    pid: int
    phys_footprint: int
    interval_max_phys_footprint: int  # high-water since the last reset_footprint_interval()
    lifetime_max_phys_footprint: int
    resident_size: int


def read_footprint(pid: int) -> Footprint | None:
    """Kernel memory counters for `pid`, or None if libproc is missing or the process is gone."""
    lib = _libproc()
    if lib is None:
        return None
    info = _RusageInfoV4()
    if lib.proc_pid_rusage(int(pid), RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None
    return Footprint(
        pid=int(pid),
        phys_footprint=info.ri_phys_footprint,
        interval_max_phys_footprint=info.ri_interval_max_phys_footprint,
        lifetime_max_phys_footprint=info.ri_lifetime_max_phys_footprint,
        resident_size=info.ri_resident_size,
    )


def reset_footprint_interval(pid: int) -> bool:
    """Restart the kernel high-water window for `pid` (done at case start). True on success."""
    lib = _libproc()
    if lib is None:
        return False
    return lib.proc_reset_footprint_interval(int(pid)) == 0


def read_rss(pid: int) -> int | None:
    """Resident set size in bytes (reported for completeness; overcounts mmapped weights)."""
    try:
        return int(psutil.Process(pid).memory_info().rss)
    except (psutil.Error, OSError):
        return None


# --------------------------------------------------------------------------------------------------
# GPU (AGX IOAccelerator) via ioreg

def _run(argv: list[str], timeout_s: float = 5.0) -> str | None:
    """Run a read-only system command; None on any failure (missing tool, non-zero exit, timeout)."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


_PERF_STATS_RE = re.compile(r'"PerformanceStatistics" = \{([^}]*)\}')
_PERF_ITEM_RE = re.compile(r'"([^"]+)"=(\d+)')


def gpu_performance_statistics() -> dict[str, int] | None:
    """The first IOAccelerator's PerformanceStatistics as {name: int}, or None if unavailable."""
    out = _run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
    match = _PERF_STATS_RE.search(out or "")
    if match is None:
        return None
    return {name: int(value) for name, value in _PERF_ITEM_RE.findall(match.group(1))}


def gpu_in_use_system_memory() -> int | None:
    """Bytes of unified memory the GPU driver reports as in use (system-wide), or None."""
    stats = gpu_performance_statistics()
    if stats is None or "In use system memory" not in stats:
        return None
    return stats["In use system memory"]


def gpu_busy_percent() -> int | None:
    """The GPU's 'Device Utilization %' (0-100) right now (NVIDIA: utilization.gpu), or None if unavailable."""
    stats = gpu_performance_statistics()
    if stats is not None:
        return stats.get("Device Utilization %")
    return nvidia_gpu()["busy_pct"]


def _nvidia_rows(fields: str) -> list[list[str]]:
    """One row of `fields` per NVIDIA GPU from nvidia-smi; [] without nvidia-smi."""
    out = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    return [[cell.strip() for cell in line.split(",")] for line in (out or "").splitlines() if line.strip()]


def nvidia_gpu() -> dict[str, int | None]:
    """The first NVIDIA GPU's busy % and dedicated memory in use (bytes); None per field when unreadable
    (no nvidia-smi, or "[N/A]", as unified-memory parts like the GB10 report memory)."""
    rows = _nvidia_rows("utilization.gpu,memory.used")
    busy, used = (rows[0] + [None, None])[:2] if rows else (None, None)
    return {"busy_pct": int(busy) if busy and busy.isdigit() else None,
            "memory_used": int(used) * MIB if used and used.isdigit() else None}


def system_memory() -> tuple[int, int] | None:
    """(used, total) bytes of the whole machine; used = total - available, as Activity Monitor counts it."""
    try:
        vm = psutil.virtual_memory()
    except (psutil.Error, OSError):
        return None
    return vm.total - vm.available, vm.total


class CpuMeter:
    """CPU use since the previous read, for the whole machine and for single processes, each as a share of
    all cores (0-100). psutil measures between two reads, so the first read of a process is 0."""

    def __init__(self) -> None:
        self._cores = psutil.cpu_count() or 1
        self._procs: dict[int, psutil.Process] = {}

    def machine(self) -> float | None:
        try:
            return round(psutil.cpu_percent(None), 1)
        except (psutil.Error, OSError):
            return None

    def process(self, pid: int | None) -> float | None:
        if pid is None:
            return None
        try:
            proc = self._procs.setdefault(pid, psutil.Process(pid))
            return round(proc.cpu_percent(None) / self._cores, 1)
        except (psutil.Error, OSError):
            self._procs.pop(pid, None)
            return None


def measure_gpu_baseline(samples: int = 3, interval_s: float = 0.1,
                         read_gpu: Callable[[], int | None] = gpu_in_use_system_memory) -> int | None:
    """Median of a few idle readings, taken before llama-server launches; None if unreadable."""
    readings: list[int] = []
    for i in range(samples):
        value = read_gpu()
        if value is not None:
            readings.append(value)
        if i + 1 < samples:
            time.sleep(interval_s)
    return int(statistics.median(readings)) if readings else None


def _mb(value: int | float | None) -> float | None:
    return None if value is None else round(value / MIB, 1)


# --------------------------------------------------------------------------------------------------
# MemoryProbe: one per case

MEMORY_NOTES = [
    "peak_ram_mb: sum over the listed processes of the kernel high-water ri_interval_max_phys_footprint "
    "(proc_pid_rusage RUSAGE_INFO_V4, interval reset at case start); per-process peaks may occur at "
    "different moments, so the sum is an upper bound of the simultaneous peak.",
    "peak_vram_mb: Apple Silicon has unified memory (no separate VRAM). Value = max over 5 Hz + "
    "turn-boundary samples of IOAccelerator 'In use system memory' minus the idle baseline taken "
    "before llama-server started. System-wide scope; it overlaps peak_ram_mb, never add the two.",
    "rss_mb: sum of per-process peak RSS; it includes mmapped model weights and overcounts on "
    "unified memory, reported only for comparison.",
    "mb = MiB (2^20 bytes).",
]


@dataclass
class _ProcessPeaks:
    pid: int
    sampled_max_footprint: int = 0
    last_footprint: int | None = None
    max_rss: int = 0


class MemoryProbe:
    """Tracks memory of a fixed set of processes over one case (start -> mark()* -> report()).

    `pids` maps a label ("adapter", "llama-server") to a pid. The readers are injectable so unit
    tests can drive the probe with fake numbers; production uses the libproc/ioreg/psutil readers.
    """

    def __init__(
        self,
        pids: Mapping[str, int],
        *,
        gpu_baseline_bytes: int | None,
        interval_s: float = SAMPLE_INTERVAL_S,
        read_footprint: Callable[[int], Footprint | None] = read_footprint,
        reset_interval: Callable[[int], bool] = reset_footprint_interval,
        read_gpu: Callable[[], int | None] = gpu_in_use_system_memory,
        read_rss: Callable[[int], int | None] = read_rss,
    ) -> None:
        self._procs = {label: _ProcessPeaks(pid=int(pid)) for label, pid in pids.items()}
        self._gpu_baseline = gpu_baseline_bytes
        self._interval_s = interval_s
        self._read_footprint = read_footprint
        self._reset_interval = reset_interval
        self._read_gpu = read_gpu
        self._read_rss = read_rss
        self._gpu_max: int | None = None
        self._samples = 0
        self._marks: list[str] = []
        self._reset_ok: dict[str, bool] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Reset the kernel high-water windows, take a first sample, start the 5 Hz sampler."""
        self._reset_ok = {label: self._reset_interval(p.pid) for label, p in self._procs.items()}
        self._sample()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="memory-probe", daemon=True)
        self._thread.start()

    def mark(self, label: str) -> None:
        """Synchronous sample at a turn boundary (so peaks between 5 Hz ticks at boundaries count)."""
        self._marks.append(label)
        self._sample()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._sample()

    def __enter__(self) -> MemoryProbe:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._sample()

    def _sample(self) -> None:
        readings = {label: (self._read_footprint(p.pid), self._read_rss(p.pid))
                    for label, p in self._procs.items()}
        gpu = self._read_gpu()
        with self._lock:
            self._samples += 1
            for label, (footprint, rss) in readings.items():
                peaks = self._procs[label]
                if footprint is not None:
                    peaks.last_footprint = footprint.phys_footprint
                    peaks.sampled_max_footprint = max(
                        peaks.sampled_max_footprint, footprint.phys_footprint,
                        footprint.interval_max_phys_footprint)
                if rss is not None:
                    peaks.max_rss = max(peaks.max_rss, rss)
            if gpu is not None:
                self._gpu_max = gpu if self._gpu_max is None else max(self._gpu_max, gpu)

    def report(self) -> dict[str, object]:
        """Metrics for one case (keys follow SPEC §8 result.metrics)."""
        per_process: dict[str, dict[str, object]] = {}
        peak_total = 0
        with self._lock:
            for label, peaks in self._procs.items():
                # The kernel high-water is authoritative while the process lives; the sampled max is
                # the fallback once it has exited (e.g. a crashed server).
                kernel = self._read_footprint(peaks.pid)
                peak = max(peaks.sampled_max_footprint,
                           kernel.interval_max_phys_footprint if kernel else 0)
                peak_total += peak
                per_process[label] = {
                    "pid": peaks.pid,
                    "peak_phys_footprint_mb": _mb(peak),
                    "phys_footprint_mb": _mb(kernel.phys_footprint if kernel else peaks.last_footprint),
                    "peak_rss_mb": _mb(peaks.max_rss),
                    "source": "kernel high-water" if kernel else "samples (process not readable)",
                    "interval_reset": self._reset_ok.get(label, False),
                }
            peak_vram = self._gpu_peak_over_baseline()
            notes = list(MEMORY_NOTES)
            if self._gpu_max is None:
                notes.append("peak_vram_mb is null: IOAccelerator statistics were not readable.")
            elif self._gpu_baseline is None:
                notes.append("peak_vram_mb is null: no idle baseline (server was not started by us).")
            return {
                "peak_ram_mb": _mb(peak_total),
                "peak_vram_mb": peak_vram,
                "peak_unified_gpu_mb": peak_vram,
                "rss_mb": _mb(sum(p.max_rss for p in self._procs.values())),
                "memory_architecture": "unified",
                "memory_notes": notes,
                "per_process": per_process,
                "gpu_in_use_peak_mb_absolute": _mb(self._gpu_max),
                "gpu_baseline_mb": _mb(self._gpu_baseline),
                "memory_samples": self._samples,
                "memory_marks": len(self._marks),
            }

    def _gpu_peak_over_baseline(self) -> float | None:
        if self._gpu_max is None or self._gpu_baseline is None:
            return None
        # Clamped: other processes may free GPU memory during the case, which is not "negative use".
        return _mb(max(0, self._gpu_max - self._gpu_baseline))


# --------------------------------------------------------------------------------------------------
# Static hardware declaration (identical across runs on the same machine)

def _sysctl(name: str) -> str | None:
    out = _run(["sysctl", "-n", name])
    return out.strip() if out else None


def _sw_vers(key: str) -> str | None:
    out = _run(["sw_vers", f"-{key}"])
    return out.strip() if out else None


_GPU_CORES_RE = re.compile(r'"gpu-core-count" = (\d+)')
_LSCPU_MODEL_RE = re.compile(r"^Model name:\s*(.+)$", re.MULTILINE)
_FREE_TOTAL_RE = re.compile(r"^Mem:\s+(\d+)", re.MULTILINE)


def _linux_cpu_cores_memory() -> tuple[str | None, str | None, str | None]:
    """(CPU model, core count, RAM bytes) as text from lscpu/nproc/free, each None when unreadable."""
    model = _LSCPU_MODEL_RE.search(_run(["lscpu"]) or "")
    cores = (_run(["nproc"]) or "").strip()
    total = _FREE_TOTAL_RE.search(_run(["free", "-b"]) or "")
    name = model.group(1).strip() if model else ""
    return name if name not in ("", "-") else None, cores or None, total.group(1) if total else None  # "-": lscpu unknown

POWER_POINTER = "measured per run, not declared: see configuration.run_conditions (pmset power source, powermode, thermal)"


@functools.lru_cache(maxsize=1)
def static_hardware() -> dict[str, object]:
    """Root `hardware` object: static identifiers only, derived once per process.

    Nothing here may change between runs (no free memory, battery %, temperatures): the validator's
    `--baseline` check compares this object for equality. Power is a reading, so `power` only says
    where the measured state is (run_conditions(), before and after every run).
    """
    cpu = _sysctl("machdep.cpu.brand_string")
    cores = _sysctl("hw.ncpu")
    memsize = _sysctl("hw.memsize")
    if cpu is None:  # not macOS
        cpu, cores, memsize = _linux_cpu_cores_memory()
    cpu = cpu or "unknown CPU"
    accel = _run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"]) or ""
    gpu_cores = _GPU_CORES_RE.findall(accel)
    nvidia = [] if gpu_cores else _nvidia_rows("name,driver_version")
    product, build = _sw_vers("productVersion"), _sw_vers("buildVersion")
    is_arm64 = _sysctl("hw.optional.arm64") == "1"
    gpu = (f"{cpu} integrated GPU ({gpu_cores[0]} cores)" if gpu_cores
           else " + ".join(row[0] for row in nvidia) if nvidia else "unknown GPU")
    kernel = (_run(["uname", "-sr"]) or "").strip()
    return {
        "cpu": cpu,
        "cpu_cores": int(cores) if cores and cores.isdigit() else None,
        "gpu": gpu,
        "gpu_count": len(gpu_cores) or len(nvidia),
        "ram_gb": round(int(memsize) / 2**30) if memsize and memsize.isdigit() else None,
        "unified_memory": is_arm64,
        "os": f"macOS {product} ({build})" if product and build else kernel or "unknown OS",
        "driver": (f"Metal (bundled with macOS {build}; no separate GPU driver)" if build
                   else f"NVIDIA driver {nvidia[0][-1]}" if nvidia else "unknown driver"),
        "power": POWER_POINTER,
    }


# --------------------------------------------------------------------------------------------------
# Live run conditions (go to configuration.run_conditions, never into hardware)

_BATT_PCT_RE = re.compile(r"(\d+)%")
_POWERMODE_RE = re.compile(r"^\s*(?:powermode|lowpowermode)\s+(\d+)", re.MULTILINE)
_SLEEP_RE = re.compile(r"^\s*sleep\s+(\d+)(.*)$", re.MULTILINE)
_CPU_SPEED_RE = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")
CPU_HOG_THRESHOLD_PCT = 50.0


def _power_conditions() -> dict[str, object]:
    batt = _run(["pmset", "-g", "batt"])
    source = None
    if batt:
        source = "AC Power" if "'AC Power'" in batt else "Battery Power" if "'Battery Power'" in batt else None
    pct = _BATT_PCT_RE.search(batt or "")
    settings = _run(["pmset", "-g"]) or ""
    mode = _POWERMODE_RE.search(settings)
    sleep = _SLEEP_RE.search(settings)
    therm = _run(["pmset", "-g", "therm"])
    therm_lines = [line.strip() for line in (therm or "").splitlines() if line.strip()]
    speed = _CPU_SPEED_RE.search(therm or "")
    reading = {
        "power_source": source,
        "on_ac_power": None if source is None else source == "AC Power",
        "battery_percent": int(pct.group(1)) if pct else None,
        "powermode": int(mode.group(1)) if mode else None,
        "low_power_mode": None if mode is None else mode.group(1) != "0",
        "sleep_prevented": None if sleep is None else (sleep.group(1) == "0" or "prevented" in sleep.group(2)),
        "thermal": therm_lines if therm is not None else None,
        # "Note: No thermal warning level has been recorded" is the healthy state; a recorded level
        # or a CPU speed limit below 100 % means throttling.
        "thermal_warning": None if therm is None else (
            any("warning level" in line and "No " not in line for line in therm_lines)
            or (speed is not None and int(speed.group(1)) < 100)),
    }
    return {**reading, "power_summary": power_summary(reading)}


def power_summary(reading: Mapping[str, object]) -> str:
    """One line for reports: 'AC Power, powermode 0, no thermal warning, sleep prevented'."""
    mode, warning, sleep = reading.get("powermode"), reading.get("thermal_warning"), reading.get("sleep_prevented")
    return ", ".join([
        str(reading.get("power_source") or "power source unknown"),
        "powermode unknown" if mode is None else f"powermode {mode}",
        "thermal unknown" if warning is None else "thermal warning" if warning else "no thermal warning",
        "sleep unknown" if sleep is None else "sleep prevented" if sleep else "sleep allowed",
    ])


def _process_rows() -> list[dict[str, object]]:
    """`ps` rows as {pid, name, cpu_pct, mem_pct}; `comm` is the executable path without arguments,
    and only its basename is kept. ps reports a decaying %CPU average, which is what 'busy now' needs."""
    out = _run(["ps", "-Ao", "pid=,%cpu=,%mem=,comm="])
    rows = []
    for line in (out or "").splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[0].isdigit():
            rows.append({"pid": int(parts[0]), "name": os.path.basename(parts[3].strip()),
                         "cpu_pct": float(parts[1]), "mem_pct": float(parts[2])})
    return rows


def run_conditions(own_pids: Iterable[int] = ()) -> dict[str, object]:
    """Live machine state before/after a run: power, thermal, load, competing processes."""
    own = {int(p) for p in own_pids} | {os.getpid()}
    rows = _process_rows()
    others = [r for r in rows if r["pid"] not in own]
    gpu = gpu_performance_statistics() or {}
    return {
        **_power_conditions(),
        "load_avg_1_5_15": [round(x, 2) for x in os.getloadavg()] if hasattr(os, "getloadavg") else None,
        "other_llama_servers": [r for r in others if r["name"] == "llama-server"],
        "cpu_heavy_processes": [r for r in others if float(r["cpu_pct"]) >= CPU_HOG_THRESHOLD_PCT],
        "gpu_device_utilization_pct": gpu.get("Device Utilization %") if gpu else nvidia_gpu()["busy_pct"],
        "gpu_in_use_system_memory_mb": _mb(gpu.get("In use system memory")),
    }
