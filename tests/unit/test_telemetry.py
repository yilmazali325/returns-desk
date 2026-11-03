"""Unit tests for desk/telemetry.py.

Kernel/ioreg readers are exercised for real on this Mac (they need no privileges for our own
processes) and with injected fakes for the arithmetic and the error paths.
"""

from __future__ import annotations

import mmap
import os
import subprocess
import sys
import time

import pytest

from desk import telemetry as T

DARWIN = sys.platform == "darwin"
needs_macos = pytest.mark.skipif(not DARWIN, reason="libproc/ioreg are macOS-only")
MIB = 1024 * 1024

IOREG = ('+-o AGXAcceleratorG17X  <class AGXAcceleratorG17X>\n    {\n      "model" = "Apple M5 Pro"\n'
         '      "gpu-core-count" = 16\n      "PerformanceStatistics" = {"In use system memory (driver)"=0,'
         '"Alloc system memory"=3731292160,"Device Utilization %"=20,"In use system memory"=842137600}\n    }\n')
PMSET_BATT = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=24248419)\t80%; AC attached; not charging\n"
PMSET_G = (" Sleep On Power Button 1\n disksleep            10\n sleep                0 (sleep prevented by "
           "caffeinate)\n displaysleep         0\n powermode            0\n")
PMSET_THERM = ("Note: No thermal warning level has been recorded\nNote: No performance warning level has been "
               "recorded\n")
# `ps -Ao pid=,%cpu=,%mem=,comm=`: comm is the executable path only (no arguments), maybe with spaces.
PS = (f"  {os.getpid()}  99.0  0.4 /x/.venv/bin/python\n"
      "  501   0.0  0.1 /opt/homebrew/bin/llama-server\n"
      "  777  375.9  4.2 /opt/homebrew/bin/ffmpeg\n"
      "  778   1.0  0.0 /bin/zsh\n"
      "  779  60.0  1.5 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome\n"
      "garbage line\n")


def fake_run(outputs: dict[str, str | None]):
    """A stand-in for telemetry._run keyed by the command's first two words."""
    def run(argv, timeout_s=5.0):
        return outputs.get(" ".join(argv[:2]), outputs.get(argv[0]))
    return run


SYSCTL = {"machdep.cpu.brand_string": "Apple M5 Pro\n", "hw.ncpu": "15\n", "hw.memsize": "25769803776\n",
          "hw.optional.arm64": "1\n"}


def full_machine(argv, timeout_s=5.0):
    if argv[0] == "sysctl":
        return SYSCTL.get(argv[-1])
    if argv[0] == "sw_vers":
        return {"-productVersion": "26.4.1\n", "-buildVersion": "25E253\n"}[argv[1]]
    return {"ioreg": IOREG, "pmset": None}.get(argv[0])


@pytest.fixture(autouse=True)
def clear_caches():
    T.static_hardware.cache_clear()
    T._libproc.cache_clear()
    yield
    T.static_hardware.cache_clear()
    T._libproc.cache_clear()


# --------------------------------------------------------------------------------------------------
# libproc readers

@needs_macos
def test_read_footprint_of_own_process_and_reset():
    reading = T.read_footprint(os.getpid())
    assert reading is not None and reading.pid == os.getpid()
    assert reading.phys_footprint > 0 and reading.resident_size > 0
    assert reading.lifetime_max_phys_footprint >= reading.interval_max_phys_footprint > 0
    assert T.reset_footprint_interval(os.getpid()) is True


@needs_macos
def test_kernel_high_water_catches_a_spike_that_sampling_misses():
    pid = os.getpid()
    assert T.reset_footprint_interval(pid)
    before = T.read_footprint(pid)
    block = mmap.mmap(-1, 256 * MIB)
    for offset in range(0, 256 * MIB, mmap.PAGESIZE):
        block[offset] = 1
    block.close()  # freed again before anyone samples
    after = T.read_footprint(pid)
    assert after.interval_max_phys_footprint - before.phys_footprint >= 200 * MIB
    assert after.phys_footprint < after.interval_max_phys_footprint


@needs_macos
def test_readers_return_none_or_false_for_dead_or_foreign_processes():
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    assert T.read_footprint(dead.pid) is None
    assert T.read_rss(dead.pid) is None
    assert T.reset_footprint_interval(1) is False  # launchd: not ours


def test_readers_without_libproc(monkeypatch):
    monkeypatch.setattr(T, "LIBPROC_PATH", "/nonexistent/libproc.dylib")
    T._libproc.cache_clear()
    assert T.read_footprint(os.getpid()) is None
    assert T.reset_footprint_interval(os.getpid()) is False


def test_read_rss_of_own_process():
    assert T.read_rss(os.getpid()) > 0


# --------------------------------------------------------------------------------------------------
# ioreg / GPU

def test_gpu_statistics_parsing(monkeypatch):
    monkeypatch.setattr(T, "_run", fake_run({"ioreg": IOREG}))
    stats = T.gpu_performance_statistics()
    assert stats["In use system memory"] == 842137600 and stats["Device Utilization %"] == 20
    assert T.gpu_in_use_system_memory() == 842137600


@pytest.mark.parametrize("output", [None, "no accelerator here", '"PerformanceStatistics" = {"Other"=1}'])
def test_gpu_statistics_unavailable(monkeypatch, output):
    monkeypatch.setattr(T, "_run", fake_run({"ioreg": output}))
    assert T.gpu_in_use_system_memory() is None


def test_gpu_busy_percent_reads_device_utilization(monkeypatch):
    monkeypatch.setattr(T, "_run", fake_run({"ioreg": IOREG}))
    assert T.gpu_busy_percent() == 20
    monkeypatch.setattr(T, "_run", fake_run({"ioreg": None}))
    assert T.gpu_busy_percent() is None


def test_system_memory_is_used_of_total_and_degrades_to_none(monkeypatch):
    used, total = T.system_memory()
    assert 0 < used < total

    def broken():
        raise OSError("no sysctl")

    monkeypatch.setattr(T.psutil, "virtual_memory", broken)
    assert T.system_memory() is None


def test_cpu_meter_reports_shares_of_all_cores_and_none_for_a_gone_process(monkeypatch):
    meter = T.CpuMeter()
    assert meter.process(None) is None
    assert meter.process(os.getpid()) == 0.0  # psutil measures between two reads
    end = time.perf_counter() + 0.3
    while time.perf_counter() < end:  # keep one core busy
        pass
    share = meter.process(os.getpid())
    assert 0 < share <= 100 / meter._cores + 1
    assert 0 <= meter.machine() <= 100
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    assert meter.process(gone.pid) is None

    def broken(interval):
        raise OSError("no counters")

    monkeypatch.setattr(T.psutil, "cpu_percent", broken)
    assert meter.machine() is None


@needs_macos
def test_gpu_in_use_memory_is_readable_on_this_mac():
    value = T.gpu_in_use_system_memory()
    assert isinstance(value, int) and value > 0


def test_measure_gpu_baseline_takes_the_median(monkeypatch):
    monkeypatch.setattr(T.time, "sleep", lambda s: None)
    readings = iter([300, None, 100, 200])
    assert T.measure_gpu_baseline(samples=4, read_gpu=lambda: next(readings)) == 200
    assert T.measure_gpu_baseline(samples=2, read_gpu=lambda: None) is None


def test_run_returns_none_on_failure():
    assert T._run(["/nonexistent/tool"]) is None
    assert T._run([sys.executable, "-c", "import sys; sys.exit(3)"]) is None
    assert T._run([sys.executable, "-c", "import time; time.sleep(5)"], timeout_s=0.2) is None
    assert T._run([sys.executable, "-c", "print('ok')"], timeout_s=60) == "ok\n"  # generous: a loaded machine


# --------------------------------------------------------------------------------------------------
# MemoryProbe

class FakeMachine:
    """Injectable readers: per-pid footprint/interval-max/RSS in MiB, and a GPU counter."""

    def __init__(self):
        self.footprint = {10: 100, 20: 500}
        self.interval_max = {10: 150, 20: 900}
        self.rss = {10: 120, 20: 3000}
        self.gpu = 1000
        self.resets: list[int] = []

    def read_footprint(self, pid):
        if pid not in self.footprint:
            return None
        return T.Footprint(pid, self.footprint[pid] * MIB, self.interval_max[pid] * MIB, 0, self.rss[pid] * MIB)

    def reset(self, pid):
        self.resets.append(pid)
        return pid != 20

    def read_gpu(self):
        return None if self.gpu is None else self.gpu * MIB

    def read_rss(self, pid):
        return self.rss[pid] * MIB if pid in self.rss else None


def make_probe(machine, baseline_mib=600, interval_s=60.0):
    return T.MemoryProbe({"adapter": 10, "llama-server": 20},
                         gpu_baseline_bytes=None if baseline_mib is None else baseline_mib * MIB,
                         interval_s=interval_s, read_footprint=machine.read_footprint,
                         reset_interval=machine.reset, read_gpu=machine.read_gpu, read_rss=machine.read_rss)


def test_probe_sums_kernel_high_water_and_subtracts_gpu_baseline():
    machine = FakeMachine()
    probe = make_probe(machine)
    probe.start()
    assert machine.resets == [10, 20]
    machine.gpu = 3900  # model loaded and running
    probe.mark("turn 1")
    machine.gpu = 3500
    machine.interval_max[20] = 1400  # kernel saw a spike between samples
    probe.mark("turn 2")
    probe.stop()
    report = probe.report()
    assert report["peak_ram_mb"] == 150 + 1400
    assert report["peak_vram_mb"] == report["peak_unified_gpu_mb"] == 3900 - 600
    assert report["rss_mb"] == 120 + 3000
    assert report["memory_architecture"] == "unified"
    assert report["gpu_in_use_peak_mb_absolute"] == 3900 and report["gpu_baseline_mb"] == 600
    assert report["memory_marks"] == 2 and report["memory_samples"] == 4  # start + 2 marks + stop
    server = report["per_process"]["llama-server"]
    assert server == {"pid": 20, "peak_phys_footprint_mb": 1400, "phys_footprint_mb": 500, "peak_rss_mb": 3000,
                      "source": "kernel high-water", "interval_reset": False}
    assert report["per_process"]["adapter"]["interval_reset"] is True
    assert len(report["memory_notes"]) == len(T.MEMORY_NOTES)


def test_probe_falls_back_to_samples_when_a_process_has_exited():
    machine = FakeMachine()
    with make_probe(machine) as probe:
        machine.interval_max[20] = 2000
        probe.mark("turn 1")
        del machine.footprint[20]  # server crashed
    report = probe.report()
    assert report["per_process"]["llama-server"]["source"] == "samples (process not readable)"
    assert report["per_process"]["llama-server"]["peak_phys_footprint_mb"] == 2000
    assert report["per_process"]["llama-server"]["phys_footprint_mb"] == 500  # last sampled
    assert report["peak_ram_mb"] == 150 + 2000


def test_probe_keeps_the_rss_peak_when_rss_becomes_unreadable():
    machine = FakeMachine()
    readable = {"rss": True}
    probe = T.MemoryProbe({"adapter": 10, "llama-server": 20}, gpu_baseline_bytes=600 * MIB, interval_s=60.0,
                          read_footprint=machine.read_footprint, reset_interval=machine.reset,
                          read_gpu=machine.read_gpu,
                          read_rss=lambda pid: machine.read_rss(pid) if readable["rss"] else None)
    with probe:
        probe.mark("turn 1")
        readable["rss"] = False  # rss no longer readable; the footprint still is
        probe.mark("turn 2")
    assert probe.report()["per_process"]["llama-server"]["peak_rss_mb"] == 3000


def test_probe_gpu_clamped_and_null_cases():
    machine = FakeMachine()
    machine.gpu = 400  # below the baseline: other apps freed GPU memory
    with make_probe(machine) as probe:
        pass
    assert probe.report()["peak_vram_mb"] == 0

    with make_probe(machine, baseline_mib=None) as no_baseline:
        pass
    report = no_baseline.report()
    assert report["peak_vram_mb"] is None and "no idle baseline" in report["memory_notes"][-1]

    machine.gpu = None
    with make_probe(machine) as no_gpu:
        pass
    report = no_gpu.report()
    assert report["peak_vram_mb"] is None and "not readable" in report["memory_notes"][-1]
    assert report["gpu_in_use_peak_mb_absolute"] is None


def test_probe_background_sampler_runs_at_its_interval():
    machine = FakeMachine()
    probe = make_probe(machine, interval_s=0.01)
    probe.start()
    deadline = time.time() + 2
    while probe.report()["memory_samples"] < 5 and time.time() < deadline:
        time.sleep(0.01)
    probe.stop()
    assert probe.report()["memory_samples"] >= 5
    probe.stop()  # idempotent


@needs_macos
def test_probe_with_real_readers_on_this_process():
    baseline = T.measure_gpu_baseline(samples=1)
    with T.MemoryProbe({"adapter": os.getpid()}, gpu_baseline_bytes=baseline) as probe:
        probe.mark("turn 1")
    report = probe.report()
    assert report["peak_ram_mb"] > 0 and report["rss_mb"] > 0
    assert report["per_process"]["adapter"]["source"] == "kernel high-water"
    assert report["per_process"]["adapter"]["interval_reset"] is True
    assert report["peak_vram_mb"] is not None and report["peak_vram_mb"] >= 0


# --------------------------------------------------------------------------------------------------
# static_hardware / run_conditions

def test_static_hardware_matches_spec_strings(monkeypatch):
    monkeypatch.setattr(T, "_run", full_machine)
    assert T.static_hardware() == {
        "cpu": "Apple M5 Pro", "cpu_cores": 15, "gpu": "Apple M5 Pro integrated GPU (16 cores)", "gpu_count": 1,
        "ram_gb": 24, "unified_memory": True, "os": "macOS 26.4.1 (25E253)",
        "driver": "Metal (bundled with macOS 25E253; no separate GPU driver)",
        "power": T.POWER_POINTER,
    }


def test_hardware_power_is_a_pointer_and_the_reading_lives_in_run_conditions(monkeypatch):
    """On battery the static declaration must not claim AC power; run_conditions carries the reading."""
    def on_battery(argv, timeout_s=5.0):
        if argv[:3] == ["pmset", "-g", "batt"]:
            return "Now drawing from 'Battery Power'\n -InternalBattery-0\t54%; discharging;\n"
        return full_machine(argv, timeout_s)

    monkeypatch.setattr(T, "_run", on_battery)
    assert "AC" not in str(T.static_hardware()["power"]) and "run_conditions" in T.static_hardware()["power"]
    reading = T.run_conditions()
    assert reading["on_ac_power"] is False and reading["power_summary"].startswith("Battery Power")


def test_static_hardware_unknown_machine(monkeypatch):
    monkeypatch.setattr(T, "_run", lambda argv, timeout_s=5.0: None)
    hw = T.static_hardware()
    assert hw["cpu"] == "unknown CPU" and hw["cpu_cores"] is None and hw["gpu"] == "unknown GPU"
    assert hw["gpu_count"] == 0 and hw["ram_gb"] is None and hw["unified_memory"] is False
    assert hw["os"] == "unknown OS" and hw["driver"] == "unknown driver"


@needs_macos
def test_static_hardware_is_identical_across_calls_and_static():
    first = T.static_hardware()
    T.static_hardware.cache_clear()
    assert T.static_hardware() == first
    assert set(first) == {"cpu", "cpu_cores", "gpu", "gpu_count", "ram_gb", "unified_memory", "os", "driver", "power"}
    if first["cpu"] == "Apple M5 Pro":  # the declared bench machine (SPEC §8)
        assert first["os"].startswith("macOS ") and first["ram_gb"] == 24 and first["cpu_cores"] == 15


def test_run_conditions_on_ac_with_competitors(monkeypatch):
    outputs = {"pmset -g": PMSET_G, "ps -Ao": PS, "ioreg": IOREG}

    def run(argv, timeout_s=5.0):
        if argv[:3] == ["pmset", "-g", "batt"]:
            return PMSET_BATT
        if argv[:3] == ["pmset", "-g", "therm"]:
            return PMSET_THERM
        return outputs.get(" ".join(argv[:2]), outputs.get(argv[0]))

    monkeypatch.setattr(T, "_run", run)
    rc = T.run_conditions(own_pids=[12345])
    assert rc["power_source"] == "AC Power" and rc["on_ac_power"] is True and rc["battery_percent"] == 80
    assert rc["powermode"] == 0 and rc["low_power_mode"] is False and rc["sleep_prevented"] is True
    assert rc["thermal_warning"] is False and len(rc["thermal"]) == 2
    assert len(rc["load_avg_1_5_15"]) == 3
    assert rc["other_llama_servers"] == [{"pid": 501, "name": "llama-server", "cpu_pct": 0.0, "mem_pct": 0.1}]
    assert [(p["pid"], p["name"]) for p in rc["cpu_heavy_processes"]] == [(777, "ffmpeg"), (779, "Google Chrome")]
    assert all(set(p) == {"pid", "name", "cpu_pct", "mem_pct"} for p in rc["cpu_heavy_processes"])  # never a command line
    assert rc["power_summary"] == "AC Power, powermode 0, no thermal warning, sleep prevented"
    assert rc["gpu_device_utilization_pct"] == 20 and rc["gpu_in_use_system_memory_mb"] == round(842137600 / MIB, 1)


def test_run_conditions_on_battery_low_power_and_throttled(monkeypatch):
    def run(argv, timeout_s=5.0):
        if argv[:3] == ["pmset", "-g", "batt"]:
            return "Now drawing from 'Battery Power'\n -InternalBattery-0\t41%; discharging\n"
        if argv[:3] == ["pmset", "-g", "therm"]:
            return "CPU_Scheduler_Limit \t= 100\nCPU_Speed_Limit \t= 70\n"
        if argv[:2] == ["pmset", "-g"]:
            return " sleep                1\n lowpowermode         1\n"
        return None

    monkeypatch.setattr(T, "_run", run)
    rc = T.run_conditions()
    assert rc["power_source"] == "Battery Power" and rc["on_ac_power"] is False and rc["battery_percent"] == 41
    assert rc["powermode"] == 1 and rc["low_power_mode"] is True and rc["sleep_prevented"] is False
    assert rc["thermal_warning"] is True
    assert rc["other_llama_servers"] == [] and rc["gpu_device_utilization_pct"] is None
    assert rc["power_summary"] == "Battery Power, powermode 1, thermal warning, sleep allowed"


def test_run_conditions_when_nothing_is_readable(monkeypatch):
    monkeypatch.setattr(T, "_run", lambda argv, timeout_s=5.0: None)
    rc = T.run_conditions()
    assert rc["power_source"] is None and rc["on_ac_power"] is None and rc["battery_percent"] is None
    assert rc["powermode"] is None and rc["low_power_mode"] is None and rc["sleep_prevented"] is None
    assert rc["thermal"] is None and rc["thermal_warning"] is None
    assert rc["gpu_in_use_system_memory_mb"] is None
    assert rc["power_summary"] == "power source unknown, powermode unknown, thermal unknown, sleep unknown"


def test_run_conditions_unknown_power_source_and_recorded_warning(monkeypatch):
    def run(argv, timeout_s=5.0):
        if argv[:3] == ["pmset", "-g", "batt"]:
            return "Now drawing from 'UPS Power'\n"
        if argv[:3] == ["pmset", "-g", "therm"]:
            return "Thermal warning level set to 2.\n"
        return None

    monkeypatch.setattr(T, "_run", run)
    rc = T.run_conditions()
    assert rc["power_source"] is None and rc["thermal_warning"] is True


@needs_macos
def test_run_conditions_live_smoke():
    rc = T.run_conditions()
    assert rc["power_source"] in ("AC Power", "Battery Power")
    assert isinstance(rc["other_llama_servers"], list)
