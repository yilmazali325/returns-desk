"""scripts/run.sh, check_engine.sh, test.sh and screenshots.sh, run against stubs (no model, no server).

scripts/run.sh must start with no extra flags under macOS's /bin/bash 3.2.

bash 3.2 with `set -u` treats an empty "${arr[@]}" as unbound, which once broke the documented
default command. The test runs a copy of the script against a stub `.venv/bin/python` that just
records its arguments and exits, so no model or server is needed.
"""

import re
import shutil
import subprocess
from pathlib import Path

from desk.profiles import ENGINE_VERSION

ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path, *args):
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "run.sh", tmp_path / "scripts" / "run.sh")
    stub = tmp_path / ".venv" / "bin" / "python"
    stub.parent.mkdir(parents=True)
    stub.write_text('#!/bin/sh\necho "$@" > "$(dirname "$0")/argv.txt"\n')
    stub.chmod(0o755)
    proc = subprocess.run(
        ["/bin/bash", str(tmp_path / "scripts" / "run.sh"), "--port", "18999", *args],
        capture_output=True, text=True, timeout=30,
    )
    argv = (stub.parent / "argv.txt").read_text() if (stub.parent / "argv.txt").exists() else ""
    return proc, argv


def test_default_command_reaches_the_server_without_unbound_error(tmp_path):
    proc, argv = _run(tmp_path)
    assert "unbound variable" not in proc.stderr
    assert "-m server.app --port 18999" in argv
    assert argv.strip().endswith("--start-llama")


def test_extra_flags_are_passed_through(tmp_path):
    proc, argv = _run(tmp_path, "--fake", "--no-warmup")
    assert "unbound variable" not in proc.stderr
    assert argv.strip().endswith("--start-llama --fake --no-warmup")


def test_the_allow_engine_mismatch_flag_reaches_the_ui_server(tmp_path):
    proc, argv = _run(tmp_path, "--allow-engine-mismatch")
    assert "unbound variable" not in proc.stderr
    assert argv.strip().endswith("--start-llama --allow-engine-mismatch")


# --- scripts/check_engine.sh (called by setup.sh): pinned build, another build, none --------------


def _check_engine(tmp_path, version_line):
    (tmp_path / "scripts").mkdir(exist_ok=True)
    shutil.copy(ROOT / "scripts" / "check_engine.sh", tmp_path / "scripts" / "check_engine.sh")
    env = {"PATH": "/usr/bin:/bin"}
    if version_line is not None:
        fake = tmp_path / "llama-server"
        fake.write_text(f'#!/bin/sh\necho "{version_line}" >&2\n')
        fake.chmod(0o755)
        env["LLAMA_SERVER_BIN"] = str(fake)
    proc = subprocess.run(["/bin/bash", str(tmp_path / "scripts" / "check_engine.sh")], capture_output=True,
                          text=True, timeout=30, env=env)
    record = tmp_path / "var" / "setup" / "engine.txt"
    return proc, record.read_text() if record.exists() else ""


def test_check_engine_pins_the_same_version_as_the_profiles():
    pinned = re.search(r'PINNED_VERSION="([^"]+)"', (ROOT / "scripts" / "check_engine.sh").read_text()).group(1)
    assert pinned == ENGINE_VERSION
    assert ENGINE_VERSION.split("commit ")[1].rstrip(")") in (ROOT / "scripts" / "check_engine.sh").read_text()


def test_check_engine_accepts_the_pinned_build_and_records_it(tmp_path):
    proc, record = _check_engine(tmp_path, f"version: {ENGINE_VERSION}")
    assert proc.returncode == 0 and proc.stdout.startswith(f"ok  llama.cpp {ENGINE_VERSION}")
    assert f"found: {ENGINE_VERSION}\npinned: {ENGINE_VERSION}" in record


def test_check_engine_accepts_a_source_build_of_the_pinned_commit(tmp_path):
    proc, _ = _check_engine(tmp_path, "version: 0.7.0-dev (build 11669, commit 620d2225c)")
    assert proc.returncode == 0 and proc.stdout.startswith("ok  llama.cpp 0.7.0-dev (build 11669, commit 620d2225c)")


def test_check_engine_warns_about_another_build_and_says_how_to_get_the_pinned_one(tmp_path):
    proc, record = _check_engine(tmp_path, "version: 0.7.1 (build 11702, commit abcdef123)")
    assert proc.returncode == 0 and "WARNING: llama.cpp is '0.7.1 (build 11702, commit abcdef123)'" in proc.stdout
    assert "--strict-engine" in proc.stdout and "--allow-engine-mismatch" in proc.stdout
    assert "git checkout 620d2225c" in proc.stdout and "found: 0.7.1" in record


def test_check_engine_without_llama_server_or_with_unreadable_output(tmp_path):
    missing, _ = _check_engine(tmp_path, None)
    assert missing.returncode == 1 and "llama-server not found" in missing.stdout
    garbled, record = _check_engine(tmp_path, "llama.cpp, no version line")
    assert garbled.returncode == 2 and "cannot read the version" in garbled.stdout and "found: unreadable" in record


def test_test_and_screenshot_scripts_call_pytest_with_the_right_selection(tmp_path):
    for name, expected in (("test.sh", "-m not llm -x"), ("screenshots.sh", "-m e2e and not llm tests/e2e -x")):
        copy = tmp_path / name.replace(".sh", "")
        (copy / "scripts").mkdir(parents=True)
        shutil.copy(ROOT / "scripts" / name, copy / "scripts" / name)
        stub = copy / ".venv" / "bin" / "pytest"
        stub.parent.mkdir(parents=True)
        stub.write_text('#!/bin/sh\necho "$@" > "$(dirname "$0")/argv.txt"; echo "$E2E_SHOTS_DIR" > "$(dirname "$0")/shots.txt"\n')
        stub.chmod(0o755)
        proc = subprocess.run(["/bin/bash", str(copy / "scripts" / name), "-x"], capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, proc.stderr
        assert (stub.parent / "argv.txt").read_text().strip() == expected
        shots = (stub.parent / "shots.txt").read_text().strip()
        assert shots == (f"{copy}/var/screenshots" if name == "screenshots.sh" else "")


def test_test_script_needs_the_venv(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "test.sh", tmp_path / "scripts" / "test.sh")
    proc = subprocess.run(["/bin/bash", str(tmp_path / "scripts" / "test.sh")], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 1 and "run scripts/setup.sh first" in proc.stderr
