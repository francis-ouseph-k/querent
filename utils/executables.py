"""
utils/executables.py
────────────────────
EXECUTABLE PROBING — verify an external binary is not just PRESENT but actually
LAUNCHABLE on this OS/architecture, before a long-running pipeline depends on it.

WHY THIS EXISTS
───────────────
Pipelines that shell out to precompiled binaries (llama.cpp's llama-quantize /
llama-server, ffmpeg, etc.) commonly guard with `path.exists()`. That is not
enough: a file can exist yet be unrunnable —
  • wrong CPU architecture (Windows: OSError/WinError 216; Linux: "Exec format error")
  • a missing runtime dependency (Windows: absent DLL, WinError 126;
    Linux: missing shared object, "cannot open shared object file")
  • on Linux, the execute bit was never set (chmod +x) so the OS refuses to run it
In every one of these, `.exists()` returns True and the failure only surfaces
LATER — often after an expensive step (e.g. a ~12 GB model merge). Probing up
front turns a 40-minutes-in crash into a ~1-second, clearly-worded error.

WHAT THIS DOES *NOT* DO
───────────────────────
It probes an EXPLICIT path you give it. It does NOT search $PATH — if you want
"find llama-quantize on PATH", use shutil.which() first, then probe the result.

CROSS-PLATFORM
──────────────
Works on Windows and Linux/WSL2. The only OS-specific rule is the Linux
execute-bit check (os.X_OK), which is meaningless on Windows and skipped there.
Binary NAMES (llama-quantize.exe vs llama-quantize) are the caller's concern —
this module probes whatever Path it is given.

USAGE
─────
    from utils.executables import probe_executable, require_executable

    # Soft form — returns an error string (or None if OK); caller aggregates.
    err = probe_executable(quantize_bin, "quantiser binary")
    if err:
        errors.append(err)

    # Hard form — raises FileNotFoundError / PermissionError / TimeoutError /
    # OSError on failure.
    require_executable(quantize_bin, "quantiser binary")
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from utils.logging_config import get_logger

logger = get_logger(__name__)

# How long to let a probe launch run before giving up. A well-formed CLI binary
# invoked with no useful args prints usage and exits almost immediately, so this
# is generous headroom, NOT an expected wait. Kept small so a wedged binary
# cannot stall prerequisite checks.
_PROBE_TIMEOUT_S = 15

# ── Message prefixes (single source of truth) ────────────────────────────────
# require_executable() maps an error string back to an exception TYPE. Keying
# that mapping off these constants — rather than off inline literals scattered
# in two places — means a reworded message can never silently break the mapping
# (REVIEW FIX: the string-coupling fragility). Both producer (probe_executable)
# and consumer (require_executable) reference the same constants.
_MSG_NOT_FOUND = "not found"
_MSG_NOT_FILE  = "is not a file"
_MSG_NOT_EXEC  = "is not executable"
_MSG_TIMEOUT   = "did not exit within"
_MSG_NO_LAUNCH = "will not execute"


def probe_executable(path: Path, label: str) -> str | None:
    """
    Confirm `path` is a present, launchable executable on this OS/architecture.

    Returns None when the binary is runnable, otherwise a human-readable error
    string describing the specific failure (and, where useful, the fix). This
    "soft" shape lets a prerequisite check collect several problems and report
    them together rather than dying on the first.

    The probe LAUNCHES the binary with no arguments and a short timeout. We do
    not care about its exit code — a valid CLI typically exits nonzero with a
    "usage" message when given no args, which is fine; we only care that the OS
    was ABLE to execute it. Wrong-architecture or missing-dependency binaries
    raise OSError here instead of at the real call site later.

    NOTE (limitation): a binary that exits 0 with no output on no-args (rare)
    passes this generic probe. We deliberately keep the probe generic rather
    than binary-specific (e.g. `--help`), since not all builds support the same
    flags and a brittle flag-probe is worse than a permissive launch-probe.

    Args:
        path:  Full path to the binary (name/extension already resolved by caller).
        label: Human name used in messages, e.g. "quantiser binary".

    Returns:
        None if the binary is runnable, else an error string.
    """
    # Resolve to an absolute path (a relative input resolves against the CWD) so
    # the existence check and every error message name a concrete location, not
    # a confusing "../foo". This does NOT consult $PATH — see module docstring.
    path = path.resolve()

    if not path.exists():
        return f"{label} {_MSG_NOT_FOUND}: {path}"

    if not path.is_file():
        return f"{label} {_MSG_NOT_FILE}: {path}"

    # Linux/WSL only: a file can exist without the execute bit, which makes it
    # unrunnable regardless of contents. os.X_OK is not meaningful on Windows
    # (which decides executability by extension / PE header), so skip it there.
    if os.name != "nt" and not os.access(path, os.X_OK):
        return (
            f"{label} exists but {_MSG_NOT_EXEC}: {path}\n"
            f"    Fix:  chmod +x {path}"
        )

    # Final proof: actually launch it. subprocess.run(timeout=) KILLS and REAPS
    # the child on timeout (verified against CPython: it calls process.kill()
    # then wait()/communicate() before re-raising, and the Popen context manager
    # waits again on exit) — so there is no zombie/orphan to clean up here.
    try:
        subprocess.run(
            [str(path)],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        # A timeout means the binary LAUNCHED but did not exit — the OPPOSITE of
        # a wrong-arch/missing-dependency failure. Report it as its own case so
        # the message doesn't send anyone down the wrong debugging path.
        return (
            f"{label} launched but {_MSG_TIMEOUT} {_PROBE_TIMEOUT_S}s: {path}\n"
            f"    It ran, so this is NOT a wrong-architecture or missing-library\n"
            f"    problem — it is waiting on input or stuck. Try it manually: {path}"
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # OSError/WinError 216 = wrong architecture; 126 / missing .dll / .so =
        # unmet runtime dependency.
        return (
            f"{label} exists but {_MSG_NO_LAUNCH}: {path}\n"
            f"    {type(exc).__name__}: {exc}\n"
            f"    Likely a wrong-architecture build or a missing runtime "
            f"dependency (.dll on Windows, .so on Linux)."
        )

    logger.info(component="executables", event="probe_ok", label=label, path=str(path))
    return None


def require_executable(path: Path, label: str) -> Path:
    """
    Hard variant of probe_executable: raise instead of returning a string.

    Useful when a single binary is an unconditional prerequisite and you want to
    stop immediately with a specific exception type. Prefer probe_executable when
    you are aggregating several checks into one combined error report.

    The error→exception mapping keys off the module-level _MSG_* constants that
    probe_executable used to build the message, so rewording a message cannot
    silently break the mapping.

    Raises:
        FileNotFoundError: binary missing or not a regular file.
        PermissionError:   present but not marked executable (Linux).
        TimeoutError:      launched but did not exit within the probe timeout.
        OSError:           present but the OS could not launch it (arch/deps).
    """
    err = probe_executable(path, label)
    if err is None:
        return path

    logger.error(component="executables", event="probe_failed", label=label,
                 path=str(path), reason=err)

    if _MSG_NOT_FOUND in err or _MSG_NOT_FILE in err:
        raise FileNotFoundError(err)
    if _MSG_NOT_EXEC in err:
        raise PermissionError(err)
    if _MSG_TIMEOUT in err:
        raise TimeoutError(err)
    raise OSError(err)