"""
grader_core.py  --  AI 2000 shared autograder engine
====================================================

House law: "Restart & Run All must pass."

This engine runs the student's work in a *fresh process*, then runs per-level
probe snippets in that same live process to pull each student function's output
onto disk as an artifact.  A checker (running outside that process) compares the
artifact against a reference and awards weighted, tiered credit.  Nothing is
scraped from printed output; checks call the student's functions with
grader-controlled inputs.

Two entry modes, same contract:

* entry="notebook" (default) — the A1–A4 path.  nbclient executes the notebook
  top to bottom in a fresh kernel, then the probes run in that kernel.
* entry="repo" — the team project's path.  A subprocess adds the repo to
  sys.path and imports the student's package, then the probes run in that
  interpreter.  The import IS the Restart & Run All: a module imported into a
  clean interpreter has no hidden state to carry, which is the whole reason the
  project graduates out of notebooks.

The engine is assignment-agnostic.  An assignment ships its own thin `grader.py`
that defines a list of `Level` objects and calls `run_grader(...)`.  The same
engine backs the instructor's headless `batch_regrade.py` (the grade of record).

Design notes
------------
* allow_errors defaults to True: a single unfilled TODO must NOT zero an entire
  notebook of otherwise-working levels.  The engine still reports whether the
  notebook would have survived a strict Run-All, so the discipline is taught,
  not enforced with a sledgehammer.  Set strict=True to flip this.
* Determinism is the assignment's job (integer arithmetic, pinned reductions);
  the engine only compares.  Value checks default to exact (`array_equal`) with
  an optional per-level fractional-tolerance for float ops, where a
  different-but-valid summation order can shift a few least-significant bits.
* results.json is always written next to the notebook so batch tooling is
  trivial: it re-runs this engine and diffs the machine-readable record.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


# --------------------------------------------------------------------------- #
# Terminal styling (leveled, colored, targeted messages)                       #
# --------------------------------------------------------------------------- #
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    GREY = "\033[90m"


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("GRADER_FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


_COLOR = _supports_color()


def _c(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}" if _COLOR else text


# --------------------------------------------------------------------------- #
# Result records                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class LevelResult:
    name: str
    title: str
    weight: float
    earned: float = 0.0
    status: str = "PENDING"          # PASS | PARTIAL | FAIL | SKIP | ERROR
    messages: list[str] = field(default_factory=list)
    bonus: bool = False

    def add(self, msg: str) -> None:
        self.messages.append(msg)


@dataclass
class Level:
    """One graded unit of an assignment.

    name       : short id used for SKIP-gating / results.json keys (e.g. "L2")
    title      : human label; match the notebook's section heading
    weight     : points this level contributes
    probe      : Python source executed *inside the student kernel*.  It should
                 call the student's function and np.save its result to
                 os.path.join(GRADE_DIR, f"{name}.npy") (or write a sentinel).
                 GRADE_DIR and np are injected before the probe runs.
    check      : Callable(grade_dir: Path, result: LevelResult) -> None.
                 Runs OUTSIDE the kernel; loads the probe's artifact, compares
                 to a reference, mutates `result` (earned/status/messages).
    requires   : names of levels that must have status PASS/PARTIAL for this one
                 to run; otherwise this level is SKIPped.
    bonus      : bonus levels are reported separately and not counted in the
                 100-point base.
    """
    name: str
    title: str
    weight: float
    probe: str
    check: Callable[[Path, LevelResult], None]
    requires: list[str] = field(default_factory=list)
    bonus: bool = False


# --------------------------------------------------------------------------- #
# Probe construction                                                           #
# --------------------------------------------------------------------------- #
def probe_call(key: str, expr: str) -> str:
    """Build probe source that evaluates `expr` in the student kernel and saves
    the result to GRADE_DIR/{key}.npy.  Any exception (including the
    NotImplementedError raised by an unattempted stub) is recorded to
    GRADE_DIR/{key}.error instead, so one broken level can't take down the rest.

    `expr` must be an expression calling the student's function, e.g.
    "to_grayscale(_TIMG)".
    """
    return (
        f"try:\n"
        f"    _r = {expr}\n"
        f"    np.save(_os.path.join(GRADE_DIR, '{key}.npy'), np.asarray(_r))\n"
        f"except Exception as _e:\n"
        f"    _grader_fail('{key}', repr(_e))\n"
    )


# --------------------------------------------------------------------------- #
# Tiered array checker  (returns-> shape -> dtype -> values)                   #
# --------------------------------------------------------------------------- #
def check_array(
    grade_dir: Path,
    result: LevelResult,
    name: str,
    expected,
    *,
    tiers=(0.05, 0.10, 0.85),
    exact: bool = True,
    frac_tol: float = 0.0,
    atol: float = 0.0,
    rtol: float = 0.0,
    hint_values: str = "check your arithmetic order (clip before cast?)",
    hint_dtype: str | None = None,
) -> None:
    """Load {name}.npy, award tiered credit against `expected`.

    tiers = (returns_ok, shape_ok, dtype_ok+values).  The last tier splits
    internally: dtype gets 1/8 of it, values 7/8 — so with the defaults a
    correctly-shaped, correctly-typed answer with wrong values tops out around
    25% of the level (the "you built the plumbing" floor). Value credit beyond
    that is reserved for genuine near-misses: partial value credit only accrues
    above 90% element match, so a wrong answer that happens to overlap the
    target (e.g. a zeros array on a dark image) earns plumbing credit only.

    Three value regimes.  Pick by what the level computes, not by taste:

    * exact (the default) — `array_equal`.  Integers: indices, counts, labels,
      and any pixel arithmetic that stays in fixed point.
    * `frac_tol` — fraction of *exactly equal* elements, for integer-valued
      results where a few elements may legitimately differ.  Integers only;
      handing this a float array raises, because it compares as int64 and a
      cosine score would silently truncate to -1, 0 or 1.
    `hint_dtype` overrides the default "cast at the end" advice, which is only
    true when the values are right and the container is wrong.  On a level where
    the wrong dtype means the student computed the wrong *thing* — returning
    scores where indices were asked for — that default sends them off to cast a
    correct answer out of a wrong one.  Name the confusion instead.

    * `atol` / `rtol` — `np.isclose`, for real-valued results where a
      different-but-valid summation order shifts the last bits.  A2 grades
      every float level this way (atol=1e-5, rtol=1e-4: two decades above the
      observed drift between implementations, three below the smallest
      conceptual mistake).  `frac_tol` may be combined with it to allow a
      fraction of elements to miss entirely.
    """
    import numpy as np

    w = result.weight
    t_return, t_shape, t_rest = tiers
    t_dtype = t_rest * 0.125
    t_values = t_rest * 0.875

    artifact = grade_dir / f"{name}.npy"
    err = grade_dir / f"{name}.error"
    if err.exists():
        result.status = "FAIL"
        result.add(_c("✗ " + err.read_text().strip(), C.RED))
        return
    if not artifact.exists():
        result.status = "FAIL"
        result.add(_c(f"✗ function did not run / produced no output — is it defined and returning a value?", C.RED))
        return

    try:
        student = np.load(artifact, allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        result.status = "FAIL"
        result.add(_c(f"✗ output was not a NumPy array ({e})", C.RED))
        return

    expected = np.asarray(expected)
    earned = t_return * w
    result.add(_c("✓ returns an array", C.GREEN))

    # shape
    if student.shape == expected.shape:
        earned += t_shape * w
        result.add(_c(f"✓ shape {student.shape}", C.GREEN))
    else:
        result.status = "PARTIAL"
        result.add(_c(f"✗ shape {student.shape}, expected {expected.shape}", C.YELLOW))
        result.earned = round(earned, 4)
        return  # can't meaningfully compare values on a shape mismatch

    # dtype
    if student.dtype == expected.dtype:
        earned += t_dtype * w
        result.add(_c(f"✓ dtype {student.dtype}", C.GREEN))
    else:
        advice = (hint_dtype if hint_dtype is not None
                  else f"cast with .astype(np.{expected.dtype}) at the end")
        result.add(_c(f"✗ dtype {student.dtype}, expected {expected.dtype} "
                      f"({advice})", C.YELLOW))

    # values
    if atol or rtol:
        # real-valued: "close enough" replaces "identical", tiers unchanged
        close = np.isclose(student.astype(np.float64), expected.astype(np.float64),
                           atol=atol, rtol=rtol, equal_nan=False)
        frac = float(np.mean(close)) if student.size else 0.0
        ok = frac >= (1.0 - frac_tol)
    elif exact and frac_tol == 0.0:
        ok = bool(np.array_equal(student, expected))
        frac = 1.0 if ok else float(np.mean(student == expected)) if student.size else 0.0
    else:
        # exact-match-with-grace: fraction of elements exactly equal.  int64 only
        # — see the docstring.  The guard reads `expected`, never `student`: a
        # float-valued reference is an authoring bug and verify.sh should hear
        # about it immediately, but a student who forgot the final cast has made
        # a student mistake, already priced in the dtype tier, and their values
        # should still grade.
        if np.issubdtype(expected.dtype, np.floating):
            raise ValueError(
                f"{name}: frac_tol compares as int64 and would truncate the "
                f"reference; use atol=/rtol= for a real-valued level")
        frac = float(np.mean(student.astype(np.int64) == expected.astype(np.int64))) if student.size else 0.0
        ok = frac >= (1.0 - frac_tol)

    if ok:
        earned += t_values * w
        result.status = "PASS"
        result.add(_c("✓ values match", C.GREEN))
    else:
        # graceful partial: scale value credit by how close they are, but only
        # above a floor so a totally-wrong answer earns nothing on this tier
        scaled = max(0.0, (frac - 0.9) / 0.1) * t_values * w if frac > 0.9 else 0.0
        earned += scaled
        result.status = "PARTIAL"
        pct = f"{frac*100:.1f}%"
        result.add(_c(f"✗ values off ({pct} of elements match) — {hint_values}", C.YELLOW))

    result.earned = round(earned, 4)


# --------------------------------------------------------------------------- #
# Multi-part levels: combine several tiered sub-checks into one level          #
# --------------------------------------------------------------------------- #
def check_parts(grade_dir: Path, result: LevelResult, parts) -> None:
    """Grade a level made of several sub-checks, each a share of the level.

    parts: list of (key, label, expected, fraction_of_level, check_array_kwargs).
    The fractions should sum to 1.0.  The level PASSes only if every part
    passes; it FAILs only if nothing was earned at all; otherwise PARTIAL.
    """
    earned = 0.0
    statuses = []
    for key, label, expected, frac, kw in parts:
        sub = LevelResult(name=key, title=label, weight=result.weight * frac)
        check_array(grade_dir, sub, key, expected, **kw)
        earned += sub.earned
        statuses.append(sub.status)
        for m in sub.messages:
            result.add(f"{label}: {m}")
    result.earned = round(earned, 4)
    if all(s == "PASS" for s in statuses):
        result.status = "PASS"
    elif earned == 0:
        result.status = "FAIL"
    else:
        result.status = "PARTIAL"

# --------------------------------------------------------------------------- #
# Notebook execution in a fresh kernel                                         #
# --------------------------------------------------------------------------- #
_PROBE_PREAMBLE = (
    "import os as _os, numpy as _np_grader\n"
    "import numpy as np\n"
    "GRADE_DIR = _os.environ['GRADE_DIR']\n"
    "def _grader_fail(_name, _exc):\n"
    "    with open(_os.path.join(GRADE_DIR, _name + '.error'), 'w') as _f:\n"
    "        _f.write(str(_exc))\n"
)


def _run_notebook(nb_path: Path, grade_dir: Path, levels: list[Level],
                  strict: bool, timeout: int) -> dict:
    """Execute the notebook, then run each probe in the same kernel.

    Returns a dict with keys: 'run_all_ok' (bool), 'run_all_msg' (str),
    'first_error_cell' (int|None).
    """
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=not strict,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )

    run_all_ok = True
    run_all_msg = "notebook executed top-to-bottom with no errors"
    first_error_cell = None

    with client.setup_kernel():
        # 1) Restart & Run All the student's cells in order.
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            try:
                client.execute_cell(cell, idx)
            except CellExecutionError as e:  # only reachable when strict=True
                run_all_ok = False
                first_error_cell = idx
                run_all_msg = f"cell {idx} raised: {str(e).splitlines()[-1][:200]}"
                break
            # In non-strict mode nbclient records errors in outputs instead of
            # raising; detect them so we can still report Run-All honesty.
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    if run_all_ok:
                        first_error_cell = idx
                        run_all_msg = (f"cell {idx} raised {out.get('ename','?')}: "
                                       f"{out.get('evalue','')[:160]} "
                                       f"(a strict Run-All would stop here)")
                    run_all_ok = False

        # 2) Inject the probe preamble, then each level's probe, in the same
        #    kernel.  A probe failure is captured as a per-level .error sentinel.
        client.execute_cell(nbformat.v4.new_code_cell(_PROBE_PREAMBLE), -1)
        for lvl in levels:
            wrapped = (
                "try:\n"
                + "\n".join("    " + line for line in lvl.probe.splitlines())
                + f"\nexcept Exception as _e:\n"
                f"    _grader_fail({lvl.name!r}, repr(_e))\n"
            )
            client.execute_cell(nbformat.v4.new_code_cell(wrapped), -1)

    return {
        "run_all_ok": run_all_ok,
        "run_all_msg": run_all_msg,
        "first_error_cell": first_error_cell,
    }


# --------------------------------------------------------------------------- #
# Repo execution in a fresh interpreter                                        #
# --------------------------------------------------------------------------- #
# The driver runs as its own process so student code never shares a namespace
# with the grader.  All probes run inside this one process, in LEVELS order,
# exactly as they share one kernel in notebook mode -- so the probe
# self-containment rule (every PROBE_* imports what it uses) applies unchanged.
# Per-level subprocesses would make that rule unnecessary and would also pay
# torch's import cost once per level.
_REPO_DRIVER = r'''
import os, sys, json, importlib, pkgutil

REPO, SPEC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, REPO)
os.chdir(REPO)          # student code reads provided/ relative to the repo root

with open(SPEC) as f:
    spec = json.load(f)

status = {"import_ok": True, "failed_module": None,
          "import_msg": "package imported cleanly in a fresh interpreter"}
pkg_name = spec["package"]

try:
    pkg = importlib.import_module(pkg_name)
    # Walk the submodules too.  An empty __init__.py imports fine while a
    # broken module beside it is never touched; a notebook's later cells always
    # run, so importing only the package would grade a repo more leniently than
    # a notebook for the same mistake.
    if hasattr(pkg, "__path__"):
        for m in pkgutil.iter_modules(pkg.__path__):
            if m.name.startswith("_"):
                continue
            try:
                importlib.import_module(pkg_name + "." + m.name)
            except Exception as e:
                status["import_ok"] = False
                status["failed_module"] = pkg_name + "." + m.name
                status["import_msg"] = ("%s.%s raised %s: %s"
                    % (pkg_name, m.name, type(e).__name__, str(e)[:160]))
                break
except Exception as e:
    status["import_ok"] = False
    status["failed_module"] = pkg_name
    status["import_msg"] = ("import %s raised %s: %s"
        % (pkg_name, type(e).__name__, str(e)[:160]))

# Written before the probes run, so a probe that hard-crashes the interpreter
# still leaves an import verdict behind.
with open(OUT, "w") as f:
    json.dump(status, f)

if spec["strict"] and not status["import_ok"]:
    sys.exit(0)

ns = {}
exec(spec["preamble"], ns)
for name, src in spec["probes"]:
    try:
        exec(src, ns)
    except Exception as e:
        ns["_grader_fail"](name, repr(e))
'''


def _run_repo(repo_path: Path, grade_dir: Path, levels: list[Level],
              strict: bool, timeout: int, package: str) -> dict:
    """Import the student's package in a fresh interpreter, then run the probes.

    The repo analogue of Restart & Run All.  Returns the same three keys as
    `_run_notebook` so everything downstream is indifferent to which ran.
    """
    import subprocess

    spec = {
        "package": package,
        "preamble": _PROBE_PREAMBLE,
        "probes": [[l.name, l.probe] for l in levels],
        "strict": bool(strict),
    }

    with tempfile.TemporaryDirectory(prefix="ai2000_driver_") as tmp:
        t = Path(tmp)
        (t / "spec.json").write_text(json.dumps(spec))
        (t / "_driver.py").write_text(_REPO_DRIVER)
        status_f = t / "status.json"

        env = dict(os.environ)
        env["GRADE_DIR"] = str(grade_dir)

        try:
            proc = subprocess.run(
                [sys.executable, str(t / "_driver.py"), str(repo_path),
                 str(t / "spec.json"), str(status_f)],
                capture_output=True, text=True, timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "run_all_ok": False,
                "run_all_msg": f"the repo did not finish within {timeout}s",
                "first_error_cell": None,
            }

        if status_f.exists():
            status = json.loads(status_f.read_text())
            return {
                "run_all_ok": bool(status["import_ok"]),
                "run_all_msg": status["import_msg"],
                "first_error_cell": status["failed_module"],
            }

        # No status file means the driver died before it could write one --
        # a syntax error at import time, or a hard interpreter crash.
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {
            "run_all_ok": False,
            "run_all_msg": ("the interpreter died before the package finished importing: "
                            + (tail[-1][:200] if tail else "no output")),
            "first_error_cell": None,
        }


# --------------------------------------------------------------------------- #
# Top-level entry point                                                        #
# --------------------------------------------------------------------------- #
def run_grader(
    target: str,
    levels: list[Level],
    *,
    assignment_id: str = "A?",
    assignment_title: str = "",
    strict: bool = False,
    timeout: int = 180,
    results_path: Optional[str] = None,
    clean_outputs: Optional[list[str]] = None,
    quiet: bool = False,
    entry: str = "notebook",
    repo_package: Optional[str] = None,
) -> dict:
    """Grade `target` against `levels`; print a leveled report; write results.json.

    target        : path to the student's notebook (entry="notebook") or to the
                    root of their repo (entry="repo").
    entry         : "notebook" runs nbclient in a fresh kernel; "repo" imports
                    `repo_package` in a fresh interpreter.  Everything after
                    execution -- SKIP-gating, tiered checks, scoring, the report,
                    results.json -- is identical either way.
    repo_package  : required when entry="repo"; the importable package name,
                    e.g. "facedetect".
    clean_outputs : filenames (relative to the notebook, or to the repo root)
                    removed *before* execution, so credit for a produced
                    artifact can only be earned if THIS run regenerates it
                    (no stale-file leakage).

    Returns the results dict (also written to disk).
    """
    if entry not in ("notebook", "repo"):
        raise ValueError(f"entry must be 'notebook' or 'repo', got {entry!r}")
    if entry == "repo" and not repo_package:
        raise ValueError("entry='repo' requires repo_package=<importable name>")

    target_path = Path(target).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"{'repo' if entry == 'repo' else 'notebook'} "
                                f"not found: {target_path}")

    # Where sibling files (artifacts, results.json) live: next to the notebook,
    # or at the root of the repo.
    base_dir = target_path if entry == "repo" else target_path.parent

    for fname in (clean_outputs or []):
        stale = base_dir / fname
        if stale.exists():
            stale.unlink()

    base_levels = [l for l in levels if not l.bonus]
    base_total = sum(l.weight for l in base_levels)

    with tempfile.TemporaryDirectory(prefix="ai2000_grade_") as tmp:
        grade_dir = Path(tmp)
        os.environ["GRADE_DIR"] = str(grade_dir)
        try:
            if entry == "repo":
                run_info = _run_repo(target_path, grade_dir, levels, strict,
                                     timeout, repo_package)
            else:
                run_info = _run_notebook(target_path, grade_dir, levels, strict, timeout)
        finally:
            os.environ.pop("GRADE_DIR", None)

        # Evaluate each level (respecting SKIP-gating on requires).
        results: dict[str, LevelResult] = {}
        passed_names: set[str] = set()
        for lvl in levels:
            r = LevelResult(name=lvl.name, title=lvl.title, weight=lvl.weight, bonus=lvl.bonus)
            unmet = [req for req in lvl.requires if req not in passed_names]
            if unmet:
                r.status = "SKIP"
                r.add(_c(f"⤼ skipped — needs {', '.join(unmet)} to pass first", C.GREY))
            else:
                try:
                    lvl.check(grade_dir, r)
                except Exception as e:  # noqa: BLE001 -- a broken checker must not crash the run
                    r.status = "ERROR"
                    r.add(_c(f"‼ grader internal error: {e}", C.RED))
                    r.add(_c(traceback.format_exc(), C.DIM))
            if r.status in ("PASS", "PARTIAL"):
                passed_names.add(lvl.name)
            results[lvl.name] = r

    base_earned = round(sum(results[l.name].earned for l in base_levels), 4)
    bonus_earned = round(sum(results[l.name].earned for l in levels if l.bonus), 4)

    payload = {
        "assignment_id": assignment_id,
        "assignment_title": assignment_title,
        # "notebook" keeps its name so existing consumers (batch_regrade, any
        # saved results.json) still read; for a repo it holds the repo dir name.
        "notebook": str(target_path.name),
        "entry": entry,
        "run_all_ok": run_info["run_all_ok"],
        "run_all_msg": run_info["run_all_msg"],
        # Both executors compute this and it used to be dropped on the floor.
        # Notebook mode: index of the first cell that raised.  Repo mode: the
        # dotted name of the module that failed to import.  None if clean.
        "first_error_cell": run_info["first_error_cell"],
        "base_earned": base_earned,
        "base_total": base_total,
        "bonus_earned": bonus_earned,
        "score_pct": round(100.0 * base_earned / base_total, 2) if base_total else 0.0,
        "levels": {name: asdict(r) for name, r in results.items()},
        "strict": strict,
    }

    out_path = Path(results_path) if results_path else base_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2))

    if not quiet:
        _print_report(payload, results, base_levels, levels)

    return payload


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
_STATUS_STYLE = {
    "PASS": (C.GREEN, "PASS "),
    "PARTIAL": (C.YELLOW, "PART "),
    "FAIL": (C.RED, "FAIL "),
    "SKIP": (C.GREY, "SKIP "),
    "ERROR": (C.RED, "ERR! "),
    "PENDING": (C.GREY, ".... "),
}


def _bar(pct: float, width: int = 24) -> str:
    filled = int(round(pct / 100 * width))
    return "█" * filled + "·" * (width - filled)


def _print_report(payload, results, base_levels, all_levels):
    line = "─" * 66
    print()
    print(_c(f"  AI 2000 · {payload['assignment_id']} — {payload['assignment_title']}", C.BOLD + C.CYAN))
    print(_c("  " + line, C.DIM))

    # Run-All honesty banner.  Same rule, named the way the student meets it.
    repo = payload.get("entry") == "repo"
    label = "Fresh import" if repo else "Restart & Run All"
    if payload["run_all_ok"]:
        print("  " + _c(f"{label}: clean ✓", C.GREEN))
    else:
        print("  " + _c(f"{label}: WOULD HALT — {payload['run_all_msg']}", C.YELLOW))
        print("  " + _c("  (levels below graded independently; fix the halt before you submit)", C.DIM))
    print(_c("  " + line, C.DIM))

    for lvl in all_levels:
        r = results[lvl.name]
        color, tag = _STATUS_STYLE.get(r.status, (C.GREY, r.status))
        head = f"  [{_c(tag, color)}] {lvl.name}  {r.title}"
        pts = "" if r.status == "SKIP" else f"{r.earned:g}/{r.weight:g}"
        if lvl.bonus:
            pts = _c(f"+{r.earned:g} bonus", C.BLUE) if r.earned else _c("+0 bonus", C.DIM)
        print(f"{head}{'':>2}{_c(pts, C.DIM):>8}")
        for m in r.messages:
            print(f"        {m}")

    print(_c("  " + line, C.DIM))
    pct = payload["score_pct"]
    print(f"  Base score  {_c(_bar(pct), C.CYAN)}  "
          f"{_c(f'{payload['base_earned']:g}/{payload['base_total']:g}', C.BOLD)}  ({pct:g}%)")
    if payload["bonus_earned"]:
        print(f"  Bonus       {_c(f'+{payload['bonus_earned']:g}', C.BLUE)}")
    print(_c("  " + line, C.DIM))
    what = "repo (fresh import)" if repo else "notebook (Restart & Run All)"
    print(_c("  This is a SCORE PREVIEW. The grade of record is the instructor's", C.DIM))
    print(_c(f"  batch re-grade of your submitted {what}.", C.DIM))
    print()
