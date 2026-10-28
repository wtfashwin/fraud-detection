"""Run the real local MLflow checks in a separate process to isolate global state."""
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


EXPECTED_CASES = 31


def _report_path(basetemp):
    return basetemp.parent / f"{basetemp.name}-junit.xml"


def _run_isolated_pytest(case_file, basetemp):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("MLFLOW_") and key not in {"PYTEST_ADDOPTS", "PYTEST_PLUGINS"}}
    env["MLFLOW_ENABLE_TELEMETRY"] = "false"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
         f"--basetemp={basetemp}", f"--junitxml={_report_path(basetemp)}", str(case_file)],
        cwd=Path(__file__).resolve().parents[1], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
    )


def _require_complete_report(result, report_path):
    if result.returncode != 0:
        raise AssertionError("Child pytest failed:\n" + result.stdout)
    try:
        root = ET.parse(report_path).getroot()
        suites = root.findall("testsuite")
        if root.tag != "testsuites" or len(suites) != 1:
            raise ValueError("expected one test suite")
        suite = suites[0]
        counts = {name: int(suite.attrib[name]) for name in ("tests", "failures", "errors", "skipped")}
        if counts != {"tests": EXPECTED_CASES, "failures": 0, "errors": 0, "skipped": 0}:
            raise ValueError(f"unexpected case counts: {counts}")
        cases = suite.findall("testcase")
        identities = {(case.attrib["classname"], case.attrib["name"]) for case in cases}
        if len(cases) != EXPECTED_CASES or len(identities) != EXPECTED_CASES:
            raise ValueError("case count or identities do not match the receipt")
        if any(case.find(kind) is not None for case in cases for kind in ("skipped", "failure", "error")):
            raise ValueError("the receipt contains a nonpassing case")
    except (OSError, ET.ParseError, KeyError, ValueError) as exc:
        raise AssertionError(f"Incomplete child pytest receipt: {exc}\n{result.stdout}") from exc


def test_auc_contract_in_isolated_local_mlflow_process(tmp_path):
    cases = Path(__file__).with_name("mlflow_auc_integration_cases.py")
    basetemp = tmp_path / "pytest-inner"
    result = _run_isolated_pytest(cases, basetemp)
    _require_complete_report(result, _report_path(basetemp))


@pytest.mark.parametrize("kind", ["missing", "malformed", "skipped", "incomplete", "contradictory", "hidden-failure"])
def test_rejects_incomplete_inner_report(tmp_path, kind):
    report = tmp_path / "inner.xml"
    if kind == "malformed":
        report.write_text("<testsuites>")
    elif kind not in {"missing", "malformed"}:
        count = 30 if kind == "incomplete" else EXPECTED_CASES
        entries = [f'<testcase classname="test" name="case-{i}"/>' for i in range(count)]
        if kind == "skipped":
            entries[0] = '<testcase classname="test" name="case-0"><skipped/></testcase>'
        if kind == "contradictory":
            entries.pop()
        if kind == "hidden-failure":
            entries[0] = '<testcase classname="test" name="case-0"><failure/></testcase>'
        report.write_text(
            f'<testsuites><testsuite tests="{count}" failures="0" errors="0" skipped="{int(kind == "skipped")}">'
            + "".join(entries) + "</testsuite></testsuites>"
        )
    child = subprocess.CompletedProcess([], 0, stdout="child diagnostics")
    with pytest.raises(AssertionError, match="child diagnostics"):
        _require_complete_report(child, report)


def test_accepts_complete_inner_report(tmp_path):
    report = tmp_path / "inner.xml"
    entries = [f'<testcase classname="test" name="case-{i}"/>' for i in range(EXPECTED_CASES)]
    report.write_text(
        f'<testsuites><testsuite tests="{EXPECTED_CASES}" failures="0" errors="0" skipped="0">'
        + "".join(entries) + "</testsuite></testsuites>"
    )
    _require_complete_report(subprocess.CompletedProcess([], 0, stdout=""), report)
