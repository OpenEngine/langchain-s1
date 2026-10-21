import runpy
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_quickstart_runs_without_credentials(capsys):
    runpy.run_path(str(EXAMPLES / "quickstart.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert out.count("route = billing") == 2
    assert "gated: {}" in out
