from pathlib import Path
import json
import os
import sys
import tempfile

import nbformat
from nbclient import NotebookClient


def test_all_notebooks_run_all() -> None:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temporary:
        kernel = Path(temporary) / "kernels" / "sstbr-test"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "SST-BR test",
            "language": "python",
        }), encoding="utf-8")
        previous = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = temporary
        try:
            for path in sorted((root / "notebooks").glob("*.ipynb")):
                book = nbformat.read(path, as_version=4)
                NotebookClient(book, timeout=180, kernel_name="sstbr-test", resources={"metadata": {"path": str(root)}}).execute()
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous
