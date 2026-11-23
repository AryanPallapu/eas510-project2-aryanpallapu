"""Generate the project notebooks from cell definitions, then execute them in place.

    python scripts/build_notebooks.py            # build + execute all
    python scripts/build_notebooks.py 01 03      # only these
"""
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nb_eda  # noqa: E402

NOTEBOOKS = {"01": ("01_ai4i_eda.ipynb", lambda: nb_eda.AI4I),
             "03": ("03_aps_eda.ipynb", lambda: nb_eda.APS)}
try:
    import nb_modeling  # noqa: E402
    NOTEBOOKS.update({"02": ("02_ai4i_modeling.ipynb", lambda: nb_modeling.cells("ai4i")),
                      "04": ("04_aps_modeling.ipynb", lambda: nb_modeling.cells("aps")),
                      "05": ("05_comparative_analysis.ipynb", lambda: nb_modeling.COMPARE)})
except ImportError:
    pass


def build(key):
    fname, get = NOTEBOOKS[key]
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in get()]
    path = ROOT / "notebooks" / fname
    nbf.write(nb, path)
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                    "--inplace", "--ExecutePreprocessor.timeout=3600", str(path)],
                   check=True, cwd=ROOT / "notebooks")
    print("executed", path)


if __name__ == "__main__":
    for k in (sys.argv[1:] or sorted(NOTEBOOKS)):
        build(k)
