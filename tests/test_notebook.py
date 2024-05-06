import os

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


class TestDeCompletion:
    """
    Testing that the notebook runs to completion

    """

    def test_1(self):
        f = os.path.dirname(os.path.realpath(__file__)) + "/../scripts/Tutorial.ipynb"
        run_path = os.path.dirname(f)
        with open(f) as file:
            nb = nbformat.read(file, as_version=4)
            ep = ExecutePreprocessor(kernel_name="python3")
            ep.preprocess(nb, {"metadata": {"path": run_path}})
