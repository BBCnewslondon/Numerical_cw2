import os
import shutil
import tempfile

from cw import create_all_plots


def test_create_all_plots_fast_mode():
    tmpdir = tempfile.mkdtemp(prefix="plots_test_")
    try:
        create_all_plots(outdir=tmpdir, run_full=False)
        # Check some expected files exist
        expected_files = [
            "scenarioA_trajectory.png",
            "scenarioB_terrain.png",
            "scenarioB_trajectory.png",
            "scenarioC_terrain.png",
            "scenarioC_shot1.png",
            "scenarioC_shot2.png",
        ]
        for fname in expected_files:
            assert os.path.exists(os.path.join(tmpdir, fname)), f"Expected {fname} to exist"
    finally:
        shutil.rmtree(tmpdir)
