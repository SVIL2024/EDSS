"""Generated manifests preserve normal labels and frame-annotation order."""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_ucf_normals_follow_anomalies_and_have_normal_label(tmp_path):
    root = tmp_path / "features"
    normal = "Testing_Normal_Videos_Anomaly/Normal_Videos_003_x264"
    anomaly = "Vandalism/Vandalism007_x264"
    split = tmp_path / "split.txt"
    split.write_text(f"{normal}.mp4\n{anomaly}.mp4\n")
    for name in [normal, anomaly]:
        feature = root / (name + "__5.npy")
        feature.parent.mkdir(parents=True, exist_ok=True)
        feature.touch()
    output = tmp_path / "features.csv"
    subprocess.run([
        sys.executable, str(ROOT / "list/make_list_ucf.py"),
        "--feature-root", str(root), "--split", str(split),
        "--indices", "5", "--output", str(output),
    ], check=True, capture_output=True)
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["label"] for row in rows] == ["Vandalism", "Normal"]
    assert [Path(row["path"]).name for row in rows] == [
        "Vandalism007_x264__5.npy", "Normal_Videos_003_x264__5.npy"]


@pytest.mark.parametrize("test_features", [False, True])
def test_xd_order_and_test_feature_index(tmp_path, test_features):
    names = ["A.movie_label_A__0.npy", "C.movie_label_B2-G-0__0.npy",
             "B.movie_label_B1-0-0__0.npy", "B.movie_label_B1-0-0__1.npy"]
    for name in names:
        (tmp_path / name).touch()
    output = tmp_path / "features.csv"
    arguments = [
        sys.executable, str(ROOT / "list/make_list_xd.py"),
        "--feature-root", str(tmp_path), "--output", str(output),
    ]
    if test_features:
        arguments += ["--indices", "0"]
    subprocess.run(arguments, check=True, capture_output=True)
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    expected = [names[2], names[1], names[0]] if test_features else [
        names[2], names[3], names[1], names[0]]
    assert [Path(row["path"]).name for row in rows] == expected
    assert rows[-1]["label"] == "A"
