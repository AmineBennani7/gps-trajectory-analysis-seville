import os

from gps_seville.env import load_env_file


def test_env_file_does_not_override(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\nA_TEST_KEY='one'\nB_TEST_KEY=two=2\n\nnot a line\n")
    monkeypatch.delenv("A_TEST_KEY", raising=False)
    monkeypatch.setenv("B_TEST_KEY", "kept")
    assert load_env_file(f) == ["A_TEST_KEY"]
    assert os.environ["A_TEST_KEY"] == "one"
    assert os.environ["B_TEST_KEY"] == "kept"
    monkeypatch.delenv("A_TEST_KEY")


def test_missing_env_file(tmp_path):
    assert load_env_file(tmp_path / "nope") == []
