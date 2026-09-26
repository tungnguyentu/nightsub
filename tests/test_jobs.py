from autosub import jobs


def test_roundtrip(tmp_path):
    db = str(tmp_path / "j.db")
    i = jobs.add(db, "/v/a.mp4", "en")
    jobs.update(db, i, stage="gated", stage_times={"gate": 1.5})
    j = jobs.get(db, i)
    assert (j["stage"], j["stage_times"], j["error"]) == ("gated", {"gate": 1.5}, None)
    jobs.fail(db, i, "boom")
    assert jobs.at_stage(db, "gated") == [] and jobs.get(db, i)["error"] == "boom"
