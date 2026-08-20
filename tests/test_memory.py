from travel_planner.memory import MemoryPolicy, MemoryStore


def test_layered_memory_uses_exact_keys_and_dual_timestamps(tmp_path):
    store = MemoryStore(tmp_path, window_size=2)
    store.save_turn(
        "user-1",
        profile_updates={"pace": "slow"},
        summary="new summary",
        skill_results={"beijing_family_trip": "outputs/beijing.md"},
        event_at="2026-08-02T00:00:00Z",
        recorded_at="2026-08-02T01:00:00Z",
    )
    store.save_turn(
        "user-1",
        profile_updates={"pace": "fast"},
        summary="late old summary",
        event_at="2026-08-01T00:00:00Z",
        recorded_at="2026-08-03T00:00:00Z",
    )
    store.save_turn(
        "user-1",
        profile_updates={"pace": "relaxed"},
        event_at="2026-08-02T00:00:00Z",
        recorded_at="2026-08-02T02:00:00Z",
    )

    layers = store.load_layers(
        "user-1",
        {"session_id": "throw-away", "channel": "web", "requested_skills": ["beijing_family_trip"]},
        [{"role": "user", "content": "old"}, {"role": "assistant", "content": "middle"}, {"role": "user", "content": "now"}],
    )

    assert layers["profile"] == {"pace": "relaxed"}
    assert layers["recent_summary"] == "new summary"
    assert layers["sliding_window"] == ["assistant: middle", "user: now"]
    assert layers["skills"]["beijing_family_trip"]["path"] == "outputs/beijing.md"
    assert store.get_skill("user-1", "beijing") is None
    assert store.read_view("user-2")["profile"] == {}
    assert "throw-away" not in store._ledger_path("user-1").read_text(encoding="utf-8")
    assert "throw-away" not in MemoryPolicy.render(layers, "writer")
