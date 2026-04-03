from eqsys.tracker import DependencyTracker


def test_tracker_inactive_by_default():
    t = DependencyTracker()
    assert not t.active


def test_tracker_start_stop():
    t = DependencyTracker()
    t.start()
    assert t.active
    deps = t.stop()
    assert not t.active
    assert deps == set()


def test_tracker_register_while_active():
    t = DependencyTracker()
    t.start()
    t.register("P1")
    t.register("P2")
    t.register("P1")  # duplicate — should be in set only once
    deps = t.stop()
    assert deps == {"P1", "P2"}


def test_tracker_register_while_inactive():
    """Register is a no-op when tracker is inactive."""
    t = DependencyTracker()
    t.register("P1")
    t.start()
    deps = t.stop()
    assert deps == set()


def test_tracker_reset_on_start():
    """Each start() clears previous deps."""
    t = DependencyTracker()
    t.start()
    t.register("P1")
    t.stop()

    t.start()
    deps = t.stop()
    assert deps == set()
