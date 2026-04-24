from back.worker import loop as worker_loop


def test_worker_module_exposes_public_api() -> None:
    assert callable(worker_loop.claim_pending)
    assert callable(worker_loop.finalize)
    assert callable(worker_loop.execute)
    assert callable(worker_loop.loop)
    assert callable(worker_loop.main)


def test_worker_constants() -> None:
    assert worker_loop.POLL_INTERVAL_SECONDS > 0
    assert worker_loop.STDOUT_TAIL_LIMIT >= 1024


def test_execute_uses_shlex_split_on_command() -> None:
    import inspect

    src = inspect.getsource(worker_loop.execute)
    assert "shlex.split(run.command)" in src
    assert "cwd=str(settings.research_root)" in src
