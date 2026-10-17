"""Opt-in PostgreSQL ownership tests; synthetic calculators and manual deliveries.

No Redis, serving models, real transactions, or broker-redelivery claims. Each
test owns a temporary schema in the explicitly named local throwaway database.
"""
from dataclasses import dataclass
import multiprocessing
import os
import queue
import time
import uuid

import pytest
from sqlalchemy import create_engine, insert, select, text, update
from sqlalchemy.engine import make_url

from db.models import Explanation


WAIT = 35


def _engine(url, schema=None, application=None, isolation="READ COMMITTED"):
    options = {"connect_timeout": 3}
    if schema:
        options["options"] = f"-csearch_path={schema}"
    if application:
        options["application_name"] = application
    return create_engine(url, connect_args=options, pool_pre_ping=True,
                         pool_size=1, max_overflow=0, isolation_level=isolation)


def _task_process(url, schema, application, transaction_id, marker, entered,
                  release, done, output, calls, fail, collision, isolation):
    """Run the actual task against PostgreSQL; only calculation/Redis are controlled."""
    from celery.exceptions import Retry
    from api import worker

    engine = _engine(url, schema, application, isolation)
    worker.engine = engine
    if collision:
        worker._explanation_lock_key = lambda _transaction_id: 17

    class Calculator:
        explanation_columns = ["synthetic_feature"]

        def explain(self, _features):
            with calls.get_lock():
                calls.value += 1
            entered.set()
            if not release.wait(WAIT):
                raise TimeoutError("Synthetic calculation release was not signalled")
            if fail:
                raise ValueError("Synthetic calculation failure")
            return [float(marker)], 0.0, "synthetic_controlled"

    def scheduled_retry(**_kwargs):
        raise Retry("Synthetic scheduled retry; no broker publication")

    worker.get_bundle = lambda: Calculator()
    worker.compute_shap.retry = scheduled_retry
    worker.compute_shap.push_request(retries=0)
    try:
        result = worker.compute_shap.run(transaction_id, [float(marker)], "synthetic-correlation")
        output.put({"outcome": "completed", "result": result})
    except Retry:
        output.put({"outcome": "retry"})
    except BaseException as error:
        # Never send a DSN, password, traceback, or exception message to pytest.
        output.put({"outcome": "error", "type": type(error).__name__})
    finally:
        worker.compute_shap.pop_request()
        engine.dispose()
        done.set()


@dataclass
class Job:
    process: object
    application: str
    entered: object
    release: object
    done: object
    output: object


class Database:
    def __init__(self, url, schema, engine):
        self._url = url
        self.schema = schema
        self.engine = engine
        self.context = multiprocessing.get_context("spawn")
        self.calls = self.context.Value("i", 0)
        self.jobs = []

    def seed(self, transaction_id, status="PENDING", values=None):
        with self.engine.begin() as connection:
            connection.execute(insert(Explanation).values(
                transaction_id=transaction_id, correlation_id="synthetic-correlation",
                status=status, shap_values=values,
                feature_names=["synthetic_feature"] if values is not None else None))

    def row(self, transaction_id):
        with self.engine.connect() as connection:
            return dict(connection.execute(select(Explanation.__table__).where(
                Explanation.transaction_id == transaction_id)).mappings().one())

    def start(self, transaction_id, marker=1, *, fail=False, collision=False,
              isolation="READ COMMITTED", released=False):
        entered, release, done = [self.context.Event() for _ in range(3)]
        output = self.context.Queue()
        application = "fraud_test_" + uuid.uuid4().hex
        if released:
            release.set()
        process = self.context.Process(target=_task_process, args=(
            self._url, self.schema, application, transaction_id, marker, entered,
            release, done, output, self.calls, fail, collision, isolation))
        job = Job(process, application, entered, release, done, output)
        self.jobs.append(job)
        process.start()
        return job

    def finish(self, job):
        assert job.done.wait(WAIT), "Task did not finish within its bounded deadline"
        job.process.join(5)
        assert job.process.exitcode == 0, "Task subprocess did not exit cleanly"
        try:
            return job.output.get(timeout=3)
        except queue.Empty:
            pytest.fail("Task exited without its result receipt")

    def backend(self, job):
        with self.engine.connect() as connection:
            return connection.execute(text(
                "SELECT pid FROM pg_stat_activity WHERE application_name=:application"
            ), {"application": job.application}).scalar_one_or_none()

    def observe_wait_or_calculation(self, owner, duplicate):
        """Observe actual PG blocking or the competing calculation, never a sleep verdict."""
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            if duplicate.entered.is_set():
                return "calculated"
            if duplicate.done.is_set():
                return "finished_without_wait"
            with self.engine.connect() as connection:
                rows = connection.execute(text(
                    "SELECT a.pid, pg_blocking_pids(b.pid) AS blockers "
                    "FROM pg_stat_activity a CROSS JOIN pg_stat_activity b "
                    "WHERE a.application_name=:owner AND b.application_name=:duplicate"
                ), {"owner": owner.application, "duplicate": duplicate.application}).first()
            if rows is not None and rows.pid in rows.blockers:
                return "blocked"
            time.sleep(0.02)
        pytest.fail("Neither a competing calculation nor PostgreSQL blocking was observed")

    def cleanup(self):
        for job in self.jobs:
            # SIGKILL can strand Event waiters; signalling a dead job can hang teardown.
            if job.process.is_alive():
                job.release.set()
        for job in self.jobs:
            job.process.join(5)
            if job.process.is_alive():
                job.process.kill()
                job.process.join(5)
            job.output.close()
            job.output.join_thread()


@pytest.fixture
def pg_database():
    value = os.getenv("FRAUD_TEST_DATABASE_URL")
    if not value:
        pytest.skip("Set FRAUD_TEST_DATABASE_URL for local throwaway PostgreSQL tests")
    try:
        parsed = make_url(value)
    except Exception:
        pytest.fail("FRAUD_TEST_DATABASE_URL is not a valid database URL", pytrace=False)
    if parsed.query:
        pytest.fail("FRAUD_TEST_DATABASE_URL must not contain query parameters", pytrace=False)
    if (not parsed.drivername.startswith("postgresql") or
            parsed.host not in {"127.0.0.1", "localhost", "::1"} or
            parsed.database != "fraud_worker_tests"):
        pytest.fail("Integration tests require local PostgreSQL database fraud_worker_tests", pytrace=False)
    schema = "worker_test_" + uuid.uuid4().hex
    admin = _engine(value)
    scoped = None
    database = None
    try:
        with admin.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        scoped = _engine(value, schema)
        Explanation.__table__.create(scoped)
        database = Database(value, schema, scoped)
        yield database
    finally:
        if database is not None:
            database.cleanup()
        if scoped is not None:
            scoped.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin.dispose()


@pytest.mark.parametrize("query", [
    "host=remote.invalid",
    "hostaddr=203.0.113.1",
    "dbname=postgres",
    "service=external-test-service",
    "sslmode=require",
], ids=["host", "hostaddr", "dbname", "service", "other-option"])
def test_database_url_query_rejected_before_engine(monkeypatch, request, query):
    monkeypatch.setenv("FRAUD_TEST_DATABASE_URL",
                       "postgresql+psycopg2://synthetic:synthetic@127.0.0.1/"
                       "fraud_worker_tests?" + query)

    def forbidden_engine(*_args, **_kwargs):
        raise AssertionError("Engine creation reached before URL-query rejection")

    monkeypatch.setitem(globals(), "_engine", forbidden_engine)
    with pytest.raises(pytest.fail.Exception, match="query parameters"):
        request.getfixturevalue("pg_database")


@pytest.mark.parametrize("isolation", ["READ COMMITTED", "REPEATABLE READ", "AUTOCOMMIT"],
                         ids=["read-committed", "repeatable-read", "autocommit"])
def test_same_id_computes_once(pg_database, isolation):
    db = pg_database
    db.seed("same-id")
    owner = db.start("same-id", isolation=isolation)
    assert owner.entered.wait(WAIT), "Owner did not enter its calculation"
    duplicate = db.start("same-id", isolation=isolation)
    observed = db.observe_wait_or_calculation(owner, duplicate)
    assert observed == "blocked", f"Duplicate entered calculation while owner was held: {observed}"
    assert db.backend(owner) != db.backend(duplicate)
    assert db.calls.value == 1
    assert db.row("same-id")["status"] == "PENDING"
    owner.release.set()
    duplicate.release.set()
    assert db.finish(owner)["outcome"] == "completed"
    assert db.finish(duplicate)["outcome"] == "completed"
    assert db.calls.value == 1
    assert db.row("same-id")["shap_values"] == [1.0]


def test_completed_replay_does_not_recompute(pg_database):
    db = pg_database
    db.seed("completed", "COMPLETED", [7.0])
    job = db.start("completed", marker=99, released=True)
    assert db.finish(job)["outcome"] == "completed"
    assert db.calls.value == 0
    assert db.row("completed")["shap_values"] == [7.0]


def test_different_ids_compute_in_parallel(pg_database):
    db = pg_database
    db.seed("one")
    db.seed("two")
    first = db.start("one")
    assert first.entered.wait(WAIT)
    second = db.start("two", marker=2)
    assert second.entered.wait(WAIT), "Different transaction was serialized behind the owner"
    assert db.calls.value == 2
    first.release.set()
    second.release.set()
    assert db.finish(first)["outcome"] == db.finish(second)["outcome"] == "completed"
    assert db.row("one")["shap_values"] == [1.0]
    assert db.row("two")["shap_values"] == [2.0]


def test_hash_collision_waits_and_completes_both_ids(pg_database):
    db = pg_database
    db.seed("collision-one")
    db.seed("collision-two")
    first = db.start("collision-one", collision=True)
    assert first.entered.wait(WAIT)
    second = db.start("collision-two", marker=2, collision=True)
    assert db.observe_wait_or_calculation(first, second) == "blocked"
    first.release.set()
    second.release.set()
    assert db.finish(first)["outcome"] == db.finish(second)["outcome"] == "completed"
    assert db.calls.value == 2
    assert db.row("collision-one")["shap_values"] == [1.0]
    assert db.row("collision-two")["shap_values"] == [2.0]


def test_calculation_failure_rolls_back_before_retry_then_recovers(pg_database):
    db = pg_database
    db.seed("retry")
    failed = db.start("retry", fail=True, released=True)
    assert db.finish(failed)["outcome"] == "retry"
    assert db.row("retry")["status"] == "PENDING"
    recovery = db.start("retry", marker=2, released=True)
    assert db.finish(recovery)["outcome"] == "completed"
    assert db.row("retry")["shap_values"] == [2.0]


def test_killed_owner_releases_lock_for_manual_replay(pg_database):
    db = pg_database
    db.seed("killed")
    owner = db.start("killed")
    assert owner.entered.wait(WAIT)
    owner.process.kill()
    owner.process.join(5)
    assert not owner.process.is_alive()
    assert db.row("killed")["status"] == "PENDING"
    replay = db.start("killed", marker=2, released=True)
    assert db.finish(replay)["outcome"] == "completed"
    assert db.calls.value == 2
    assert db.row("killed")["shap_values"] == [2.0]


def test_lost_database_session_cannot_save_stale_result(pg_database):
    db = pg_database
    db.seed("lost-session")
    old = db.start("lost-session", marker=1)
    assert old.entered.wait(WAIT)
    backend = db.backend(old)
    assert backend is not None
    with db.engine.begin() as connection:
        assert connection.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": backend}).scalar()
    recovery = db.start("lost-session", marker=2, released=True)
    assert db.finish(recovery)["outcome"] == "completed"
    assert db.row("lost-session")["shap_values"] == [2.0]
    old.release.set()
    assert db.finish(old)["outcome"] == "retry"
    assert db.calls.value == 2  # Session loss can permit overlapping CPU work.
    assert db.row("lost-session")["status"] == "COMPLETED"
    assert db.row("lost-session")["shap_values"] == [2.0]


def test_dispatch_failure_status_update_does_not_wait_for_calculation(pg_database):
    db = pg_database
    db.seed("dispatch")
    owner = db.start("dispatch")
    assert owner.entered.wait(WAIT)
    with db.engine.begin() as connection:
        connection.execute(text("SET LOCAL lock_timeout = '750ms'"))
        changed = connection.execute(update(Explanation).where(
            Explanation.transaction_id == "dispatch", Explanation.status == "PENDING"
        ).values(status="FAILED"))
        assert changed.rowcount == 1
    assert not owner.release.is_set()
    assert db.row("dispatch")["status"] == "FAILED"
    owner.release.set()
    assert db.finish(owner)["outcome"] == "completed"
    assert db.row("dispatch")["status"] == "COMPLETED"
