"""Lightweight broker client shared by API and worker, with bounded transport waits."""
import os
from celery import Celery

celery_app = Celery("xai_tasks", broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"))
celery_app.conf.update(
    broker_connection_timeout=2,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"socket_timeout": 2, "socket_connect_timeout": 2, "max_retries": 0},
    task_publish_retry=False,
    task_ignore_result=True,
    worker_prefetch_multiplier=1,
)


def check_broker():
    try:
        with celery_app.connection_for_write() as connection:
            connection.ensure_connection(max_retries=0)
            # Opening the channel establishes the Redis connection rather than inspecting config.
            channel = connection.default_channel
            if hasattr(channel, "client"):
                channel.client.ping()
        return True
    except Exception:
        return False
