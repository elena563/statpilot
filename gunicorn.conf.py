import logging
import os

from services.session import init_cleanup

workers = int(os.environ.get("GUNICORN_WORKERS", 1))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", 120))
bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:" + os.environ.get("PORT", "5000"))

logger = logging.getLogger("gunicorn.error")


def on_starting(server):
    try:
        init_cleanup()
        logger.info("Cleanup thread started via on_starting hook")
    except Exception as e:
        logger.warning("on_starting cleanup hook failed: %s", e)
