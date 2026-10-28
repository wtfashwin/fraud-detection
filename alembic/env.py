from logging.config import fileConfig
import os
from sqlalchemy import engine_from_config
from sqlalchemy.engine import URL
from sqlalchemy import pool
from alembic import context

config = context.config

fileConfig(config.config_file_name)

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.models import Base
from db.config import database_url

db_url = database_url()
if not isinstance(db_url, URL):
    config.set_main_option('sqlalchemy.url', db_url.replace('%', '%%'))
    db_url = config.get_main_option('sqlalchemy.url')

target_metadata = Base.metadata

def run_migrations_offline():
    context.configure(
        url=db_url, target_metadata=target_metadata, literal_binds=True
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    options = config.get_section(config.config_ini_section)
    options['sqlalchemy.url'] = db_url
    connectable = engine_from_config(
        options,
        prefix='sqlalchemy.',
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
