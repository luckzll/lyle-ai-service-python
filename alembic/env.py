from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from app.core.config import settings

config = context.config
sqlalchemy_url = settings.postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1)
config.set_main_option("sqlalchemy.url", sqlalchemy_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def run_migrations_offline() -> None:
    context.configure(
        url=sqlalchemy_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(sqlalchemy_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        # Alembic defaults version_num to VARCHAR(32), while this project already
        # contains descriptive revision ids longer than 32 characters.
        connection.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS alembic_version (
              version_num VARCHAR(128) NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            ALTER TABLE alembic_version
            ALTER COLUMN version_num TYPE VARCHAR(128)
            """
        )
        connection.commit()
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
