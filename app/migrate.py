"""
Lightweight schema reconciliation for SQLite.

Why this exists
---------------
`Base.metadata.create_all()` creates tables that do not exist. It does NOT
alter tables that already exist with a different shape. The repository
shipped a committed `landslide.db` whose `alerts` table still had the old
`location_name` column, while `app/models.py` had moved on to
`corridor_name` plus a required `risk_score`. Every insert therefore failed
with "table alerts has no column named corridor_name" -- on a database that
looked perfectly healthy.

This module reconciles the two on startup by adding any columns the models
declare but the table lacks. It is intentionally minimal: SQLite can add
columns but cannot drop or retype them, so this handles the common case and
leaves anything harder to a real migration tool.

For production, use Alembic. For a hackathon deployment on Render with a
SQLite file, this is the right amount of machinery.
"""

from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateTable

from app.database import Base

log = logging.getLogger(__name__)

# SQLite cannot add a NOT NULL column without a default, so we supply one.
_SQLITE_DEFAULTS = {
    "INTEGER": "0",
    "FLOAT": "0.0",
    "REAL": "0.0",
    "VARCHAR": "''",
    "TEXT": "''",
    "DATETIME": "CURRENT_TIMESTAMP",
}


def _orphan_required_columns(inspector, table_name: str, table) -> list[str]:
    """Columns the DB requires but the models no longer write.

    A legacy NOT NULL column with no default blocks every insert, because
    SQLAlchemy never supplies a value for a column the model does not know
    about. Adding the new columns is not enough -- the old one has to go.
    """
    model_cols = {c.name for c in table.columns}
    orphans = []
    for col in inspector.get_columns(table_name):
        if col["name"] in model_cols:
            continue
        if not col.get("nullable", True) and col.get("default") is None:
            orphans.append(col["name"])
    return orphans


# Legacy column -> current column, per table. Without this a rebuild would
# discard the data in a renamed column. Keep it explicit rather than trying
# to guess renames from column types.
RENAMES: dict[str, dict[str, str]] = {
    "alerts": {"location_name": "corridor_name"},
}


def _rebuild_table(engine: Engine, table_name: str, table) -> None:
    """Rebuild a table to match the model, preserving what data it can.

    SQLite's ALTER TABLE cannot drop a constrained column, so the standard
    remedy is create-copy-drop-rename inside one transaction.

    Each model column is sourced, in order of preference, from: the same
    column in the old table; a legacy column named in RENAMES; a type
    default when the column is required; otherwise NULL.
    """
    inspector = inspect(engine)
    db_cols = {c["name"] for c in inspector.get_columns(table_name)}
    renames = RENAMES.get(table_name, {})
    # current name -> legacy name
    reverse = {new: old for old, new in renames.items()}

    target_cols: list[str] = []
    select_exprs: list[str] = []

    for column in table.columns:
        name = column.name
        if name in db_cols:
            source = name
        elif name in reverse and reverse[name] in db_cols:
            source = reverse[name]
        elif not column.nullable:
            base = column.type.compile(dialect=engine.dialect).split("(")[0].upper()
            source = _SQLITE_DEFAULTS.get(base, "''")
        else:
            continue  # nullable and unsourceable: let it default to NULL
        target_cols.append(name)
        select_exprs.append(source)

    shared = target_cols
    col_list = ", ".join(target_cols)
    select_list = ", ".join(select_exprs)

    tmp = f"{table_name}__rebuild"
    # CreateTable produces the real DDL; table.compile() only yields the name.
    ddl = str(CreateTable(table).compile(engine)).strip()
    ddl_tmp = ddl.replace(f"CREATE TABLE {table_name}", f"CREATE TABLE {tmp}", 1)
    if f"CREATE TABLE {tmp}" not in ddl_tmp:
        raise RuntimeError(f"could not retarget DDL for {table_name}")

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {tmp}"))
        conn.execute(text(ddl_tmp))
        if shared:
            conn.execute(
                text(f"INSERT INTO {tmp} ({col_list}) "
                     f"SELECT {select_list} FROM {table_name}")
            )
        conn.execute(text(f"DROP TABLE {table_name}"))
        conn.execute(text(f"ALTER TABLE {tmp} RENAME TO {table_name}"))

    # Indexes live outside CREATE TABLE and are lost with the old table.
    for index in table.indexes:
        try:
            index.create(bind=engine, checkfirst=True)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("could not recreate index %s: %s", index.name, exc)


def reconcile_schema(engine: Engine) -> list[str]:
    """Bring existing tables into line with the models.

    Two repairs, in order:
      1. rebuild any table holding a legacy required column the models no
         longer write (otherwise every insert fails)
      2. add any column the models declare but the table lacks

    Returns a list of human-readable changes applied, so startup can log
    exactly what it did rather than changing the database silently.
    """
    changes: list[str] = []
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    # Pass 1: rebuild tables with orphaned required columns.
    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            continue
        orphans = _orphan_required_columns(inspector, table_name, table)
        if not orphans:
            continue
        try:
            _rebuild_table(engine, table_name, table)
            changes.append(
                f"{table_name}: rebuilt, dropped legacy required column(s) "
                f"{', '.join(orphans)}"
            )
            log.warning("schema reconciled: %s", changes[-1])
        except Exception as exc:  # pragma: no cover - defensive
            log.error("could not rebuild %s: %s", table_name, exc)

    # Pass 2: add missing columns (re-inspect, pass 1 may have changed things).
    inspector = inspect(engine)
    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            continue  # create_all() will handle it

        existing_cols = {c["name"] for c in inspector.get_columns(table_name)}

        for column in table.columns:
            if column.name in existing_cols:
                continue

            col_type = column.type.compile(dialect=engine.dialect)
            base_type = col_type.split("(")[0].upper()

            clause = f"{column.name} {col_type}"
            if not column.nullable:
                default = _SQLITE_DEFAULTS.get(base_type, "''")
                clause += f" NOT NULL DEFAULT {default}"

            stmt = f"ALTER TABLE {table_name} ADD COLUMN {clause}"
            try:
                with engine.begin() as conn:
                    conn.execute(text(stmt))
                changes.append(f"{table_name}.{column.name} ({col_type})")
                log.warning("schema reconciled: added %s", changes[-1])
            except Exception as exc:  # pragma: no cover - defensive
                log.error("could not add %s.%s: %s", table_name, column.name, exc)

    return changes
