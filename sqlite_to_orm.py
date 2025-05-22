"""Convert a SQLite schema to an ORM schema."""

from __future__ import annotations

import textwrap
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import doctyper
import sqlglot
from sqlglot.expressions import (
    CheckColumnConstraint,
    ColumnDef,
    Constraint,
    Create,
    DataType,
    ForeignKey,
    NotNullColumnConstraint,
    PrimaryKey,
    UniqueColumnConstraint,
)

if TYPE_CHECKING:
    from collections.abc import Callable


__version__ = "1.0.35"

MAX_LINE_LENGTH = 100


def parse_varchar(kind: DataType) -> tuple[str, str]:
    """Parse VARCHAR(200) to String(200)."""
    if len(kind.expressions) != 1:
        raise ValueError(f"Invalid varchar args: {kind.expressions}")
    length: str = kind.expressions[0].this.this
    if not length.isdigit():
        raise ValueError(f"Invalid varchar args: {length}")
    return "str", f"String({length})"


def parse_decimal(kind: DataType) -> tuple[str, str]:
    """Parse DECIMAL to Numeric() and DECIMAL(10, 2) to Numeric(10, 2)."""
    if not kind.expressions:
        return "float", ""
    if len(kind.expressions) != 2:  # noqa: PLR2004
        raise ValueError(f"Invalid decimal args: {kind.expressions}")
    prec: str = kind.expressions[0].this.this
    scale: str = kind.expressions[1].this.this
    if not prec.isdigit() or not scale.isdigit():
        raise ValueError(f"Invalid decimal args: {prec}, {scale}")
    return "float", f"Numeric({prec}, {scale})"


SQL_TYPE_MAP: dict[DataType.Type, tuple[str, str] | Callable[[DataType], tuple[str, str]]] = {
    DataType.Type.VARCHAR: parse_varchar,
    DataType.Type.TEXT: ("str", "Text()"),
    DataType.Type.SMALLINT: ("int", "SmallInteger()"),
    DataType.Type.BIGINT: ("int", "BigInteger()"),
    DataType.Type.INT: ("int", ""),
    DataType.Type.DECIMAL: parse_decimal,
    DataType.Type.DATETIME: ("datetime", ""),
}


def convert_sqlite_to_orm(input_path: Path, output_path: Path, dialect: str = "sqlite") -> None:  # noqa: C901, PLR0912, PLR0915
    """Convert a SQLite database to an ORM schema."""
    expressions = sqlglot.parse(input_path.read_text(), read=dialect)
    if not all(isinstance(node, Create) for node in expressions):
        raise ValueError("Input file must contain only CREATE TABLE statements.")

    header = textwrap.dedent(f'''\
    """ORM schema."""
    # ruff: noqa: E501

    from __future__ import annotations

    from datetime import datetime  # noqa: TC003

    from sqlalchemy import (
        BigInteger,
        CheckConstraint,
        ForeignKeyConstraint,
        Index,
        Numeric,
        PrimaryKeyConstraint,
        SmallInteger,
        String,
        Text,
        UniqueConstraint,
    )
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

    __version__ = "{__version__}"


    class Base(DeclarativeBase):
        """ORM base class."""


    ''')
    lines = []

    if not {expr.kind for expr in expressions}.issubset({"TABLE", "INDEX"}):
        raise ValueError("Input file must contain only CREATE TABLE and CREATE INDEX statements.")

    indexes = [expr for expr in expressions if expr.kind == "INDEX"]
    tables = [expr for expr in expressions if expr.kind == "TABLE"]

    table_indexes: dict[str, list[str]] = {}

    for expr in indexes:
        index_name: str = expr.this.this.name
        unique = expr.args.get("unique", False)
        unique_str = ", unique=True" if unique else ""
        table: str = expr.this.args["table"].this.this
        cols = expr.this.args["params"].args["columns"]
        col_names = [f'"{col.this.this}"' for col in cols]
        index = f'Index("{index_name}", {", ".join(col_names)}{unique_str})'
        table_indexes.setdefault(table, []).append(index)

    for expr in tables:
        table: str = expr.this.this.name
        if table.startswith("sqlite_"):
            continue
        class_name = "".join(word.capitalize() for word in table.split("_"))
        lines.append(f"class {class_name}(Base):")
        lines.append(f'    """ORM class for {table}."""')
        lines.append("")
        lines.append(f'    __tablename__ = "{table}"')

        if not all(isinstance(expr, (ColumnDef, Constraint)) for expr in expr.expressions):
            raise ValueError("Invalid table definition: only columns and constraints allowed")

        primary_key: list[str] = []
        table_args: list[str] = []

        for const in expr.find_all(Constraint):
            if len(const.expressions) != 1:
                raise ValueError("Invalid constraint definition: only single expression allowed")
            name = const.this.this
            this = const.expressions[0]
            if isinstance(this, PrimaryKey):
                if primary_key:
                    raise ValueError("Multiple primary keys found")
                primary_key.extend(col.this.this.this for col in this.expressions)
                quoted = [f'"{col}"' for col in primary_key]
                table_args.append(f'PrimaryKeyConstraint({", ".join(quoted)}, name="{name}")')
            elif isinstance(this, UniqueColumnConstraint):
                cols: list[str] = [f'"{ident.this}"' for ident in this.this.expressions]
                table_args.append(f'UniqueConstraint({", ".join(cols)}, name="{name}")')
            elif isinstance(this, CheckColumnConstraint):
                if not len(const.expressions) == 1:
                    raise ValueError("Only one column allowed in check constraint")
                const_expr = const.expressions[0]
                table_args.append(f'CheckConstraint("{const_expr.this}", name="{name}")')
            elif isinstance(this, ForeignKey):
                curr_cols = [f'"{ident.this}"' for ident in this.expressions]
                head_ref = this.args["reference"]
                opts = head_ref.args["options"]
                ref = head_ref.this
                ref_table: str = ref.this.this.this
                ref_cols = [f'"{ref_table}.{ident.this}"' for ident in ref.expressions]
                option_str = ""
                for opt in opts:
                    if opt == "ON DELETE CASCADE":
                        option_str += ', ondelete="CASCADE"'
                    else:
                        raise ValueError(f"Unknown option {opt}")
                table_args.append(
                    f'ForeignKeyConstraint([{", ".join(curr_cols)}], [{", ".join(ref_cols)}], name="{name}"{option_str})'  # noqa: E501
                )
            else:
                raise TypeError(f"Unknown constraint {this}")

        table_args.extend(table_indexes.get(table, []))

        if len(", ".join(table_args)) + 23 > MAX_LINE_LENGTH:
            lines.append("    __table_args__ = (")
            lines.extend(f"        {line}," for line in table_args)
            lines.append("    )")
        elif table_args:
            lines.append(f"    __table_args__ = ({', '.join(table_args)},)")

        for col in expr.find_all(ColumnDef):
            col_name = col.name
            if col.kind.this not in SQL_TYPE_MAP:
                raise TypeError(f"Unknown column type {col.kind}")
            type_cb = SQL_TYPE_MAP[col.kind.this]
            if callable(type_cb):
                map_python, map_alchemy = type_cb(col.kind)
            else:
                if col.kind.expressions:
                    raise ValueError("Simple mapping types should not have expressions")
                map_python, map_alchemy = type_cb
            can_be_null = True
            for col_const in col.constraints:
                if isinstance(col_const.kind, NotNullColumnConstraint):
                    can_be_null = False
                else:
                    raise TypeError("Unknown column constraint type")

            if map_alchemy:
                map_alchemy = f" = mapped_column({map_alchemy})"

            optional_suffix = " | None" if can_be_null and col_name not in primary_key else ""
            col_def = f"{col_name}: Mapped[{map_python}{optional_suffix}]{map_alchemy}"
            lines.append(f"    {col_def}")

        lines.append("")  # spacing
        lines.append("")

    output_path.write_text(header + "\n".join(lines))


if __name__ == "__main__":
    app = doctyper.SlimTyper()
    app.command()(convert_sqlite_to_orm)
    app()
