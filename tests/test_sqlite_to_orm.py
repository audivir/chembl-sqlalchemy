"""Tests for the `sqlite_to_orm` schema conversion script."""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import sqlglot
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKeyConstraint,
    Numeric,
    SmallInteger,
    UniqueConstraint,
)
from sqlalchemy import String as SaString
from sqlalchemy import Text as SaText
from sqlalchemy.types import DateTime, Integer
from sqlglot.expressions import ColumnDef
from sqlite_to_orm import convert_sqlite_to_orm, parse_decimal, parse_varchar

if TYPE_CHECKING:
    from sqlglot.expressions import DataType

FIXTURE_SCHEMA = Path("tests/fixtures/sample_schema.sql")


def _column_kind(sql: str) -> DataType:
    stmt = sqlglot.parse_one(sql, read="sqlite")
    kind = next(stmt.find_all(ColumnDef)).kind
    assert kind is not None
    return kind


class TestParseVarchar:
    def test_parses_length(self) -> None:
        kind = _column_kind("CREATE TABLE t (c VARCHAR(50))")
        assert parse_varchar(kind) == ("str", "String(50)")

    def test_rejects_multiple_args(self) -> None:
        with pytest.raises(ValueError, match="Invalid varchar args"):
            parse_varchar(_column_kind("CREATE TABLE t (c VARCHAR(1, 2))"))

    def test_rejects_non_digit_length(self) -> None:
        with pytest.raises(ValueError, match="Invalid varchar args"):
            parse_varchar(_column_kind("CREATE TABLE t (c VARCHAR(abc))"))


class TestParseDecimal:
    def test_parses_precision_and_scale(self) -> None:
        assert parse_decimal(_column_kind("CREATE TABLE t (c DECIMAL(10, 2))")) == (
            "float",
            "Numeric(10, 2)",
        )

    def test_bare_decimal_has_no_args(self) -> None:
        assert parse_decimal(_column_kind("CREATE TABLE t (c DECIMAL)")) == ("float", "")

    def test_rejects_wrong_arg_count(self) -> None:
        with pytest.raises(ValueError, match="Invalid decimal args"):
            parse_decimal(_column_kind("CREATE TABLE t (c DECIMAL(10, 2, 3))"))

    def test_rejects_non_digit_precision(self) -> None:
        with pytest.raises(ValueError, match="Invalid decimal args"):
            parse_decimal(_column_kind("CREATE TABLE t (c DECIMAL(a, b))"))


def _convert(tmp_path: Path, sql: str) -> dict[str, object]:
    input_path = tmp_path / "schema.sql"
    output_path = tmp_path / "orm.py"
    input_path.write_text(sql)
    convert_sqlite_to_orm(input_path, output_path, chembl_version="test")
    module_name = f"generated_orm_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, output_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[module_name]
    return vars(module)


class TestConvertSqliteToOrm:
    def test_generates_orm_classes_from_fixture_schema(self, tmp_path: Path) -> None:
        namespace = _convert(tmp_path, FIXTURE_SCHEMA.read_text())

        assert set(namespace["Base"].metadata.tables) == {  # type: ignore[attr-defined]
            "simple_table",
            "constrained_table",
        }
        assert namespace["__doc__"] == "ORM schema for ChEMBL test."

    def test_single_constraint_table_uses_inline_table_args(self, tmp_path: Path) -> None:
        namespace = _convert(tmp_path, FIXTURE_SCHEMA.read_text())
        table = namespace["Base"].metadata.tables["simple_table"]  # type: ignore[attr-defined]

        assert [col.name for col in table.primary_key.columns] == ["id"]
        assert table.c.id.nullable is False
        assert table.c.name.nullable is False
        assert isinstance(table.c.name.type, SaString)
        assert table.c.name.type.length == 50
        assert table.c.note.nullable is True
        assert isinstance(table.c.note.type, SaText)

    def test_multi_constraint_table_wraps_table_args_and_maps_types(self, tmp_path: Path) -> None:
        namespace = _convert(tmp_path, FIXTURE_SCHEMA.read_text())
        table = namespace["Base"].metadata.tables["constrained_table"]  # type: ignore[attr-defined]

        assert [col.name for col in table.primary_key.columns] == ["id"]
        unique = next(c for c in table.constraints if isinstance(c, UniqueConstraint))
        assert [col.name for col in unique.columns] == ["code"]
        check = next(c for c in table.constraints if isinstance(c, CheckConstraint))
        assert "small_val > 0" in str(check.sqltext)
        fk = next(c for c in table.constraints if isinstance(c, ForeignKeyConstraint))
        assert [col.name for col in fk.columns] == ["simple_id"]
        assert next(iter(fk.elements)).column.table.name == "simple_table"
        assert fk.ondelete == "CASCADE"
        index_names = {index.name for index in table.indexes}
        assert index_names == {"ix_constrained_table_code", "ux_constrained_table_big_val"}
        unique_index = next(i for i in table.indexes if i.name == "ux_constrained_table_big_val")
        assert unique_index.unique is True

        assert isinstance(table.c.small_val.type, SmallInteger)
        assert isinstance(table.c.big_val.type, BigInteger)
        assert isinstance(table.c.price.type, Numeric)
        assert (table.c.price.type.precision, table.c.price.type.scale) == (10, 2)
        assert isinstance(table.c.generic_amount.type, Numeric)
        assert isinstance(table.c.created_at.type, DateTime)
        assert isinstance(table.c.id.type, Integer)

    def test_table_without_constraints_or_indexes_has_no_table_args(self, tmp_path: Path) -> None:
        input_path = tmp_path / "schema.sql"
        output_path = tmp_path / "orm.py"
        input_path.write_text("CREATE TABLE bare_table (label VARCHAR(20) NOT NULL);")
        convert_sqlite_to_orm(input_path, output_path, chembl_version="test")

        assert "__table_args__" not in output_path.read_text()

    def test_skips_internal_sqlite_tables(self, tmp_path: Path) -> None:
        namespace = _convert(tmp_path, FIXTURE_SCHEMA.read_text())

        assert "sqlite_stat1" not in namespace["Base"].metadata.tables  # type: ignore[attr-defined]

    def test_rejects_non_create_statements(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="only CREATE TABLE statements"):
            _convert(tmp_path, "SELECT 1;")

    def test_rejects_unsupported_create_statements(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="only CREATE TABLE and CREATE INDEX statements"):
            _convert(tmp_path, "CREATE VIEW v AS SELECT 1;")

    def test_rejects_multiple_primary_keys(self, tmp_path: Path) -> None:
        sql = """
        CREATE TABLE t (
            a INT NOT NULL,
            b INT NOT NULL,
            CONSTRAINT pk1 PRIMARY KEY (a),
            CONSTRAINT pk2 PRIMARY KEY (b)
        );
        """
        with pytest.raises(ValueError, match="Multiple primary keys found"):
            _convert(tmp_path, sql)

    def test_rejects_unknown_foreign_key_option(self, tmp_path: Path) -> None:
        sql = """
        CREATE TABLE other (id INT NOT NULL, CONSTRAINT other_pk PRIMARY KEY (id));
        CREATE TABLE t (
            ref_id INT,
            CONSTRAINT t_fk FOREIGN KEY (ref_id) REFERENCES other (id) ON UPDATE CASCADE
        );
        """
        with pytest.raises(ValueError, match="Unknown option"):
            _convert(tmp_path, sql)

    def test_rejects_unknown_column_type(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="Unknown column type"):
            _convert(tmp_path, "CREATE TABLE t (c FLOAT);")

    def test_rejects_expressions_on_simple_mapping_type(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Simple mapping types should not have expressions"):
            _convert(tmp_path, "CREATE TABLE t (c TEXT(10));")

    def test_rejects_unknown_column_constraint(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="Unknown column constraint type"):
            _convert(tmp_path, "CREATE TABLE t (c INT DEFAULT 5);")
