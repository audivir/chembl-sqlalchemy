"""Tests for the generated ChEMBL ORM schemas."""

from __future__ import annotations

import importlib

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from chembl_sqlalchemy.chembl_35 import ActionType, Base

SCHEMA_MODULES = [
    "chembl_sqlalchemy.chembl_35",
    "chembl_sqlalchemy.chembl_36",
    "chembl_sqlalchemy.chembl_37",
]


@pytest.fixture(params=SCHEMA_MODULES)
def schema_module(request: pytest.FixtureRequest) -> object:
    return importlib.import_module(request.param)


def test_metadata_registers_all_generated_tables(schema_module: object) -> None:
    assert len(schema_module.Base.metadata.tables) > 50  # type: ignore[attr-defined]


def test_action_type_roundtrips_through_sqlite(schema_module: object) -> None:
    action_type_cls = schema_module.ActionType  # type: ignore[attr-defined]
    metadata = schema_module.Base.metadata  # type: ignore[attr-defined]
    engine = create_engine("sqlite:///:memory:")
    try:
        metadata.create_all(engine, tables=[metadata.tables["action_type"]])

        with Session(engine) as session:
            session.add(action_type_cls(action_type="AGONIST", description="Agonist action"))
            session.commit()

            row = session.execute(
                select(action_type_cls).where(action_type_cls.action_type == "AGONIST")
            ).scalar_one()

            assert row.description == "Agonist action"
            assert row.parent_type is None
    finally:
        engine.dispose()


def test_top_level_import_is_deprecated_and_resolves_to_chembl_35() -> None:
    import chembl_sqlalchemy

    with pytest.warns(DeprecationWarning, match="chembl_sqlalchemy.chembl_35"):
        action_type = chembl_sqlalchemy.ActionType

    with pytest.warns(DeprecationWarning, match="chembl_sqlalchemy.chembl_35"):
        base = chembl_sqlalchemy.Base

    assert action_type is ActionType
    assert base is Base


def test_submodule_import_does_not_warn(recwarn: pytest.WarningsRecorder) -> None:
    importlib.import_module("chembl_sqlalchemy.chembl_35")

    assert not [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]


def test_submodule_attribute_access_does_not_warn(recwarn: pytest.WarningsRecorder) -> None:
    import chembl_sqlalchemy

    module = chembl_sqlalchemy.__getattr__("chembl_36")

    assert module is importlib.import_module("chembl_sqlalchemy.chembl_36")
    assert not [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]


def test_unknown_attribute_raises_attribute_error() -> None:
    import chembl_sqlalchemy

    with pytest.raises(AttributeError, match="has no attribute 'DoesNotExist'"):
        chembl_sqlalchemy.DoesNotExist  # noqa: B018
