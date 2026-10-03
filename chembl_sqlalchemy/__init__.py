"""SQLAlchemy ORM schemas for the ChEMBL database.

Each supported ChEMBL release has its own submodule (`chembl_sqlalchemy.chembl_35`,
`chembl_sqlalchemy.chembl_36`, `chembl_sqlalchemy.chembl_37`), since table and column
definitions differ between releases. Importing from the top-level package still works for
backward compatibility and resolves to the ChEMBL 35 schema, but is deprecated: it emits a
`DeprecationWarning` and will be removed in a future major version. Import the submodule
matching the ChEMBL database version in use instead.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

__version__ = "1.1.1"

__all__ = [
    "ActionType",
    "Activities",
    "ActivityProperties",
    "ActivitySmid",
    "ActivityStdsLookup",
    "ActivitySupp",
    "ActivitySuppMap",
    "AssayClassMap",
    "AssayClassification",
    "AssayParameters",
    "AssayType",
    "Assays",
    "AtcClassification",
    "Base",
    "BindingSites",
    "BioComponentSequences",
    "BioassayOntology",
    "BiotherapeuticComponents",
    "Biotherapeutics",
    "CellDictionary",
    "ChemblIdLookup",
    "ChemblRelease",
    "ComponentClass",
    "ComponentDomains",
    "ComponentGo",
    "ComponentSequences",
    "ComponentSynonyms",
    "CompoundProperties",
    "CompoundRecords",
    "CompoundStructuralAlerts",
    "CompoundStructures",
    "ConfidenceScoreLookup",
    "CurationLookup",
    "DataValidityLookup",
    "DefinedDailyDose",
    "Docs",
    "Domains",
    "DrugIndication",
    "DrugMechanism",
    "DrugWarning",
    "Formulations",
    "FracClassification",
    "GoClassification",
    "HracClassification",
    "IndicationRefs",
    "IracClassification",
    "LigandEff",
    "MechanismRefs",
    "Metabolism",
    "MetabolismRefs",
    "MoleculeAtcClassification",
    "MoleculeDictionary",
    "MoleculeFracClassification",
    "MoleculeHierarchy",
    "MoleculeHracClassification",
    "MoleculeIracClassification",
    "MoleculeSynonyms",
    "OrganismClass",
    "PatentUseCodes",
    "PredictedBindingDomains",
    "ProductPatents",
    "Products",
    "ProteinClassSynonyms",
    "ProteinClassification",
    "RelationshipType",
    "ResearchCompanies",
    "ResearchStem",
    "SiteComponents",
    "Source",
    "StructuralAlertSets",
    "StructuralAlerts",
    "TargetComponents",
    "TargetDictionary",
    "TargetRelations",
    "TargetType",
    "TissueDictionary",
    "UsanStems",
    "VariantSequences",
    "Version",
    "WarningRefs",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        warnings.warn(
            f"Importing `{name}` directly from `chembl_sqlalchemy` is deprecated and resolves "
            "to the ChEMBL 35 schema. Import from `chembl_sqlalchemy.chembl_35` (or the "
            "submodule matching your ChEMBL database version) instead. This default will be "
            "removed in version 2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = importlib.import_module("chembl_sqlalchemy.chembl_35")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
