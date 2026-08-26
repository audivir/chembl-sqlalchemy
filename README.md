# chembl-sqlalchemy

SQLAlchemy ORM models for the [ChEMBL](https://www.ebi.ac.uk/chembl/) database, enabling programmatic access to ChEMBL data using Python.

This package allows you to query and explore ChEMBL bioactivity data using SQLAlchemy, without having to manually define the table schemas yourself.

## Prerequisites

- Python 3.10 to 3.14.
- A local ChEMBL database file, downloaded separately (see Database Files below).

## Installation

Install via pip:

```bash
pip install chembl-sqlalchemy
```

## Usage

Table and column definitions differ between ChEMBL releases, so each supported release has its
own submodule: `chembl_sqlalchemy.chembl_35`, `chembl_sqlalchemy.chembl_36`, and
`chembl_sqlalchemy.chembl_37`. Import the submodule matching the ChEMBL database version in use:

```python
from chembl_sqlalchemy.chembl_37 import Activities
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# Connect to a local ChEMBL SQLite database
engine = create_engine("sqlite:///chembl_37.db")
Session = sessionmaker(bind=engine)
session = Session()

# Example query: Get first 1000 non-null pChEMBL values
query = (
    select(Activities.molregno, Activities.pchembl_value, Activities.standard_type)
    .where(Activities.pchembl_value.isnot(None))
    .limit(1000)
)

results = session.execute(query).fetchall()

for molregno, pchembl_value, standard_type in results:
    print(molregno, pchembl_value, standard_type)
```

Importing directly from `chembl_sqlalchemy` (e.g. `from chembl_sqlalchemy import Activities`)
still works and resolves to the ChEMBL 35 schema, for backward compatibility with releases before
`1.1.0`. It emits a `DeprecationWarning` and will be removed in `2.0.0`: there is no default
schema, since silently picking one could break queries against a different ChEMBL release.

## Versioning

The package follows plain semantic versioning. A single package version bundles ORM schemas for
multiple ChEMBL releases as separate submodules, rather than targeting one ChEMBL version per
package version.

## Database Files

The package does not include the ChEMBL database file itself. You can download the corresponding SQLite file from the [ChEMBL downloads page](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/).

Place it in your project directory or reference it by path when creating the SQLAlchemy engine.

## Acknowledgments

This package derives its ORM schemas from the structure of the [ChEMBL](https://www.ebi.ac.uk/chembl/)
database, produced by EMBL-EBI. No ChEMBL data is bundled with this package; download the
database separately (see Database Files above) and note the release number, per ChEMBL
attribution requirements.

For publications using ChEMBL data, cite:

Mendez D, Gaulton A, Bento AP, Chambers J, De Veij M, Félix E, Magariños MP, Mosquera JF, Mutowo
P, Nowotka M, Gordillo-Marañón M, Hunter F, Junco L, Mugumbate G, Rodriguez-Lopez M, Atkinson F,
Bosc N, Radoux CJ, Segura-Cabrera A, Hersey A, Leach AR. ChEMBL: towards direct deposition of
bioassay data. Nucleic Acids Res. 2019 47(D1):D930-D940. DOI: 10.1093/nar/gky1075

## License

MIT, see `LICENSE`.
