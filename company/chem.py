"""Cheminformatics inputs — drug → structure resolution (spec §5).

The pipeline's candidates are drug *names*; every structure-based method (ChemBERTa
similarity today, Boltz-2 docking next) needs an actual molecular structure. This
module is that resolution layer: a curated SMILES table for the approved-drug pool,
plus the leave-one-out reference logic the structure method scores against.

The SMILES are real and canonical. A PubChem/ChEMBL lookup can replace the table
without changing the interface — `smiles_for()` is the seam.
"""
from __future__ import annotations

from typing import Optional

# Canonical SMILES for the approved-drug library (science._DRUG_POOL).
_SMILES: dict[str, str] = {
    "sildenafil": "CCCc1nn(C)c2c1nc([nH]c2=O)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1",
    "tadalafil": "CN1CC(=O)N2[C@@H](c3[nH]c4ccccc4c3C[C@H]2C1=O)c1ccc2c(c1)OCO2",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "spironolactone": "CC(=O)SC1CC2=CC(=O)CCC2(C)C2CCC3(C)C(CCC33CCC(=O)O3)C12",
    "loratadine": "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1",
    "thalidomide": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
    "lenalidomide": "NC1=CC=CC2=C1C(=O)N(C1CCC(=O)NC1=O)C2",
    "minoxidil": "Nc1cc(N2CCCCC2)[n+]([O-])c(N)n1",
    "finasteride": "CC(C)(C)NC(=O)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4NC(=O)C=C[C@]4(C)[C@H]3CC[C@]12C",
    "raloxifene": "OCCN1CCCCC1.O=C(c1ccc(O)cc1)c1c(-c2ccc(O)cc2)sc2cc(O)ccc12",
    "tamoxifen": "CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
}

# Known actives per disease, used ONLY as leave-one-out references for the
# structure method. Kept deliberately separate from the validation gold set's
# purpose: scoring a drug never uses that same drug as its own reference.
_KNOWN_ACTIVES: dict[str, list[str]] = {
    "pulmonary arterial hypertension": ["sildenafil", "tadalafil"],
    "multiple myeloma": ["thalidomide", "lenalidomide"],
    "androgenetic alopecia": ["minoxidil", "finasteride"],
    "breast cancer": ["raloxifene", "tamoxifen"],
    "erectile dysfunction": ["sildenafil", "tadalafil"],
    "type 2 diabetes": ["metformin"],
}


def smiles_for(drug: str) -> Optional[str]:
    """Canonical SMILES for a drug name, or None if unresolved (method abstains)."""
    return _SMILES.get(drug.strip().lower())


def reference_actives(drug: str, disease: str) -> list[str]:
    """Known actives for `disease`, EXCLUDING `drug` itself — leave-one-out.

    Scoring a drug against a reference set containing itself would be circular
    (and would trivially ace the gold-set validation). Excluding it makes the
    signal a real prediction: 'is this molecule analogous to OTHER things that
    work here?' — repurposing by chemical analogy.
    """
    actives = _KNOWN_ACTIVES.get(disease.strip().lower(), [])
    return [a for a in actives if a.strip().lower() != drug.strip().lower()]


def reference_smiles(drug: str, disease: str) -> list[tuple[str, str]]:
    """(name, SMILES) pairs for the leave-one-out references that resolve."""
    out = []
    for name in reference_actives(drug, disease):
        smi = smiles_for(name)
        if smi:
            out.append((name, smi))
    return out
