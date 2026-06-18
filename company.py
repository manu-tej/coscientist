"""Entrypoint for the drug-repurposing techbio CEO console.

Usage:
    python company.py new --name Helix --disease "pulmonary arterial hypertension"
    python company.py program-new --name PAH-1
    python company.py quarter --auto
    python company.py status

See docs/superpowers/specs/2026-06-06-ai-native-repurposing-techbio-design.md.
"""
from company.cli import main

if __name__ == "__main__":
    main()
