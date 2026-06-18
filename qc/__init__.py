"""QC scientist for biomedical tabular/omics data — raw-data quality control with
calibrated uncertainty on every flag.

Design stance (earned the hard way, see the techbio spec's devil's-advocate review):
the UQ must be *real*, not LLM-estimated. So this package is a statistical core —
each detector returns a Flag carrying a test statistic, a (multiplicity-corrected)
p-value, and an effect size. The agent/LLM layer (orchestration: which checks fit
this data, interpreting results, writing the report) sits ON TOP and is optional;
its value is meant to be measured, not assumed.

The honesty contract is enforced by `harness.py`: under *clean* data the battery's
false-positive rate must track the nominal alpha (calibration), and on *injected*
corruptions its detection recall is measured. Flags you can't trust are worse than
no flags.
"""
