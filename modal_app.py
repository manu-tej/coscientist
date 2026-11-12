"""Modal GPU functions — the T2 fidelity tier (spec §5).

Deployed serverless GPU compute the model registry calls as its highest-fidelity
providers: ChemBERTa molecular embeddings back the `structure` method, and
Boltz-2 protein-ligand predictions back the gated `affinity` method.

Deploy:  modal deploy modal_app.py
Weights are baked into the image at build time, so invocations don't re-download
them (cold starts would otherwise dominate cost).
"""
from __future__ import annotations

import modal

MODEL_ID = "seyonec/ChemBERTa-zinc-base-v1"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "numpy")
    # Bake the weights into the image so each call doesn't pay the download.
    .run_commands(
        f"python -c \"from transformers import AutoTokenizer, AutoModel; "
        f"AutoTokenizer.from_pretrained('{MODEL_ID}'); "
        f"AutoModel.from_pretrained('{MODEL_ID}')\""
    )
)

app = modal.App("coscientist-chem")

# --- Boltz-2 binding affinity (the T2 affinity provider) --------------------
# Weights live in a Volume so they are downloaded ONCE on cheap CPU rather than
# re-pulled on every A100 call — otherwise GPU time is spent on I/O.
boltz_cache = modal.Volume.from_name("boltz-cache", create_if_missing=True)
boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "git")
    .pip_install("boltz[cuda]", extra_options="--extra-index-url https://download.pytorch.org/whl/cu121")
)


@app.function(image=boltz_image, volumes={"/cache": boltz_cache}, timeout=3600, cpu=4)
def warm_boltz_weights() -> str:
    """Pre-populate the weight cache on CPU (no GPU billed while downloading)."""
    import os
    os.environ["BOLTZ_CACHE"] = "/cache"
    from pathlib import Path

    from boltz.main import download_boltz2
    download_boltz2(Path("/cache"))
    boltz_cache.commit()
    return "cached: " + ", ".join(sorted(p.name for p in Path("/cache").iterdir()))


@app.function(image=boltz_image, volumes={"/cache": boltz_cache}, timeout=3600, cpu=4)
def compute_target_msa(sequence: str, target_key: str, retries: int = 4) -> dict:
    """Compute a target's MSA ONCE on CPU and cache it in the volume.

    Two reasons this belongs on CPU rather than inline in the GPU call:
      1. Cost — an MSA computed per-ligand bills A100 time while merely waiting on
         an external server. One MSA per *target* amortizes across every ligand.
      2. Robustness — the public ColabFold server is flaky (it returned a non-archive
         and crashed a GPU run earlier). Retrying here costs fractions of a cent, so
         we can retry properly instead of degrading to single-sequence.
    """
    import shutil
    import tempfile
    import time
    from pathlib import Path

    from boltz.main import compute_msa

    boltz_cache.reload()
    dest = Path("/cache/msa")
    dest.mkdir(parents=True, exist_ok=True)
    # Boltz writes MSAs as .a3m or .csv depending on version — accept either and
    # keep the extension, since the YAML `msa:` field is format-sensitive.
    existing = sorted(dest.glob(f"{target_key}.*"))
    if existing:
        return {"ok": True, "cached": True, "path": str(existing[0]),
                "bytes": existing[0].stat().st_size}

    last_error, produced = "", []
    for attempt in range(1, retries + 1):
        try:
            tmp = Path(tempfile.mkdtemp())
            compute_msa({"A": sequence}, target_key, tmp,
                        "https://api.colabfold.com", "greedy")
            produced = sorted(str(p.name) for p in tmp.rglob("*") if p.is_file())
            hits = sorted(list(tmp.rglob("*.a3m")) + list(tmp.rglob("*.csv")))
            if not hits:
                raise RuntimeError(f"MSA server produced no alignment; files={produced}")
            src = hits[0]
            final = dest / f"{target_key}{src.suffix}"
            shutil.copy(src, final)
            boltz_cache.commit()
            text = final.read_text()
            depth = text.count(">") if src.suffix == ".a3m" else text.count("\n") - 1
            return {"ok": True, "cached": False, "path": str(final), "attempts": attempt,
                    "bytes": final.stat().st_size, "depth": depth, "format": src.suffix}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(5 * attempt)          # linear backoff; server is rate-limited
    return {"ok": False, "error": last_error, "attempts": retries, "produced": produced}


@app.function(gpu="A100-40GB", image=boltz_image, volumes={"/cache": boltz_cache},
              timeout=3600)
def predict_affinity(sequence: str, smiles: str, name: str = "cplx",
                     use_msa_server: bool = False, msa_path: str = "") -> dict:
    """Boltz-2 protein-ligand binding affinity for one (target, ligand) pair.

    Returns the raw Boltz-2 affinity fields; the caller converts the regression
    value to a pharmacology-anchored [0,1] potency score.

    `use_msa_server=False` runs single-sequence mode (`msa: empty`). That trades
    some absolute accuracy for three things that matter here: no dependency on the
    flaky public ColabFold server, no A100 idle-billing while waiting on it, and
    determinism. Because we rank many ligands against ONE target, the protein is
    treated identically across candidates, so relative ranking stays meaningful —
    which is what the tournament consumes. Per-target cached MSAs are the upgrade.
    """
    import json
    import subprocess
    import time
    from pathlib import Path

    # Pick up weights committed by warm_boltz_weights in another container. Without
    # this a cold reader can see a partial cache and fail fast with no useful error.
    boltz_cache.reload()

    work = Path("/tmp/work")
    work.mkdir(parents=True, exist_ok=True)
    yaml_path = work / f"{name}.yaml"
    # Precomputed cached MSA (best: accurate + no GPU wait) > live server > none.
    if msa_path:
        msa_line = f"      msa: {msa_path}\n"
    elif use_msa_server:
        msa_line = ""
    else:
        msa_line = "      msa: empty\n"
    yaml_path.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {sequence}\n"
        f"{msa_line}"
        "  - ligand:\n"
        "      id: B\n"
        f"      smiles: '{smiles}'\n"
        "properties:\n"
        "  - affinity:\n"
        "      binder: B\n"
    )
    out = work / "out"
    cmd = ["boltz", "predict", str(yaml_path), "--cache", "/cache",
           "--out_dir", str(out), "--diffusion_samples", "1"]
    if use_msa_server:
        cmd.append("--use_msa_server")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - t0, 1)
    hits = list(out.rglob("affinity_*.json"))
    if not hits:
        tree = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())[:60]
        return {"ok": False, "elapsed_s": elapsed, "returncode": proc.returncode,
                "produced_files": tree,
                "stderr_tail": proc.stderr[-3000:],
                "stdout_tail": proc.stdout[-3000:]}
    data = json.loads(hits[0].read_text())
    return {"ok": True, "elapsed_s": elapsed,
            "binder_probability": data.get("affinity_probability_binary"),
            "affinity_pred_value": data.get("affinity_pred_value"),
            "raw": data}


@app.cls(gpu="T4", image=image, timeout=600, scaledown_window=300)
class ChemEmbedder:
    """ChemBERTa encoder held warm between calls (weights load once per container)."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).eval().cuda()

    def _embed(self, smiles: list[str]):
        tok = self.tokenizer(smiles, return_tensors="pt", padding=True,
                             truncation=True, max_length=256)
        tok = {k: v.cuda() for k, v in tok.items()}
        with self.torch.no_grad():
            out = self.model(**tok).last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # mean-pool
        return self.torch.nn.functional.normalize(emb, dim=-1)

    @modal.method()
    def max_similarity(self, query_smiles: str, reference_smiles: list[str]) -> dict:
        """Max cosine similarity of `query` to any reference molecule.

        Returns the score in [0,1] plus which reference was nearest, so the caller
        can record *why* (evidence), not just the number.
        """
        if not reference_smiles:
            return {"score": None, "nearest_index": None}
        emb = self._embed([query_smiles] + list(reference_smiles))
        sims = (emb[0:1] @ emb[1:].T).squeeze(0)
        best = int(sims.argmax().item())
        # cosine on normalized vectors is in [-1,1]; clamp to [0,1] for a score
        return {"score": max(0.0, min(1.0, float(sims[best].item()))),
                "nearest_index": best}
