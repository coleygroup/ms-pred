"""
Use npclassifier to predict the chemical classes
"""
from tqdm import tqdm
import requests, threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

datasets_to_process = ['nist20']
MAX_WORKERS = 16

BASE_URL = "https://npclassifier.gnps2.org/classify"

# ---- Thread-safe session factory (one Session per thread) ----
_tls = threading.local()
def get_session():
    if getattr(_tls, "session", None) is None:
        s = requests.Session()
        retry = Retry(
            total=3, backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={"GET"},
        )
        # Increase pool_maxsize so multiple threads can reuse connections
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100))
        s.headers.update({"User-Agent": "npclassifier-python-demo/0.2"})
        _tls.session = s
    return _tls.session

def classify_smiles(smiles: str, timeout: float = 10.0) -> dict:
    sess = get_session()
    resp = sess.get(BASE_URL, params={"smiles": smiles}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def classify_name_to_smi(name_to_smi: dict[str, str], max_workers: int = 16) -> dict[str, dict]:
    # 1) De-duplicate identical SMILES to avoid paying for repeats
    unique = {}
    for name, smi in name_to_smi.items():
        unique.setdefault(smi, []).append(name)

    # 2) Parallel fetch unique SMILES
    smi_to_result: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(classify_smiles, smi): smi for smi in unique}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Classifying"):
            smi = futures[fut]
            try:
                smi_to_result[smi] = fut.result()
            except requests.HTTPError as e:
                smi_to_result[smi] = {"error": f"HTTP {e.response.status_code}", "details": str(e)}
            except requests.RequestException as e:
                smi_to_result[smi] = {"error": "request-failed", "details": str(e)}

    # 3) Expand back to name->result (keep only pathway_results if you prefer)
    name_to_result = {}
    for smi, names in unique.items():
        for name in names:
            name_to_result[name] = smi_to_result[smi]
    return name_to_result

def name_to_superclass(name_to_smi: dict[str, str], **kwargs) -> dict[str, list[str] | None]:
    raw = classify_name_to_smi(name_to_smi, **kwargs)
    return {
        name: (res.get("superclass_results") if isinstance(res, dict) and "superclass_results" in res else None)
        for name, res in raw.items()
    }

if __name__ == '__main__':
    for dataset in datasets_to_process:
        data_folder = Path(f"data/spec_datasets/{dataset}")
        data_df = pd.read_csv(data_folder / "labels.tsv", sep="\t")

        name_to_smi = dict(data_df[["spec", "smiles"]].values)
        name_to_class = name_to_superclass(name_to_smi, max_workers=MAX_WORKERS)
        out_dicts = []
        for name, cls in name_to_class.items():
            out_dicts.append({
                "dataset": dataset,
                "spec": name,
                "smiles": name_to_smi[name],
                "class": cls[0] if len(cls) == 1 else 'unknown',
            })
        out_df = pd.DataFrame(out_dicts)
        out_df.to_csv(data_folder / "chemical_class_labels.tsv", sep='\t')
