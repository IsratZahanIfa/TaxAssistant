"""
Faithfulness (source consistency) of generated answers, using an LLM as judge.

For each answer the judge splits it into atomic claims and marks each claim as supported
or not by the retrieved context, quoting the supporting evidence verbatim.
Two scores are reported:
  * faithfulness          : supported claims / all claims (judge's verdict)
  * verified_faithfulness : supported claims whose quoted evidence really appears in the
                            context / all claims (guards against the judge inventing support)
An answer with no claims gets score None and status "no_claims" (it is not counted as 1.0).

Backends:
  --backend auto    (default) OpenAI if a real OPENAI_API_KEY is set, otherwise local Ollama
  --backend openai  uses OPENAI_API_KEY, model gpt-4o-mini
  --backend ollama  uses a local Ollama server through its OpenAI-compatible API

Usage:
  python scripts/Eval_faithfulness.py                                   # built-in demo
  python scripts/Eval_faithfulness.py --input answers.json --output faithfulness.xlsx
  python scripts/Eval_faithfulness.py --backend ollama --model gemma3:12b --input answers.csv

Input file (JSON list or CSV) needs the columns: context, answer  (optional: id, question)
Note: use a different model as judge than the one that generated the answers.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---- Use the project's .venv automatically --------------------------------------------
# VS Code's Run button may start this file with a Python that has no packages installed.
# If a .venv exists in this folder or a parent folder and we are not running inside it,
# restart this script with the .venv's Python (same arguments, same terminal).
def _reexec_in_project_venv() -> None:
    import os
    import subprocess
    if os.environ.get("TAXASSIST_VENV_REEXEC"):
        return  # already restarted once; never loop
    here = Path(__file__).resolve()
    rel = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    for folder in (here.parent, *here.parents):
        venv = folder / ".venv"
        if (venv / rel).exists():
            if Path(sys.prefix).resolve() != venv.resolve():
                print(f"[info] Restarting with the project environment: {venv / rel}")
                env = dict(os.environ, TAXASSIST_VENV_REEXEC="1")
                sys.exit(subprocess.call([str(venv / rel), str(here), *sys.argv[1:]], env=env))
            return


_reexec_in_project_venv()

try:
    from openai import OpenAI
except ModuleNotFoundError:
    sys.exit("Missing package 'openai'. From the project folder run:\n"
             "  uv pip install --python .venv openai pandas openpyxl")

SYSTEM_PROMPT = "You are a precise, objective compliance auditor for Bangladesh tax law."

USER_PROMPT = """Check the faithfulness of a generated answer against the retrieved context.

Steps:
1. Split the Generated Answer into atomic factual claims (one fact each). Keep each claim in
   the language of the answer (Bangla or English).
2. For each claim decide whether it is explicitly stated in, or directly derivable from,
   the Retrieved Context. General knowledge that is not in the context counts as NOT supported.
3. For a supported claim, copy the supporting sentence or phrase from the context VERBATIM
   into "evidence". For an unsupported claim, set "evidence" to "".

Retrieved Context:
<<<CONTEXT>>>

Generated Answer:
<<<ANSWER>>>

Return ONLY a JSON object of this form:
{"claims": [{"claim": "...", "is_supported": true, "evidence": "...", "reasoning": "..."}]}"""


# =========================== HELPERS ===========================
def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1", "supported"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def evidence_in_context(evidence: str, context: str) -> bool:
    ev = _norm(evidence)
    return bool(ev) and ev in _norm(context)


def parse_json(text: str) -> Dict[str, Any]:
    """Parse the judge's reply, tolerating code fences or text around the JSON."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _real_openai_key() -> Optional[str]:
    """The OPENAI_API_KEY if it looks real (not unset and not a copied placeholder)."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    placeholder = (not key.startswith("sk-") or len(key) < 30
                   or any(w in key.lower() for w in ("your", "here", "actual", "xxxx")))
    return None if placeholder else key


def _ollama_ready(ollama_url: str, model: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{ollama_url.rstrip('/')}/api/tags", timeout=3) as r:
            names = [m.get("name", "") for m in json.load(r).get("models", [])]
    except Exception:
        return False
    return any(n == model or n.startswith(model + ":") for n in names)


def make_client(backend: str, ollama_url: str, ollama_model: str):
    """Returns (client, backend actually used). 'auto' = OpenAI if a real key is set, else Ollama."""
    key = _real_openai_key()
    if backend == "auto":
        backend = "openai" if key else "ollama"
        print(f"[info] Judge backend: {backend}")
    if backend == "openai":
        if not key:
            sys.exit("OPENAI_API_KEY is missing or is still the placeholder text.\n"
                     "Set your real key in PowerShell:  $env:OPENAI_API_KEY = \"sk-proj-...\"\n"
                     "or use the local judge:          --backend ollama")
        return OpenAI(api_key=key), backend
    if not _ollama_ready(ollama_url, ollama_model):
        sys.exit(f"No judge available.\n"
                 f"  - OpenAI: set a real key with  $env:OPENAI_API_KEY = \"sk-proj-...\", or\n"
                 f"  - Ollama: install it from https://ollama.com/download, start it, then run "
                 f"'ollama pull {ollama_model}'.")
    return OpenAI(base_url=f"{ollama_url.rstrip('/')}/v1", api_key="ollama"), backend


# =========================== CORE ===========================
def calculate_faithfulness(context: str, generated_answer: str, client: OpenAI,
                           model: str = "gpt-4o-mini", retries: int = 3) -> Dict[str, Any]:
    """Faithfulness of one answer. Returns scores, counts, per-claim details and a status."""
    if not str(generated_answer).strip():
        return {"faithfulness_score": None, "verified_faithfulness_score": None,
                "total_claims": 0, "supported_claims": 0, "verified_claims": 0,
                "status": "empty_answer", "details": []}

    prompt = USER_PROMPT.replace("<<<CONTEXT>>>", str(context)).replace("<<<ANSWER>>>", str(generated_answer))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

    last_error: Optional[Exception] = None
    data = None
    for attempt in range(retries):
        try:
            kwargs = {"model": model, "temperature": 0.0, "messages": messages}
            if attempt == 0:
                kwargs["response_format"] = {"type": "json_object"}   # dropped on retry if unsupported
            response = client.chat.completions.create(**kwargs)
            data = parse_json(response.choices[0].message.content)
            break
        except Exception as e:  # network, rate limit, bad JSON
            last_error = e
            time.sleep(2 ** attempt)
    if data is None:
        return {"faithfulness_score": None, "verified_faithfulness_score": None,
                "total_claims": 0, "supported_claims": 0, "verified_claims": 0,
                "status": f"error: {type(last_error).__name__}: {last_error}", "details": []}

    claims: List[Dict[str, Any]] = [c for c in data.get("claims", []) if isinstance(c, dict)]
    if not claims:
        return {"faithfulness_score": None, "verified_faithfulness_score": None,
                "total_claims": 0, "supported_claims": 0, "verified_claims": 0,
                "status": "no_claims", "details": []}

    for c in claims:
        c["is_supported"] = as_bool(c.get("is_supported"))
        c["evidence_found"] = c["is_supported"] and evidence_in_context(c.get("evidence", ""), context)

    supported = sum(c["is_supported"] for c in claims)
    verified = sum(c["evidence_found"] for c in claims)
    return {
        "faithfulness_score": round(supported / len(claims), 4),
        "verified_faithfulness_score": round(verified / len(claims), 4),
        "total_claims": len(claims),
        "supported_claims": supported,
        "verified_claims": verified,
        "status": "ok",
        "details": claims,
    }


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    else:
        import pandas as pd
        rows = pd.read_csv(path).fillna("").to_dict("records")
    for i, r in enumerate(rows):
        if "context" not in r or "answer" not in r:
            raise ValueError(f"Row {i} needs 'context' and 'answer' fields.")
        r.setdefault("id", i)
    return rows


def evaluate_file(path: Path, output: Path, client: OpenAI, model: str) -> None:
    import pandas as pd
    rows = load_rows(path)
    results, claim_rows = [], []
    for r in rows:
        res = calculate_faithfulness(r["context"], r["answer"], client, model)
        results.append({"id": r["id"], "question": r.get("question", ""),
                        **{k: v for k, v in res.items() if k != "details"}})
        for c in res["details"]:
            claim_rows.append({"id": r["id"], **c})
        print(f"[{r['id']}] {res['status']}  faithfulness={res['faithfulness_score']}  "
              f"verified={res['verified_faithfulness_score']}")

    df = pd.DataFrame(results)
    scored = df[df["status"] == "ok"]
    summary = pd.DataFrame([{
        "answers": len(df), "scored": len(scored),
        "no_claims_or_empty": int(df["status"].isin(["no_claims", "empty_answer"]).sum()),
        "errors": int(df["status"].str.startswith("error").sum()),
        "mean_faithfulness": scored["faithfulness_score"].mean() if len(scored) else None,
        "mean_verified_faithfulness": scored["verified_faithfulness_score"].mean() if len(scored) else None,
        "judge_model": model,
    }])

    if output.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(output, engine="openpyxl") as w:
            summary.to_excel(w, sheet_name="Summary", index=False)
            df.to_excel(w, sheet_name="Answers", index=False)
            pd.DataFrame(claim_rows).to_excel(w, sheet_name="Claims", index=False)
    else:
        df.to_csv(output, index=False, encoding="utf-8-sig")
    print("\n" + summary.to_string(index=False))
    print(f"Saved: {output}")


# =========================== CLI ===========================
def main():
    ap = argparse.ArgumentParser(description="LLM-as-judge faithfulness evaluation")
    ap.add_argument("--input", type=Path, help="JSON or CSV with context and answer columns")
    ap.add_argument("--output", type=Path, default=Path("faithfulness_results.xlsx"))
    ap.add_argument("--backend", choices=["auto", "openai", "ollama"], default="auto")
    ap.add_argument("--model", default=None, help="default: gpt-4o-mini (openai) / gemma3:12b (ollama)")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ollama_model = args.model or "gemma3:12b"
    client, backend = make_client(args.backend, args.ollama_url, ollama_model)
    model = args.model or ("gemma3:12b" if backend == "ollama" else "gpt-4o-mini")

    if args.input:
        evaluate_file(args.input, args.output, client, model)
        return

    # Demo with illustrative text (not the actual wording of the Act).
    context = ("Under Section 56 of the Income Tax Act, non-resident individuals are subject to a "
               "maximum withholding tax rate of 30% on taxable income derived from Bangladesh.")
    answer = ("Non-resident individuals are taxed at a 30% rate under Section 56. "
              "Additionally, they receive an automatic 10% tax refund on all domestic investments.")
    print("================ GENERATION EVALUATION ================")
    res = calculate_faithfulness(context, answer, client, model)
    print(f"Status                : {res['status']}")
    if res["status"] == "ok":
        print(f"Faithfulness          : {res['faithfulness_score']:.2%}")
        print(f"Verified faithfulness : {res['verified_faithfulness_score']:.2%}")
        print(f"Supported / verified  : {res['supported_claims']} / {res['verified_claims']} "
              f"of {res['total_claims']} claims")
        print(json.dumps(res["details"], indent=2, ensure_ascii=False))
    print("=======================================================")


if __name__ == "__main__":
    main()
