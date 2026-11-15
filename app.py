"""
Bangladesh Tax & Investment Decision Support System - command-line entry point.

Engines (all in the legal_engine/ package):
  1. engine1_rag.TemporalLegalRAG          legal retrieval (optional: needs torch + BGE-M3)
  2. engine2_tax.TaxCalculationEngine      deterministic tax computation
  3. engine3_eligibility                   investment rebate eligibility (used by engine 4)
  4. engine4_planner.InvestmentPlannerEngine  investment scenarios

All tax rates, caps and limits live in the engines, never in this file.
Run from anywhere:  python app.py   (or press F5 in VS Code)
"""
import sys
from pathlib import Path

# Windows consoles can fail on non-ASCII symbols
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_project_root() -> Path:
    """Walk up from this file until a folder containing legal_engine/ is found."""
    here = Path(__file__).resolve()
    for folder in (here.parent, *here.parents):
        if (folder / "legal_engine").is_dir():
            return folder
    sys.exit(
        "Could not find the legal_engine/ folder in any parent of\n"
        f"  {here.parent}\n"
        "Create it in the project root (next to data/ and scripts/) with __init__.py, "
        "engine1_rag.py, engine2_tax.py, engine3_eligibility.py and engine4_planner.py."
    )


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from legal_engine.engine2_tax import TaxCalculationEngine, TaxpayerProfile
    from legal_engine.engine4_planner import FinancialConstraints, InvestmentPlannerEngine
except ModuleNotFoundError as e:
    sys.exit(
        f"{e}\nFound legal_engine/ at {PROJECT_ROOT / 'legal_engine'}, but a module is missing.\n"
        "It needs __init__.py, engine2_tax.py, engine3_eligibility.py and engine4_planner.py "
        "(exact names), and the packages they use (e.g. pip install pydantic)."
    )

ASSESSMENT_YEAR = "2026-27"
RISK_LEVELS = ("Conservative", "Balanced", "Growth")
RETRIEVAL_QUESTION = "What is the investment rebate rule and the employment income deduction ceiling?"

# Demo corpus for Engine 1. Replace with the real chunk file once the evaluated retriever
# is moved into legal_engine/. Check both figures against the Finance Act in force.
DEMO_CORPUS = [
    {
        "chunk_id": "c1",
        "act_title": "Income Tax Act 2023 / Finance Act 2026",
        "section": "Sec 32",
        "text": "Statutory deduction for employment income is one-third of gross salary "
                "or Tk 5,00,000, whichever is lower.",
        "effective_date": "2023-07-01",
    },
    {
        "chunk_id": "c2",
        "act_title": "Finance Act 2026",
        "section": "Sixth Schedule Part 3",
        "text": "Rebate is calculated at the lowest of 10% of eligible investment, "
                "3% of taxable income, or Tk 7,50,000.",
        "effective_date": "2026-07-01",
    },
]


# =========================== INPUT HELPERS ===========================
def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def ask_amount(label: str, default: float) -> float:
    """Ask for a non-negative amount; accepts '15,00,000' style commas; Enter = default."""
    while True:
        raw = input(f"{label} [{default:,.0f}]: ").strip().replace(",", "")
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if value < 0:
            print("  Amount cannot be negative.")
            continue
        return value


def ask_choice(label: str, options, default: str) -> str:
    opts = "/".join(options)
    while True:
        raw = input(f"{label} ({opts}) [{default}]: ").strip()
        if not raw:
            return default
        for o in options:
            if raw.lower() == o.lower():
                return o
        print(f"  Choose one of: {opts}")


# =========================== ENGINES ===========================
def run_retrieval(question: str) -> None:
    """Engine 1 is optional: the rest of the app works without torch / the embedding model."""
    print_header("ENGINE 1: TEMPORAL LEGAL RETRIEVAL")
    try:
        from legal_engine.engine1_rag import TemporalLegalRAG
        rag = TemporalLegalRAG()
        rag.load_corpus(DEMO_CORPUS)
        results = rag.retrieve(question)
    except Exception as e:
        print(f"Retrieval skipped ({type(e).__name__}: {e})")
        print("Install requirements with: pip install -r requirements.txt "
              "(the first run downloads BAAI/bge-m3, about 2 GB)")
        return
    print(f"Question: {question}")
    if not results:
        print("No provisions retrieved.")
    for d in results:
        print(f"- [{d.act_title} - {d.section}]: {d.text}")


def run_tax(profile: TaxpayerProfile) -> None:
    tax_res = TaxCalculationEngine().compute_tax(profile)
    print_header("ENGINE 2: DETERMINISTIC TAX COMPUTATION")
    print(f"Gross Income         : BDT {tax_res.gross_income:,.2f}")
    print(f"Sec 32 Deduction     : BDT {tax_res.statutory_deduction:,.2f}")
    print(f"Taxable Income       : BDT {tax_res.taxable_income:,.2f}")
    print(f"Tax-Free Threshold   : BDT {tax_res.tax_free_threshold:,.2f}")
    print(f"Gross Tax Liability  : BDT {tax_res.gross_tax_liability:,.2f}")
    if tax_res.applied_slabs:
        print("\nApplied Slabs Breakup:")
        for s in tax_res.applied_slabs:
            print(f"  - {s['label']}: BDT {s['amount']:,.2f} -> BDT {s['tax']:,.2f}")


def run_planner(profile: TaxpayerProfile, constraints: FinancialConstraints) -> None:
    scenarios = InvestmentPlannerEngine().plan(profile, constraints)
    print_header("ENGINE 3 & 4: INVESTMENT SCENARIOS")
    for sc in scenarios:
        print(f"\n> Scenario: {sc.name}")
        print(f"  Total Invested       : BDT {sc.total_investment:,.2f}")
        for asset, amount in sc.allocation.items():
            if amount > 0:
                print(f"    - {asset:<20}: BDT {amount:,.2f}")
        print(f"  Tax Rebate Earned    : BDT {sc.tax_rebate:,.2f}")
        print(f"  Final Net Tax Payable: BDT {sc.final_tax_payable:,.2f}")
        print(f"  Tax Benefit/Invested : {sc.tbi_metric * 100:.2f}% (TBI)")
        print(f"  Effective Cost       : BDT {sc.eic_metric:,.2f} (EIC)")
        print(f"  Remaining Cash       : BDT {sc.net_remaining_disposable:,.2f}")


# =========================== MAIN ===========================
def main() -> None:
    print_header(f"BANGLADESH TAX & INVESTMENT DECISION SUPPORT SYSTEM (AY {ASSESSMENT_YEAR})")

    print("\n--- ENTER TAXPAYER FINANCIAL DETAILS (press Enter for the default) ---")
    salary = ask_amount("Annual Employment Salary (BDT)", 1_500_000)
    rent = ask_amount("Annual Rent Income (BDT)", 0)
    other = ask_amount("Other Annual Income (BDT)", 0)
    savings_cap = ask_amount("Annual Investment Budget (BDT)", 300_000)
    risk = ask_choice("Risk tolerance", RISK_LEVELS, "Balanced")

    profile = TaxpayerProfile(
        category="General",            # add a prompt here once engine2 supports other categories
        employment_income=salary,
        rent_income=rent,
        other_income=other,
    )
    constraints = FinancialConstraints(disposable_savings_capacity=savings_cap, risk_tolerance=risk)

    run_retrieval(RETRIEVAL_QUESTION)
    run_tax(profile)
    run_planner(profile, constraints)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
