import sys
from pathlib import Path

# Find the project root (the folder that contains legal_engine/) so this file
# works whether it sits in E:\TaxAssist or inside E:\TaxAssist\legal_engine
_here = Path(__file__).resolve().parent
_root = _here if (_here / "legal_engine").is_dir() else _here.parent
sys.path.insert(0, str(_root))

from legal_engine.engine1 import TemporalLegalRAG
from legal_engine.engine2 import TaxCalculationEngine, TaxpayerProfile
from legal_engine.engine4 import InvestmentPlannerEngine, FinancialConstraints

# Windows consoles can choke on symbols like the bullet and arrow below
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    print_header("BANGLADESH TAX & INVESTMENT DECISION SUPPORT SYSTEM (AY 2026-27)")

    corpus = [
        {
            "chunk_id": "c1",
            "act_title": "Income Tax Act 2023 / Finance Act 2026",
            "section": "Sec 32",
            "text": "Statutory deduction for employment income is one-third of gross salary or Tk 5,00,000, whichever is lower.",
            "effective_date": "2023-07-01",
        },
        {
            "chunk_id": "c2",
            "act_title": "Finance Act 2026",
            "section": "Sixth Schedule Part 3",
            "text": "Rebate is calculated at the lowest of 10% of eligible investment, 3% of taxable income, or Tk 7,50,000.",
            "effective_date": "2026-07-01",
        },
    ]

    print("\n--- ENTER TAXPAYER FINANCIAL DETAILS ---")
    try:
        salary = float(input("Annual Employment Salary (BDT) [1500000]: ") or "1500000")
        rent = float(input("Annual Rent Income (BDT) [0]: ") or "0")
        other = float(input("Other Annual Income (BDT) [0]: ") or "0")
        savings_cap = float(input("Annual Investment Budget (BDT) [300000]: ") or "300000")
    except ValueError:
        print("Invalid numerical input. Falling back to default values.")
        salary, rent, other, savings_cap = 1500000.0, 0.0, 0.0, 300000.0

    profile = TaxpayerProfile(
        category="General",
        employment_income=salary,
        rent_income=rent,
        other_income=other,
    )
    constraints = FinancialConstraints(
        disposable_savings_capacity=savings_cap, risk_tolerance="Balanced"
    )

    # Engine 1: the app keeps working even if the embedding model can't load
    print_header("ENGINE 1: TEMPORAL LEGAL RETRIEVAL")
    try:
        rag = TemporalLegalRAG()
        rag.load_corpus(corpus)
        for d in rag.retrieve("What is the rebate rule and employment deduction ceiling?"):
            print(f"- [{d.act_title} - {d.section}]: {d.text}")
    except Exception as e:
        print(f"Retrieval skipped ({type(e).__name__}: {e})")
        print("Run: pip install -r requirements.txt  (first run downloads BAAI/bge-m3)")

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

    scenarios = InvestmentPlannerEngine().plan(profile, constraints)
    print_header("ENGINE 3 & 4: OPTIMIZED INVESTMENT SCENARIOS")
    for sc in scenarios:
        print(f"\n> Scenario: {sc.name}")
        print(f"  Total Invested       : BDT {sc.total_investment:,.2f}")
        print(f"  Tax Rebate Earned    : BDT {sc.tax_rebate:,.2f}")
        print(f"  Final Net Tax Payable: BDT {sc.final_tax_payable:,.2f}")
        print(f"  Tax Benefit/Invested : {sc.tbi_metric * 100:.2f}% (TBI)")
        print(f"  Effective Cost       : BDT {sc.eic_metric:,.2f} (EIC)")
        print(f"  Remaining Cash       : BDT {sc.net_remaining_disposable:,.2f}")


if __name__ == "__main__":
    main()