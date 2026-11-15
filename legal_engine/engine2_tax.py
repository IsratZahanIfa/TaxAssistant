from typing import Any, Dict, List

from pydantic import BaseModel


class TaxpayerProfile(BaseModel):
    category: str = "General"  # General, Female, Senior, Disabled, Freedom_Fighter
    employment_income: float = 0.0
    rent_income: float = 0.0
    business_income: float = 0.0
    financial_assets_income: float = 0.0
    other_income: float = 0.0


class TaxResult(BaseModel):
    gross_income: float
    statutory_deduction: float
    taxable_income: float
    tax_free_threshold: float
    gross_tax_liability: float
    # FIX: slabs hold a text label, so Dict[str, float] fails pydantic v2 validation
    applied_slabs: List[Dict[str, Any]]


class TaxCalculationEngine:
    # NOTE: verify every figure below against the current Finance Act / NBR circular.
    THRESHOLDS = {
        "General": 400000.0,
        "Female": 425000.0,
        "Senior": 425000.0,
        "Disabled": 475000.0,
        "Freedom_Fighter": 525000.0,
    }
    DEDUCTION_CAP = 500000.0
    SLABS = [
        (300000.0, 0.10, "Next Tk 3,00,000 @ 10%"),
        (400000.0, 0.15, "Next Tk 4,00,000 @ 15%"),
        (500000.0, 0.20, "Next Tk 5,00,000 @ 20%"),
        (2000000.0, 0.25, "Next Tk 20,00,000 @ 25%"),
        (float("inf"), 0.30, "Remaining @ 30%"),
    ]

    def __init__(self, assessment_year: str = "2026-2027"):
        self.assessment_year = assessment_year

    def get_threshold(self, category: str) -> float:
        return self.THRESHOLDS.get(category, self.THRESHOLDS["General"])

    def calculate_statutory_deduction(self, salary: float) -> float:
        if salary <= 0:
            return 0.0
        return min(salary / 3.0, self.DEDUCTION_CAP)

    def compute_tax(self, profile: TaxpayerProfile) -> TaxResult:
        gross = (
            profile.employment_income + profile.rent_income
            + profile.business_income + profile.financial_assets_income
            + profile.other_income
        )
        deduction = self.calculate_statutory_deduction(profile.employment_income)
        taxable = max(0.0, gross - deduction)
        threshold = self.get_threshold(profile.category)

        gross_tax = 0.0
        applied_slabs: List[Dict[str, Any]] = []

        balance = taxable - threshold
        for limit, rate, label in self.SLABS:
            if balance <= 0:
                break
            amount = min(balance, limit)
            tax = amount * rate
            gross_tax += tax
            applied_slabs.append(
                {"label": label, "amount": amount, "rate": rate, "tax": tax}
            )
            balance -= amount

        return TaxResult(
            gross_income=gross,
            statutory_deduction=deduction,
            taxable_income=taxable,
            tax_free_threshold=threshold,
            gross_tax_liability=gross_tax,
            applied_slabs=applied_slabs,
        )