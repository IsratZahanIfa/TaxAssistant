import sys
from pathlib import Path

# Lets this file run directly (python legal_engine\engine4.py) by adding the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List

from pydantic import BaseModel

from legal_engine.engine2 import TaxCalculationEngine, TaxpayerProfile
from legal_engine.engine3 import InvestmentEligibilityEngine, InvestmentPortfolio


class FinancialConstraints(BaseModel):
    disposable_savings_capacity: float
    risk_tolerance: str  # Conservative, Balanced, Growth


class ScenarioOutput(BaseModel):
    name: str
    total_investment: float
    allocation: Dict[str, float]
    tax_rebate: float
    final_tax_payable: float
    tbi_metric: float  # Tax Benefit per Taka Invested
    eic_metric: float  # Effective Investment Cost
    net_remaining_disposable: float


class InvestmentPlannerEngine:
    def __init__(self):
        self.tax_engine = TaxCalculationEngine()
        self.eligibility_engine = InvestmentEligibilityEngine()

    def _allocate(self, amount: float, weights: Dict[str, float]) -> Dict[str, float]:
        """Split `amount` by weights (sum to 1); DPS above its legal cap goes to sanchayapatra."""
        alloc = {
            "sanchayapatra": 0.0,
            "dps": 0.0,
            "stocks_mutual_funds": 0.0,
            "life_insurance": 0.0,
        }
        for k, w in weights.items():
            alloc[k] = amount * w
        excess = max(0.0, alloc["dps"] - self.eligibility_engine.dps_limit)
        alloc["dps"] -= excess
        alloc["sanchayapatra"] += excess
        return alloc

    def plan(
        self, profile: TaxpayerProfile, constraints: FinancialConstraints
    ) -> List[ScenarioOutput]:
        tax_res = self.tax_engine.compute_tax(profile)
        taxable_inc = tax_res.taxable_income
        gross_tax = tax_res.gross_tax_liability

        eng = self.eligibility_engine
        max_useful = (taxable_inc * eng.rebate_rate_on_income) / eng.rebate_rate_on_investment
        budget = min(constraints.disposable_savings_capacity, max_useful)

        # Allocation weights sum to 1 of each scenario's own amount
        scenarios_config = [
            ("Current Baseline", 0.0, {}),
            ("Conservative Strategy", 0.5 * budget, {"dps": 0.4, "sanchayapatra": 0.6}),
            (
                "Balanced Optimal Strategy",
                1.0 * budget,
                {"dps": 0.2, "sanchayapatra": 0.5, "stocks_mutual_funds": 0.3},
            ),
        ]

        output = []
        for name, inv_amount, weights in scenarios_config:
            alloc = self._allocate(inv_amount, weights)
            port = InvestmentPortfolio(**alloc)
            rebate_res = eng.calculate_rebate(taxable_inc, gross_tax, port)

            tbi = rebate_res.rebate_amount / inv_amount if inv_amount > 0 else 0.0
            eic = inv_amount - rebate_res.rebate_amount
            rem_cash = (
                constraints.disposable_savings_capacity
                - inv_amount
                - rebate_res.final_tax_payable
            )

            output.append(
                ScenarioOutput(
                    name=name,
                    total_investment=inv_amount,
                    allocation=alloc,
                    tax_rebate=rebate_res.rebate_amount,
                    final_tax_payable=rebate_res.final_tax_payable,
                    tbi_metric=round(tbi, 4),
                    eic_metric=round(eic, 2),
                    net_remaining_disposable=round(rem_cash, 2),
                )
            )
        return output


if __name__ == "__main__":
    profile = TaxpayerProfile(category="General", employment_income=1500000.0)
    constraints = FinancialConstraints(
        disposable_savings_capacity=300000.0, risk_tolerance="Balanced"
    )
    for sc in InvestmentPlannerEngine().plan(profile, constraints):
        print(
            f"{sc.name}: invested {sc.total_investment:,.0f}, "
            f"rebate {sc.tax_rebate:,.0f}, tax payable {sc.final_tax_payable:,.0f}"
        )