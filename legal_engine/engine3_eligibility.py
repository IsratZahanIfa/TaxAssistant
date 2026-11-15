from pydantic import BaseModel


class InvestmentPortfolio(BaseModel):
    sanchayapatra: float = 0.0
    dps: float = 0.0  # statutory limit Tk 120,000/year
    life_insurance: float = 0.0
    stocks_mutual_funds: float = 0.0


class RebateResult(BaseModel):
    total_actual_investment: float
    eligible_investment: float
    rebate_amount: float
    final_tax_payable: float
    minimum_tax_applied: bool


class InvestmentEligibilityEngine:
    # NOTE: verify these rates/caps against the current Finance Act.
    def __init__(self):
        self.dps_limit = 120000.0
        self.max_rebate_cap = 750000.0
        self.rebate_rate_on_investment = 0.10
        self.rebate_rate_on_income = 0.03
        self.minimum_tax = 5000.0

    def calculate_rebate(
        self, taxable_income: float, gross_tax: float, portfolio: InvestmentPortfolio
    ) -> RebateResult:
        total_actual = (
            portfolio.sanchayapatra + portfolio.dps
            + portfolio.life_insurance + portfolio.stocks_mutual_funds
        )
        capped_dps = min(portfolio.dps, self.dps_limit)
        eligible_inv = (
            portfolio.sanchayapatra + capped_dps
            + portfolio.life_insurance + portfolio.stocks_mutual_funds
        )

        tax_rebate = min(
            eligible_inv * self.rebate_rate_on_investment,
            taxable_income * self.rebate_rate_on_income,
            self.max_rebate_cap,
        )
        # A rebate can never exceed the tax actually due
        tax_rebate = min(tax_rebate, gross_tax)

        net_tax = gross_tax - tax_rebate
        min_tax = self.minimum_tax if taxable_income > 0 else 0.0

        if taxable_income > 0 and net_tax < min_tax:
            final_tax, min_applied = min_tax, True
        else:
            final_tax, min_applied = max(0.0, net_tax), False

        return RebateResult(
            total_actual_investment=total_actual,
            eligible_investment=eligible_inv,
            rebate_amount=tax_rebate,
            final_tax_payable=final_tax,
            minimum_tax_applied=min_applied,
        )