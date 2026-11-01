import sys
from pathlib import Path

# Lets this file run directly by adding the project root (E:\TaxAssist) to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from legal_engine.testengine2 import TaxCalculationEngine, TaxpayerProfile


def test_employment_deduction_and_slabs():
    result = TaxCalculationEngine().compute_tax(
        TaxpayerProfile(category="General", employment_income=1500000.0)
    )
    assert result.gross_income == 1500000.0
    assert result.statutory_deduction == 500000.0
    assert result.taxable_income == 1000000.0
    # 600,000 above threshold: 300k @10% + 300k @15% = 75,000
    assert result.gross_tax_liability == pytest.approx(75000.0)
    assert len(result.applied_slabs) == 2


def test_below_threshold_pays_nothing():
    result = TaxCalculationEngine().compute_tax(
        TaxpayerProfile(category="General", other_income=300000.0)
    )
    assert result.gross_tax_liability == 0.0
    assert result.applied_slabs == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))