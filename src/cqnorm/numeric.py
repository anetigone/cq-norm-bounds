"""Outward rounding for exact counts and verified certificate values."""

from decimal import Context, Decimal, ROUND_CEILING, ROUND_HALF_EVEN
from fractions import Fraction
import math


def round_up(value):
    """Convert an exact rational or finite decimal to an enclosing binary64."""
    rounded = float(value)
    if rounded < value:
        rounded = math.nextafter(rounded, math.inf)
    if not math.isfinite(rounded):
        raise OverflowError("Certificate or log budget exceeds the binary64 range")
    return rounded


def certificate_cost(weights, budgets):
    return round_up(sum((weight * Fraction(budget)
                         for weight, budget in zip(weights, budgets, strict=True) if weight), Fraction(0)))


def log2_count_upper(count, divisor=1):
    """Return an upper binary64 endpoint for log2(count)/divisor."""
    if not isinstance(count, int) or count < 0 or not isinstance(divisor, int) or divisor < 1:
        raise ValueError("Expected a nonnegative integer count and a positive integer divisor")
    if count == 0:
        return -math.inf
    if count & (count - 1) == 0:
        return round_up(Fraction(count.bit_length() - 1, divisor))
    # Decimal ln is correctly rounded to nearest. Adjacent decimals therefore
    # enclose the true logarithms. Both divisions use directed rounding.
    nearest = Context(prec=50, rounding=ROUND_HALF_EVEN)
    numerator = nearest.next_plus(nearest.ln(Decimal(count)))
    denominator = nearest.next_minus(nearest.ln(Decimal(2)))
    upward = Context(prec=50, rounding=ROUND_CEILING)
    return round_up(upward.divide(upward.divide(numerator, denominator), Decimal(divisor)))


def histogram_log_upper(histogram, order):
    if order == math.inf:
        return log2_count_upper(max((degree for degree, _ in histogram), default=0))
    if order not in (1, 2, 4, 8):
        raise ValueError("Expected an integer moment of order 1, 2, 4, 8, or infinity")
    q = int(order)
    return log2_count_upper(sum(count * degree**q for degree, count in histogram), q)
