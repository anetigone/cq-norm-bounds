"""Focused checks for count extraction, outward rounding, and cached solves."""

from decimal import Decimal, localcontext
from fractions import Fraction
import math

import duckdb

from run_jobjoin_views import degree
from run_mixed_holdout import TARGETS, measured_norm
from cqnorm import coverage_flow, shannon
from cqnorm.numeric import certificate_cost, histogram_log_upper, log2_count_upper


def check_log(upper, count, divisor=1):
    with localcontext() as ctx:
        ctx.prec = 120
        reference = Decimal(count).ln() / Decimal(2).ln() / divisor
    if Decimal(upper) < reference or upper > math.nextafter(float(reference), math.inf):
        raise AssertionError(f'Unexpected log enclosure for order {divisor}')


def main():
    assert histogram_log_upper([], 8) == -math.inf
    assert histogram_log_upper([(1, 1)], 8) == 0
    assert histogram_log_upper([(8, 4)], math.inf) == 3
    assert histogram_log_upper([(8, 256)], 8) == 4
    huge = (1 << 2048) + 17
    histogram = [(1, 10**30), (huge, 5)]
    for order in (1, 2, 4, 8):
        check_log(histogram_log_upper(histogram, order), 10**30 + 5 * huge**order, order)
    assert log2_count_upper((1 << 1024) + 1) > 1024
    assert log2_count_upper(1 << 1024) == 1024

    degree_value = 1 << 62
    first_moment = degree_value**2 + 1
    grouped = (f'SELECT * FROM (VALUES (0, 0, {degree_value}), (0, 1, 1), (1, 0, 2)) '
               'AS cells(v00, v01, degree)')
    with duckdb.connect(config={'threads': 1}) as db:
        for target, (count, divisor) in zip(TARGETS, ((degree_value**2 + 4, 2),
                                                     (first_moment**2 + 16, 4),
                                                     (first_moment, 2)), strict=True):
            upper, _ = measured_norm(db, grouped, target, 10)
            check_log(upper, count, divisor)
            empty, _ = measured_norm(db, f'SELECT * FROM ({grouped}) WHERE false', target, 10)
            assert empty is None
        try:
            db.execute("SELECT '170141183460469231731687303715884105727'::HUGEINT + 1::HUGEINT")
        except duckdb.OutOfRangeException:
            pass
        else:
            raise AssertionError('HUGEINT overflow must raise')

    first = certificate_cost((Fraction(1, 3),), (1,))
    second = certificate_cost((Fraction(7, 3),), (first,))
    assert Fraction(first) >= Fraction(1, 3)
    assert Fraction(second) >= Fraction(first) * Fraction(7, 3)
    assert Fraction(second) >= Fraction(7, 9)

    target = degree(('a',))
    model = coverage_flow.FlowProgram(('a',), (target,), target, time_limit_ms=1000)
    assert model.solve((-math.inf,)).status == 'empty'
    for budget in (0, 1, log2_count_upper(huge), 1, 0):
        inputs = [(target, budget)]
        fresh = coverage_flow.solve(('a',), inputs, target, method='flow', time_limit_ms=1000)
        cached = model.solve((budget,))
        reference = shannon.solve(('a',), inputs, target, 1000)
        assert fresh.status == cached.status == reference.status == 'optimal'
        assert fresh.log2_bound == cached.log2_bound == reference.log2_bound == budget
    print('Count, mixed, overflow, certificate rounding, and cached-model checks passed')


if __name__ == '__main__':
    main()
