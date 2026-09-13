#!/usr/bin/env python3

# Direct calculation for the numerical example from the displayed q_{f_1}(T).
# This file does not construct Q_f and does not use transvectants.
# It begins with f_1(T) and q_{f_1}(T), forms
# a=q_{f_1}(theta)/f_1'(theta), and checks the two traces
# and the characteristic polynomial.

import sys

import sympy as sp


results = []


def check(name, condition):
    ok = bool(condition)
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name)


def multiplication_by_theta(f):
    # The basis is 1, theta, ..., theta^6.
    # The last column comes from the equation f(theta)=0.
    degree = f.degree()
    matrix = sp.zeros(degree)

    for column in range(degree - 1):
        matrix[column + 1, column] = 1

    for row in range(degree):
        matrix[row, degree - 1] = -f.nth(row)

    return matrix


def evaluate_at_matrix(polynomial, matrix):
    # Evaluate one coefficient at a time over the rational numbers.
    answer = sp.zeros(matrix.rows)
    identity = sp.eye(matrix.rows)

    for coefficient in polynomial.all_coeffs():
        answer = answer * matrix + coefficient * identity

    return answer


def monic_resultant(f, q, T, Y):
    resultant = sp.Poly(
        sp.resultant(f, Y * sp.diff(f, T) - q, T),
        Y,
        domain=sp.QQ,
    )
    return sp.Poly(resultant.as_expr() / resultant.LC(), Y, domain=sp.QQ)


T, Y = sp.symbols("T Y")

f1_expression = (
    T**7
    - 5 * T**6
    + 9 * T**5
    - 5 * T**4
    - 5 * T**3
    + 9 * T**2
    - 5 * T
    + 3
)
q_f1_expression = -(T - 1) ** 2 * (7 * T - 4)

f1 = sp.Poly(f1_expression, T, domain=sp.QQ)
q_f1 = sp.Poly(q_f1_expression, T, domain=sp.QQ)
f1_prime = sp.Poly(sp.diff(f1_expression, T), T, domain=sp.QQ)

check("f_1 is monic of degree seven", f1.degree() == 7 and f1.LC() == 1)
check("f_1 is separable", sp.gcd(f1, f1_prime).degree() == 0)
check("q_{f_1}(theta) is nonzero", sp.gcd(f1, q_f1).degree() == 0)

# This matrix represents multiplication by theta in Q[T]/(f_1).
theta = multiplication_by_theta(f1)
q_f1_theta = evaluate_at_matrix(q_f1, theta)
f1_prime_theta = evaluate_at_matrix(f1_prime, theta)

check("f_1'(theta) is invertible", f1_prime_theta.det() != 0)

# This is the element a=q_{f_1}(theta)/f_1'(theta) from the example.
a = q_f1_theta * f1_prime_theta.inv()

check("tr(a)=0", sp.trace(a) == 0)
check("tr(a^3)=0", sp.trace(a**3) == 0)

characteristic_polynomial = sp.Poly(
    a.charpoly(Y).as_expr(),
    Y,
    domain=sp.QQ,
)

expected_characteristic_polynomial = sp.Poly(
    Y**7
    - sp.Rational(1350885, 7619054) * Y**5
    - sp.Rational(282963, 7619054) * Y**3
    + sp.Rational(178605, 3809527) * Y**2
    + sp.Rational(89867, 30476216) * Y
    - sp.Rational(1655105, 60952432),
    Y,
    domain=sp.QQ,
)

check(
    "the characteristic polynomial is the displayed polynomial",
    characteristic_polynomial == expected_characteristic_polynomial,
)
check("the Y^6 coefficient vanishes", characteristic_polynomial.nth(6) == 0)
check("the Y^4 coefficient vanishes", characteristic_polynomial.nth(4) == 0)
check("the characteristic polynomial is irreducible over Q", characteristic_polynomial.is_irreducible)

resultant_polynomial = monic_resultant(f1_expression, q_f1_expression, T, Y)
check(
    "the resultant gives the same characteristic polynomial",
    resultant_polynomial == characteristic_polynomial,
)

number_passed = sum(results)
number_failed = len(results) - number_passed

print()
print(characteristic_polynomial.as_expr())
print()
print(f"{len(results)} checks, {number_passed} passed, {number_failed} failed.")

sys.exit(0 if number_failed == 0 else 1)
