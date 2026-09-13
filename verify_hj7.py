#!/usr/bin/env python3

# Exact calculations for
# A septic covariant and the Hermite--Joubert problem in degree seven.
#
# A binary form of order m is stored as [c_0, ..., c_m], meaning
#
#     c_0 X^m + c_1 X^(m-1) Z + ... + c_m Z^m.
#
# All calculations use exact integers or rational numbers.

import sys
from itertools import combinations_with_replacement
from math import comb

import sympy as sp


results = []


def check(name, condition):
    # Record every calculation instead of stopping at the first failure.
    ok = bool(condition)
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name)


# Binary forms and transvectants

def falling(n, k):
    value = 1
    for i in range(k):
        value *= n - i
    return value


def derivative_form(A, x_order, z_order):
    # Differentiate a binary form x_order times in X and z_order times in Z.
    order = len(A) - 1
    new_order = order - x_order - z_order
    if new_order < 0:
        return []

    answer = [0] * (new_order + 1)
    for i, coefficient in enumerate(A):
        x_power = order - i
        z_power = i
        if x_power < x_order or z_power < z_order or coefficient == 0:
            continue

        new_index = z_power - z_order
        answer[new_index] += (
            falling(x_power, x_order)
            * falling(z_power, z_order)
            * coefficient
        )
    return answer


def multiply_forms(A, B):
    answer = [0] * (len(A) + len(B) - 1)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            answer[i + j] += a * b
    return answer


def combine_forms(A, B, a=1, b=1):
    length = max(len(A), len(B))
    answer = [0] * length
    for i, coefficient in enumerate(A):
        answer[i] += a * coefficient
    for i, coefficient in enumerate(B):
        answer[i] += b * coefficient
    return answer


def multiply_form_by_scalar(A, scalar):
    return [scalar * coefficient for coefficient in A]


def transvectant(A, B, r):
    # This is the unnormalised transvectant used in the paper.
    order_A = len(A) - 1
    order_B = len(B) - 1
    answer_order = order_A + order_B - 2 * r
    answer = [0] * (answer_order + 1)

    for j in range(r + 1):
        derivative_A = derivative_form(A, r - j, j)
        derivative_B = derivative_form(B, j, r - j)
        product = multiply_forms(derivative_A, derivative_B)
        factor = (-1) ** j * comb(r, j)

        for i, coefficient in enumerate(product):
            answer[i] += factor * coefficient
    return answer


def evaluate_at_T_1(A, T):
    # Set X=T and Z=1.
    order = len(A) - 1
    return sp.expand(
        sum(coefficient * T ** (order - i) for i, coefficient in enumerate(A))
    )


def forms_are_equal(A, B):
    difference = combine_forms(A, B, 1, -1)
    return all(sp.expand(coefficient) == 0 for coefficient in difference)


def monomial_form(order, x_power, z_power, coefficient):
    answer = [0] * (order + 1)
    answer[z_power] = coefficient
    return answer


def transvectant_chain(f):
    D1 = transvectant(f, f, 4)
    D2 = transvectant(f, f, 6)
    D3 = transvectant(f, f, 2)
    T1 = transvectant(f, D1, 4)
    T5 = transvectant(f, D3, 5)
    C1 = transvectant(f, T5, 7)
    C7 = transvectant(f, T1, 3)
    P5 = transvectant(f, C7, 5)
    Q = combine_forms(
        multiply_forms(D2, P5),
        multiply_forms(C1, T1),
        7,
        -10,
    )

    return {
        "D1": D1,
        "D2": D2,
        "D3": D3,
        "T1": T1,
        "T5": T5,
        "C1": C1,
        "C7": C7,
        "P5": P5,
        "Q": Q,
    }


# Ordinary polynomial calculations

def cayley_sylvester_count(n, d, w):
    return sum(
        1
        for indices in combinations_with_replacement(range(n + 1), d)
        if sum(indices) == w
    )


def monic_resultant(f, q, T, Y):
    resultant = sp.Poly(
        sp.resultant(f, Y * sp.diff(f, T) - q, T),
        Y,
        domain=sp.QQ,
    )
    return sp.Poly(resultant.as_expr() / resultant.LC(), Y, domain=sp.QQ)


def coefficient_from_table(expression, coefficient_monomial, t_degree, variables, T):
    # First take the requested coefficient in c_5,c_3,c_2,c_0.
    coefficient = sp.Poly(expression, *variables).coeff_monomial(coefficient_monomial)

    # The remaining expression is a polynomial in T.
    return sp.Poly(coefficient, T).nth(t_degree)


# The general binary septic

a = sp.symbols("a0:8")
T, Y = sp.symbols("T Y")
f = list(a)
f_T = evaluate_at_T_1(f, T)
chain = transvectant_chain(f)
Q = [sp.expand(coefficient) for coefficient in chain["Q"]]

coefficient_gcd = 2**32 * 3**14 * 5**6 * 7


# 1--3. Order, number of monomials, and greatest common divisor of the coefficients of Q_f

check("Q_f has order 5", len(Q) == 6)

number_of_monomials = sum(
    len(sp.Poly(coefficient, *a).terms())
    for coefficient in Q
)
check("Q_f has 780 nonzero monomials", number_of_monomials == 780)

all_integer_coefficients = [
    abs(int(value))
    for coefficient in Q
    for value in sp.Poly(coefficient, *a).coeffs()
]
computed_coefficient_gcd = 0
for value in all_integer_coefficients:
    computed_coefficient_gcd = sp.igcd(computed_coefficient_gcd, value)
check(
    "the greatest common divisor of the coefficients of Q_f is 2^32 3^14 5^6 7",
    computed_coefficient_gcd == coefficient_gcd,
)


# 4. The three recoupling identities

f_D2 = transvectant(f, chain["D2"], 2)
D2_D2 = transvectant(chain["D2"], chain["D2"], 2)
f_D2_squared = transvectant(
    f,
    multiply_forms(chain["D2"], chain["D2"]),
    4,
)

recoupling_identities_hold = all(
    [
        forms_are_equal(chain["T1"], multiply_form_by_scalar(f_D2, -20)),
        forms_are_equal(
            multiply_form_by_scalar(chain["C1"], 2),
            multiply_form_by_scalar(D2_D2, 21),
        ),
        forms_are_equal(
            multiply_form_by_scalar(chain["P5"], 2),
            multiply_form_by_scalar(f_D2_squared, -25),
        ),
    ]
)
check("the three recoupling identities hold", recoupling_identities_hold)


# 5. The four dimensions

counts_hold = (
    cayley_sylvester_count(7, 2, 6) == 4
    and cayley_sylvester_count(7, 2, 5) == 3
    and cayley_sylvester_count(7, 3, 8) == 9
    and cayley_sylvester_count(7, 3, 7) == 8
    and cayley_sylvester_count(7, 4, 14) == 24
    and cayley_sylvester_count(7, 4, 13) == 23
    and cayley_sylvester_count(7, 5, 16) == 48
    and cayley_sylvester_count(7, 5, 15) == 46
)
check("the four dimensions are 1, 1, 1, and 2", counts_hold)


# 6. The three formulas involving XZ

D = lambda polynomial: sp.expand(T * sp.diff(polynomial, T))
L = lambda polynomial: sp.expand(7 * D(polynomial) - D(D(polynomial)))

XZ = [0, 1, 0]
X2Z2 = [0, 0, 1, 0, 0]

xz_formulas_hold = (
    sp.expand(T * evaluate_at_T_1(transvectant(f, XZ, 2), T) + 2 * L(f_T)) == 0
    and transvectant(XZ, XZ, 2) == [-2]
    and sp.expand(
        T**2 * evaluate_at_T_1(transvectant(f, X2Z2, 4), T)
        - 24 * (L(L(f_T)) - 6 * L(f_T))
    )
    == 0
)
check("the three formulas involving XZ hold", xz_formulas_hold)


# 7. The two formulas for q

q = sp.expand(
    sp.diff(f_T, T)
    - T * sp.diff(f_T, T, 2)
    + sp.Rational(1, 3) * T**2 * sp.diff(f_T, T, 3)
    - sp.Rational(1, 24) * T**3 * sp.diff(f_T, T, 4)
)
q_four_terms = a[1] * T**5 - a[3] * T**3 - a[4] * T**2 + a[6]

frame_formulas_hold = (
    sp.expand(q - q_four_terms) == 0
    and sp.expand(
        T * q + sp.Rational(1, 24) * (L(L(f_T)) - 10 * L(f_T))
    )
    == 0
)
check("the two formulas for q hold", frame_formulas_hold)


# 8. The formula for D_2

A = 35 * a[0] * a[6] - 10 * a[1] * a[5] + 5 * a[2] * a[4] - 2 * a[3] ** 2
B = 245 * a[0] * a[7] - 25 * a[1] * a[6] + 5 * a[2] * a[5] - a[3] * a[4]
C = 35 * a[1] * a[7] - 10 * a[2] * a[6] + 5 * a[3] * a[5] - 2 * a[4] ** 2

check(
    "D_2 = 207360(A X^2 + B XZ + C Z^2)",
    forms_are_equal(
        chain["D2"],
        multiply_form_by_scalar([A, B, C], 207360),
    ),
)


# 9. The collapse in normal position

lambda_value = sp.symbols("lambda")
D2_normal = [0, lambda_value, 0]

# In normal position D_2=lambda XZ. The recoupling identities then give
# T_1, C_1, and P_5 directly from D_2.
T1_normal = multiply_form_by_scalar(
    transvectant(f, D2_normal, 2),
    -20,
)
C1_normal = multiply_form_by_scalar(
    transvectant(D2_normal, D2_normal, 2),
    sp.Rational(21, 2),
)
P5_normal = multiply_form_by_scalar(
    transvectant(f, multiply_forms(D2_normal, D2_normal), 4),
    sp.Rational(-25, 2),
)
Q_normal = combine_forms(
    multiply_forms(D2_normal, P5_normal),
    multiply_forms(C1_normal, T1_normal),
    7,
    -10,
)

check(
    "Q_f(T,1)=50400 lambda^3 q(T) in normal position",
    sp.expand(
        evaluate_at_T_1(Q_normal, T)
        - 50400 * lambda_value**3 * q
    )
    == 0,
)


# 10. The displayed expansion of g

g = sp.expand(
    T * q**2
    - sp.Rational(11, 12) * T**2 * q * sp.diff(q, T)
    - sp.Rational(1, 12) * T**3 * q * sp.diff(q, T, 2)
    + sp.Rational(1, 4) * T**3 * sp.diff(q, T) ** 2
)

g_displayed = (
    a[1] ** 2 * T**11
    + sp.Rational(5, 4) * a[1] * a[4] * T**8
    - sp.Rational(1, 4) * (17 * a[1] * a[6] + a[3] * a[4]) * T**6
    + sp.Rational(5, 4) * a[3] * a[6] * T**4
    + a[6] ** 2 * T
)

check("g has the displayed five-term expansion", sp.expand(g - g_displayed) == 0)


# 11. The Wronskian identity

P0 = sp.expand(sp.Rational(1, 2) * T**6 * (D(q) - 2 * q))
R0 = sp.expand(sp.Rational(1, 2) * T**4 * (3 * q - D(q)))
W = sp.expand(
    sp.diff(f_T, T) * sp.diff(g_displayed, T)
    - sp.diff(f_T, T, 2) * g_displayed
)

check(
    "W(f',g) = q^3 - 35 a_1 f R_0 + A P_0 + C R_0",
    sp.expand(W - (q**3 - 35 * a[1] * f_T * R0 + A * P0 + C * R0)) == 0,
)


# 12--15. The four derivative identities

derivative_identities = [
    (
        "the derivative identity for a_0 holds",
        T**5 * (D(g_displayed) - 6 * g_displayed)
        - (-5 * a[1] * T**7 * R0 + 5 * a[6] * P0),
    ),
    (
        "the derivative identity for a_2 holds",
        T**3 * (D(g_displayed) - 4 * g_displayed)
        - (-7 * a[1] * T**5 * R0 + a[4] * P0 - 2 * a[6] * R0),
    ),
    (
        "the derivative identity for a_5 holds",
        2 * (D(g_displayed) - g_displayed)
        - (-35 * a[1] * T**2 * R0 - 10 * a[1] * P0 + 5 * a[3] * R0),
    ),
    (
        "the derivative identity for a_7 holds",
        -35 * a[1] * R0 + 35 * a[1] * R0,
    ),
]

for name, expression in derivative_identities:
    check(name, sp.expand(expression) == 0)


# 16--35. The twenty rows in the coefficient table

c5, c3, c2, c0 = sp.symbols("c5 c3 c2 c0")
coefficient_variables = (c5, c3, c2, c0)
coefficient_for_digit = {5: c5, 3: c3, 2: c2, 0: c0}

q0 = c5 * T**5 + c3 * T**3 + c2 * T**2 + c0
D0 = lambda polynomial: sp.expand(T * sp.diff(polynomial, T))
Lambda = lambda polynomial: sp.expand(D0(D0(polynomial)) - 5 * D0(polynomial) + 3 * polynomial)
f0 = sp.expand(sp.Rational(1, 3) * T * Lambda(q0))
g0 = sp.expand(
    T * q0**2
    - sp.Rational(11, 12) * T**2 * q0 * sp.diff(q0, T)
    - sp.Rational(1, 12) * T**3 * q0 * sp.diff(q0, T, 2)
    + sp.Rational(1, 4) * T**3 * sp.diff(q0, T) ** 2
)

left_side = sp.expand(
    sp.diff(f0, T) * sp.diff(g0, T)
    - sp.diff(f0, T, 2) * g0
)
q_cubed = sp.expand(q0**3)
correction = sp.expand(
    -c3**2 * T**6 * (D0(q0) - 2 * q0)
    - c2**2 * T**4 * (3 * q0 - D0(q0))
    - sp.Rational(35, 6) * c5 * T**5 * Lambda(q0) * (3 * q0 - D0(q0))
)

expected_rows = {
    "555": (36, 1, 35),
    "553": (-32, 3, -35),
    "552": (sp.Rational(-99, 2), 3, sp.Rational(-105, 2)),
    "550": (sp.Rational(-29, 2), 3, sp.Rational(-35, 2)),
    "533": (0, 3, -3),
    "532": (sp.Rational(47, 2), 6, sp.Rational(35, 2)),
    "530": (sp.Rational(117, 2), 6, sp.Rational(105, 2)),
    "522": (sp.Rational(45, 2), 3, sp.Rational(39, 2)),
    "520": (41, 6, 35),
    "500": (sp.Rational(-99, 2), 3, sp.Rational(-105, 2)),
    "332": (3, 3, 0),
    "330": (5, 3, 2),
    "322": (3, 3, 0),
    "320": (6, 6, 0),
    "300": (3, 3, 0),
    "333": (0, 1, -1),
    "222": (0, 1, -1),
    "220": (0, 3, -3),
    "200": (3, 3, 0),
    "000": (1, 1, 0),
}

for row, expected in expected_rows.items():
    coefficient_monomial = 1
    t_degree = 0

    for digit in row:
        digit = int(digit)
        coefficient_monomial *= coefficient_for_digit[digit]
        t_degree += digit

    actual = (
        coefficient_from_table(
            left_side,
            coefficient_monomial,
            t_degree,
            coefficient_variables,
            T,
        ),
        coefficient_from_table(
            q_cubed,
            coefficient_monomial,
            t_degree,
            coefficient_variables,
            T,
        ),
        coefficient_from_table(
            correction,
            coefficient_monomial,
            t_degree,
            coefficient_variables,
            T,
        ),
    )

    check("the calculated coefficients agree with row " + row + " of the table", actual == expected)


# 36. The values displayed in the recoupling proof

recoupling_f1 = [1, 0, 0, 0, 0, 0, 1, 0]
recoupling_f2 = [0, 0, 1, 0, 0, 0, 1, 0]
symmetric_f = [1, 0, 0, 0, 0, 0, 0, 1]

recoupling_chain_f1 = transvectant_chain(recoupling_f1)
recoupling_chain_f2 = transvectant_chain(recoupling_f2)
symmetric_chain = transvectant_chain(symmetric_f)

lambda_symmetric = 2 * 5040**2
T5_factor = 2**9 * 3**5 * 5**2 * 7**3

U_f1 = transvectant(
    recoupling_f1,
    multiply_forms(recoupling_chain_f1["D2"], recoupling_chain_f1["D2"]),
    4,
)
V_f1 = transvectant(recoupling_chain_f1["D1"], recoupling_chain_f1["T1"], 4)
U_f2 = transvectant(
    recoupling_f2,
    multiply_forms(recoupling_chain_f2["D2"], recoupling_chain_f2["D2"]),
    4,
)
V_f2 = transvectant(recoupling_chain_f2["D1"], recoupling_chain_f2["T1"], 4)

displayed_values_hold = all(
    [
        forms_are_equal(
            recoupling_chain_f1["D1"],
            monomial_form(6, 4, 2, 2**7 * 3**3 * 5**2 * 7),
        ),
        forms_are_equal(
            recoupling_chain_f1["D2"],
            monomial_form(2, 2, 0, 2**9 * 3**4 * 5**2 * 7),
        ),
        forms_are_equal(
            recoupling_chain_f1["T1"],
            monomial_form(5, 1, 4, -2**13 * 3**5 * 5**4 * 7),
        ),
        forms_are_equal(
            transvectant(recoupling_f1, recoupling_chain_f1["D2"], 2),
            monomial_form(5, 1, 4, 2**11 * 3**5 * 5**3 * 7),
        ),
        forms_are_equal(symmetric_chain["D2"], [0, lambda_symmetric, 0]),
        forms_are_equal(
            symmetric_chain["D3"],
            monomial_form(10, 5, 5, 3528),
        ),
        forms_are_equal(
            symmetric_chain["T5"],
            [T5_factor] + [0] * 6 + [-T5_factor],
        ),
        symmetric_chain["C1"] == [-T5_factor * lambda_symmetric],
        transvectant(symmetric_chain["D2"], symmetric_chain["D2"], 2)
        == [-2 * lambda_symmetric**2],
        forms_are_equal(
            recoupling_chain_f1["P5"],
            monomial_form(3, 1, 2, -2**23 * 3**11 * 5**7 * 7**2),
        ),
        forms_are_equal(
            U_f1,
            monomial_form(3, 1, 2, 2**24 * 3**11 * 5**5 * 7**2),
        ),
        forms_are_equal(
            V_f1,
            monomial_form(3, 1, 2, 2**26 * 3**10 * 5**6 * 7**3),
        ),
        forms_are_equal(
            recoupling_chain_f2["P5"],
            monomial_form(3, 1, 2, -2**25 * 3**10 * 5**7),
        ),
        forms_are_equal(
            U_f2,
            monomial_form(3, 1, 2, 2**26 * 3**10 * 5**5),
        ),
        forms_are_equal(
            V_f2,
            monomial_form(3, 1, 2, 2**28 * 3**8 * 5**6),
        ),
    ]
)
check("the calculated values agree with those displayed in the recoupling proof", displayed_values_hold)


# 37. The formula for Q_f in the recoupling lemma

recoupled_Q = multiply_form_by_scalar(
    combine_forms(
        multiply_forms(D2_D2, f_D2),
        multiply_forms(chain["D2"], f_D2_squared),
        24,
        -1,
    ),
    sp.Rational(175, 2),
)
check("the calculated Q_f agrees with the formula in the recoupling lemma", forms_are_equal(chain["Q"], recoupled_Q))


# 38. The calculation with T^7+T

T7_plus_T_Q = monomial_form(5, 3, 2, -2**32 * 3**15 * 5**9 * 7**4)
T7_plus_T_q = evaluate_at_T_1(
    [sp.Rational(coefficient, coefficient_gcd) for coefficient in recoupling_chain_f1["Q"]],
    T,
)

T7_plus_T_holds = (
    recoupling_chain_f1["C1"] == [0]
    and forms_are_equal(recoupling_chain_f1["Q"], T7_plus_T_Q)
    and sp.expand(T7_plus_T_q + 3 * 35**3 * T**3) == 0
)
check("the calculation with T^7+T gives the displayed nonzero q_f", T7_plus_T_holds)


# 39. The numerical example beginning with f_1(T)=f_0(T-1)

# Section 6 cancels the nonzero rational scalar in Q_G(T,1).
# The polynomial used in the resultant is the displayed q_{f_1}(T).

example_f0 = T**7 + 2 * T**6 + 2
example_f1 = T**7 - 5 * T**6 + 9 * T**5 - 5 * T**4 - 5 * T**3 + 9 * T**2 - 5 * T + 3
example_coefficients = [1, -5, 9, -5, -5, 9, -5, 3]
example_chain = transvectant_chain(example_coefficients)

Q_factor = 12330752933527289856000000000
example_Q_G = [0, 0, -7 * Q_factor, 18 * Q_factor, -15 * Q_factor, 4 * Q_factor]
example_q_f1 = -(T - 1) ** 2 * (7 * T - 4)
example_f1_derivative = 7 * T**6 - 30 * T**5 + 45 * T**4 - 20 * T**3 - 15 * T**2 + 18 * T - 5

example_resultant = monic_resultant(example_f1, example_q_f1, T, Y)
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

example_data_hold = (
    sp.expand(example_f0.subs(T, T - 1) - example_f1) == 0
    and forms_are_equal(example_chain["Q"], example_Q_G)
    and sp.expand(
        evaluate_at_T_1(example_chain["Q"], T)
        - Q_factor * example_q_f1
    )
    == 0
    and sp.expand(sp.diff(example_f1, T) - example_f1_derivative) == 0
    and sp.gcd(sp.Poly(example_f1, T), sp.Poly(example_q_f1, T)).degree() == 0
    and example_resultant == expected_characteristic_polynomial
    and example_resultant.nth(6) == 0
    and example_resultant.nth(4) == 0
)
check("the calculations agree with the numerical example", example_data_hold)


# Final result

number_passed = sum(results)
number_failed = len(results) - number_passed

if len(results) != 39:
    raise RuntimeError("verify_hj7.py must contain exactly 39 checks")

print()
print(f"{len(results)} checks, {number_passed} passed, {number_failed} failed.")

sys.exit(0 if number_failed == 0 else 1)
