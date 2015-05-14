# -*- coding: utf-8 -*-
"""Optimal trajectory planner

This module is written to automatically generate symbolic formulas for the 
problem below:
A particle moves for certain length with n-1 order derivative of displacement 
is zero at start and end, while keeping the i-th (i<=n) order derivative 
limited during the motion.

For example, the paper below gives an solution on 4th order.
	Lambrechts, Paul, Matthijs Boerlage, and Maarten Steinbuch. "Trajectory 
	planning and feedforward design for high performance motion systems." 
	Feedback 14 (2004): 15.
"""

from sympy import symbols, integrate, poly
from sympy.core.numbers import Zero
from itertools import islice, repeat, chain
from functools import partial
from collections import OrderedDict

def optimal_trajectory(variables):
	"""Generate constraints and recursive equations

	Args:
		variables (list): string list of the variable names of n, n-1 ... 0th 
			order derivative of the displacement. (e.g. ['a', 'v', 'x'])

	Return:
		t: sympy symbol object for time
		xs: sympy symbol object list for variables, from n order to 0 order
		xlimits: sympy symbol object list for limits on variables, from n 
			order to 0 order
		constraints: OrderedDict with period time as key and constraints as 
			value, every formula in the constraints should be non-negative
		ps: recursive relationship of each period
	"""

	xs = symbols(variables, real=True)
	xms = symbols([vn+'_m' for vn in variables], real=True)
	xlimits = symbols([vn+'_l' for vn in variables], real=True)
	t, dt = symbols('t dt', real=True)
	txs = symbols(['t_'+vn for vn in variables], real=True)
	order = len(variables)

	recur = {t: t + dt} #recursive relationship
	recur.update({xm: xm for xm in xms})
	expr = 0.0
	for x in xs:
		expr = x + integrate(expr, (dt, 0, dt))
		recur[x] = expr

	ls = [0.0] * order
	ls[0] = xlimits[0]
	def profilier(n=order-1, sign=1, first=True):
		if n < 0 or (n == order - 2 and not first):
			return
		yield from profilier(n-1, sign, first)
		para = {dt: txs[n], xs[0]: sign * ls[n]}
		step = {x: expr.subs(para) for x, expr in recur.items()}
		if first:
			step.update(islice(zip(xms, xs), n, None))
		yield step
		yield from profilier(n-1, -sign, False)

	ps = [{x: Zero() for x in chain([t], xs, xms)}]
	ps[0][xs[0]] = xlimits[0]
	ps.extend(list(profilier()))
	x0 = {}
	for p in ps:
		x0 = {x: expr.subs(x0) for x, expr in p.items()}

	constraints_global = [xl - x0[xm] for xm, xl in zip(xms, xlimits)]
	def constraints_tx_iter():
		para = list(zip(txs, [0.0] * order))
		c_global = constraints_global
		while len(para) > 1:
			tx = txs[-len(para)]
			para.pop(0)
			c_global.pop(0)
			yield tx, [poly(c.subs(para), tx).all_coeffs() for c in c_global]

	constraints = OrderedDict([item for item in constraints_tx_iter()])
	return t, xs, xlimits, constraints, ps[:-1]