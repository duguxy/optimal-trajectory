from trajectory import optimal_trajectory
from codegen import CodeGenerator

variables = list('djavx')
result = optimal_trajectory(variables)
with open('output.py', 'w') as f:
	cg = CodeGenerator(*result)
	cg.write(f)