import mdp

m1 = mdp.make2DProblem()
m1.valueIteration()
m1.printValues()

m2 = mdp.makeRNProblem()
m2.policyIteration()
m1.printActions()
