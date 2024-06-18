# from testMopta.testMopta import testMopta

# Low-dimensional
from testFunctions.testFull import testFull
DIM = 50

testFull('Ackley', N_EPOCH=20, DIM=DIM, N_ITERACTIONS=30)
testFull('Levy', N_EPOCH=20, DIM=DIM, N_ITERACTIONS=30)
testFull('Griewank', N_EPOCH=20, DIM=DIM, N_ITERACTIONS=30)

# from testReal.plotReal import plotReal
# plotReal('DNA', N_ITERACTIONS=115)

# # Linear Subspace
# from testFunctions.testLinsub import testLinsub
# testLinsub('Ackley', N_EPOCH=20)
# testLinsub('Levy', N_EPOCH=10)
# testLinsub('Griewank', N_EPOCH=20)

# Nonlinear Subspace
# from testFunctions.testNlinsub import testNlinsub
# DIM = 100
# testNlinsub(func_name='Ackley', N_EPOCH=3, DIM=DIM)
# testNlinsub(func_name='Levy', N_EPOCH=3, DIM=DIM)
# testNlinsub(func_name='Griewank', N_EPOCH=3, DIM=DIM)

# Plot
# from testFunctions.plotLinsub import plotNlinsub
# plotNlinsub(func_name='Ackley')

# NAS
# from testNAS.testNASBench import testNASBench
# testNASBench(N_EPOCH=3)

# LassoBench
# from testLasso.testLasso import testLasso
# # testLasso(pick_data='DNA', N_EPOCH=20, N_ITERACTIONS=200)
# testLasso(pick_data='synt_high', N_EPOCH=20, N_ITERACTIONS=200)

# Mopta
# from testMopta.testMopta import testMopta
# testMopta(N_EPOCH=20, TOTAL_TRIALS=300)
