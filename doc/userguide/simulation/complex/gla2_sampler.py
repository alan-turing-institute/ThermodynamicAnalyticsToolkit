import math
import numpy as np
import TATi.simulation as tati

np.random.seed(426)

def gla2_update_step(nn, momenta, old_gradients, step_width, beta, gamma):
    """Implementation of GLA2 update step using TATi's simulation interface.
    
    Note:
        Parameters are contained inside nn. For momenta we use
        python variables as the evaluation of the loss does not
        depend on them.

    Args:
      nn: ref to tati simulation instance
      momenta: numpy array of parameters
      old_gradients: gradients evaluated at last step
      step_width: step width for sampling step
      beta: inverse temperature
      gamma: friction constant

    Returns:
      updated gradients and momenta

    """

    # 1. p_{n+\tfrac 1 2} = p_n - \tfrac {\lambda}{2} \nabla_x L(x_n)
    momenta -= .5*step_width * old_gradients

    # 2. x_{n+1} = x_n + \lambda p_{n+\tfrac 1 2}
    nn.parameters = nn.parameters + step_width * momenta

    # \nabla_x L(x_{n+1})
    gradients = nn.gradients()

    # 3. \widehat{p}_{n+1} = p_{n+\tfrac 1 2} - \tfrac {\lambda}{2} \nabla_x L(x_{n+1})
    momenta -= .5*step_width * gradients

    # 4. p_{n+1} = \alpha \widehat{p}_{n+1} + \sqrt{\frac{1-\alpha^2}{\beta}} \cdot \eta_n
    alpha = math.exp(-gamma*step_width)
    momenta = alpha * momenta + \
              math.sqrt((1.-math.pow(alpha,2.))/beta) * np.random.standard_normal(momenta.shape)

    return gradients, momenta

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    fix_parameters="layer1/biases/Variable:0=0.;output/biases/Variable:0=0.",
    hidden_dimension=[1],
    input_columns=["x1"],
    seed=426,
)

print("We have "+str(nn.num_parameters())+" parameters.")

gamma = 10
beta = 1e3
momenta = np.zeros((nn.num_parameters()))
old_gradients = np.zeros((nn.num_parameters()))

for i in range(100):
    old_gradients, momenta = gla2_update_step(
        nn, momenta, old_gradients, step_width=1e-1, beta=beta, gamma=gamma)
    print("Step #"+str(i)+": "+str(nn.loss())+" at " \
        +str(nn.parameters)+", gradients "+str(old_gradients))
