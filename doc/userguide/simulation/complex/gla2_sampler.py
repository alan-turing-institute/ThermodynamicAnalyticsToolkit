import math
import numpy as np
import TATi.simulation as tati

np.random.seed(426)

def gla2_update_step(nn, momenta, old_gradients, step_width, beta, gamma):
    """ Implementation of GLA2 update step using TATi's simulation interface.

    Note:
        Parameters are contained inside nn. For momenta we use
        python variables as the evaluation of the loss does not
        depend on them.

    :param nn: ref to tati simulation instance
    :param momenta: numpy array of parameters
    :param old_gradients: gradients evaluated at last step
    :param step_width: step width for sampling step
    :param beta: inverse temperature
    :param gamma: friction constant
    :return: updated gradients and momenta
    """

    # 1. p_{n+\tfrac 1 2} = p_n - \tfrac {\lambda}{2} \nabla_x L(x_n)
    momenta -= .5*step_width * old_gradients

    # 2. x_{n+1} = x_n + \lambda p_{n+\tfrac 1 2}
    nn.parameters[0] = (nn.parameters[0]) + step_width * momenta

    # \nabla_x L(x_{n+1})
    gradients = nn.gradients()

    # 3. \widehat{p}_{n+1} = p_{n+\tfrac 1 2} - \tfrac {\lambda}{2} \nabla_x L(x_{n+1})
    momenta -= .5*step_width * gradients

    # 4. p_{n+1} = \alpha \widehat{p}_{n+1} + \sqrt{\frac{1-\alpha^2}{\beta}} \cdot \eta_n
    alpha = math.exp(-gamma*step_width)
    momenta = alpha * momenta + \
              math.sqrt(1.-math.pow(alpha,2.)/beta) * np.random.standard_normal(momenta.shape)

    return gradients, momenta

print("here")

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
)

print("We have "+str(nn.num_parameters())+" parameters.")

momenta = np.zeros((nn.num_parameters()))
old_gradients = np.zeros((nn.num_parameters()))

for i in range(100):
    print("Current step #"+str(i))
    old_gradients, momenta = gla2_update_step(
        nn, momenta, old_gradients, step_width=1e-2, beta=1e3, gamma=10)
    print(nn.loss())