import tensorflow as tf

class SGLDSampler(tf.train.GradientDescentOptimizer):
    ''' implements a Stochastic Gradient Langevin Dynamics Sampler
    in the form of a TensorFlow Optimizer
    '''
    def __init__(self, rate):
        super(SGLDSampler, self).__init__(rate)
    
    def get_name(self):
        return str("SGLDSampler")
    
    def minimize(self,
                 loss,
                 global_step=None,
                 var_list=None,
                 gate_gradients=tf.train.Optimizer.GATE_OP,
                 aggregation_method=None,
                 collocate_gradients_with_ops=False,
                 name=None,
                 grad_loss=None):
        return super(SGLDSampler, self).minimize(
            loss, global_step, var_list, gate_gradients,
            aggregation_method, collocate_gradients_with_ops,
            name, grad_loss)
        
    
