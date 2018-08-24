from keras.layers import Layer, InputSpec, merge
from keras import regularizers, initializations, activations, constraints
from keras import backend as K
import theano as th
import numpy as np
import FinalLayerAccess as fla

class GraphHighway(Layer):
    def __init__(self, init='glorot_uniform', transform_bias=-2,
                 n_rel=5, mean=1,
                 activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.transform_bias = transform_bias
        self.activation = activations.get(activation)
        self.n_rel = n_rel
        self.mean = mean

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(GraphHighway, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = self.input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.init((input_dim, input_dim),
                           name='{}_W'.format(self.name))
        print("W")
        print(self.W)
        self.W_carry = self.init((input_dim, input_dim),
                                 name='{}_W_carry'.format(self.name))
        self.V = self.init((self.n_rel, input_dim, input_dim),
                           name='{}_V'.format(self.name))

        if self.bias:
            self.b = K.zeros((input_dim,), name='{}_b'.format(self.name))
            # initialize with a vector of values `transform_bias`
            self.b_carry = K.variable(np.ones((input_dim,)) * self.transform_bias,
                                      name='{}_b_carry'.format(self.name))
            self.V_carry = self.init((self.n_rel, input_dim, input_dim),
                             name='{}_V_carry'.format(self.name))

            self.trainable_weights = [self.W, self.V, self.b, self.W_carry, self.V_carry, self.b_carry]

        else:
            self.V_carry = self.init((self.n_rel, input_dim, input_dim),
                             name='{}_V_carry'.format(self.name))

            self.trainable_weights = [self.W, self.W_carry, self.V, self.V_carry]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        return (None, self.input_dim)

    def call(self, inputs, mask=None):
        x = inputs[0] #feature matrix
        rel = inputs[1] # n_nodes, n_rel, n_neigh
        rel_mask = inputs[2]
        mask_mul = rel_mask[:, 0]
        mask_div = rel_mask[:, 1]

        n_nodes, n_rel, n_neigh = rel.shape # number of nodes, number of relation types, number of neighbors for each type of relations
        dim = x.shape[-1]

        # compute the context for each type of relations in each node:
        # context = sum(all neighbors with the same relation to the node)
        context = x[rel.flatten()].reshape([n_nodes, n_rel, n_neigh, dim])
        context = context * mask_mul[:, :, :, None]
        context = K.sum(context, axis=-2) / K.sum(mask_div, axis=-1)[:, :, None]
        # -> now, context: n_nodes, n_rel, dim

        # dot(V_carry, context)
        carry_gate = K.dot(x, self.W_carry)
        carry_context = context[:, :, :, None] * self.V_carry[None, :, :, :]
        if self.mean == 0:
            carry_context = K.max(carry_context, axis=(1, 2))
        else:
            carry_context = K.sum(carry_context, axis=(1, 2)) / self.mean

        carry_gate += carry_context

        if self.bias:
             carry_gate += self.b_carry
        carry_gate = activations.sigmoid(carry_gate)

        # dot(V, context)
        context = context[:, :, :, None] * self.V[None, :, :, :]
        if self.mean == 0:
            context = K.max(context, axis=(1, 2))
        else:
            context = K.sum(context, axis=(1, 2)) / self.mean

        h = K.dot(x, self.W) + context
        if self.bias:
            h += self.b

        h = self.activation(h)
        h = carry_gate * h + (1 - carry_gate) * x
        return h

    def get_config(self):
        config = {'init': self.init.__name__,
                  'transform_bias': self.transform_bias,
                  'activation': self.activation.__name__,
                  'n_rel': self.n_rel,
                  'mean': self.mean,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(GraphHighway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


##MD+DEV+YANG -- modify constructor with additional parameters for temp P(Y)
class GraphDense(Layer):
    def __init__(self, init='glorot_uniform',
                 n_rel=5, mean=1,
                 input_dim=None, output_dim=None, class_dim=None,
                 activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.n_rel = n_rel
        self.mean = mean
        #self.prefEffect = prefEffect ### MD+DEV+YANG ---- additional Variables
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        
        
        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.class_dim = class_dim #MD
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(GraphDense, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = self.input_dim
        output_dim = self.output_dim
        class_dim = self.class_dim #MD
        # print("inputdim:   ",input_dim)
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.init((input_dim, output_dim),
                           name='{}_W'.format(self.name))
        self.V = self.init((self.n_rel, input_dim, output_dim),
                           name='{}_V'.format(self.name))
        
        self.advice_rate = self.init((self.n_rel, input_dim, output_dim),
                                 name='{}_advice_rate'.format(self.name))

        if self.bias:
            self.b = K.zeros((output_dim,), name='{}_b'.format(self.name))
            # initialize with a vector of values `transform_bias`
            self.trainable_weights = [self.W, self.V, self.advice_rate, self.b]
        else:
            self.trainable_weights = [self.W, self.V, self.advice_rate]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        return (None, self.output_dim)

    def call(self, inputs, mask=None):
        x = inputs[0] #feature matrix
        # print("x in the call ",x.shape)
        rel = inputs[1] # n_nodes, n_rel, n_neigh
        rel_mask = inputs[2]
        Iadv = inputs[3]
        W_adv_mask = inputs[4]
        c_adv_mask = inputs[5]
        yprobs = inputs[6]
        y_adv_mask = inputs[7]
        #prefEffect = inputs[6], ##MD+DEV+YANG
        mask_mul = rel_mask[:, 0]
        mask_div = rel_mask[:, 1]

        n_nodes, n_rel, n_neigh = rel.shape # number of nodes, number of relation types, number of neighbors for each type of relations
        dim = x.shape[-1]

        # compute the context for each type of relations in each node:
        # context = sum(all neighbors with the same relation to the node)

        # print("rel   ",type(rel))
        # print("n_rel",type(n_rel))
        context = x[rel.flatten()].reshape([n_nodes, n_rel, n_neigh, dim])
        
        advice_gate = K.sum((y_adv_mask - yprobs)*y_adv_mask, axis=1) #MD
        c_adv_mask = K.exp(c_adv_mask[:, :, :] * advice_gate[:, None, None])#MD
        print "ADVICE MASK", K.eval(c_adv_mask).shape
        context = context * c_adv_mask[:, :, :, None] #MD
        
        context = context * mask_mul[:, :, :, None]
        #context = context * c_adv_mask[:, :, :, None] # MD & Yang
        #print("context shape   ", x.shape)
        #print("final Layer **** ", fla.fprobs)
        context = K.sum(context, axis=-2) / K.sum(mask_div, axis=-1)[:, :, None]
        # Calculate indicator ---- MD
        #if not (True in th.tensor.isnan(yprobs)):
        
        #context = context * np.exp(np.dot(np.subtract(Iadv[:],self.prefEffect[:]),c_adv_mask[:, :, :, None]))##doing - MD+DEV+YANG
        #context = context * K.exp(advice_gate[:, None, None] * c_adv_mask[:, :, :])
        #print("context shape   ", context.shape)
        
        
        
        
        # -> now, context: n_nodes, n_rel, dim
        
        
        # dot(V, context)
        context = context[:, :, :, None] * self.V[None, :, :, :]
        context = K.sum(context, axis=(1, 2)) / self.mean
        
        
        #
        print("context   ", type(context))
        
        # print("W   ", type(self.W))
        # # print("this W")
        # # print(type(self.W))
        # # print(K.eval(self.W))
        #print("dimension of W:  ", K.eval(self.W).shape)
        # print("content of W:  ", K.eval(self.W))
        # print("this V")
        # print(type(self.V))
        #print("dimension of V:  ", K.eval(self.V).shape)
        #x = x*advice_gate[:,None]
        
        h = K.dot(x, self.W) + context
        if self.bias:
            h += self.b
        h = self.activation(h)
        ############MD+DEV+YANG --- major chnages here with Temp P(Y)
        #if this layer is the top layer, just take all the training nodes to compute loss and probability
        return h

    def get_config(self):
        config = {'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'n_rel': self.n_rel,
                  'mean': self.mean,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim,
                  'output_dim': self.output_dim}
        base_config = super(GraphDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
