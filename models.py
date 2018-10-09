from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.regularizers import l2
from objectives import cca_loss
from objectives_mcca import mcca_loss

def create_model(layer_sizes_list, input_size_list, act_='linear', 
                            learning_rate=1e-3, n_modalities=3, gamma=0.2, reg_par=1e-5):
    """
    Input:
    ..
    Output:
    ..

    builds the whole model form a list of list of layer sizes!
    !!## note this is not the Sequential style model!
    """    
    input_layers = [Input((size_i, )) for size_i in input_size_list]

    fc_output_layer_list = []

    for l_i, layer_sizes_ in enumerate(layer_sizes_list):
        # pre-create the dense(fc) layers you need
        ## USING ONLY LINEAR ACTIVATIONS FOR NOW!!
        fc_layers_ = [Dense(i,activation=act_, kernel_regularizer=l2(reg_par)) for i in layer_sizes_[:-1]]
        # no matter the layer activation, the last layer needs a sigmoid activation!
        fc_layers_.append(Dense(layer_sizes_[-1], activation=act_, kernel_regularizer=l2(reg_par)))

        D = fc_layers_[0](input_layers[l_i])
        # do this in a non-sequential style Keras model
        for d_i, d in enumerate(fc_layers_[1:]): D = d(D) 
        fc_output_layer_list.append(D)

    output = concatenate(fc_output_layer_list)
    model = Model(input_layers, [output])

    model_optimizer = RMSprop(lr=learning_rate)
    model.compile(loss=mcca_loss(n_modalities, 0.2), optimizer=model_optimizer)

    return model


##### older code - ignore


def build_mlp_net(layer_sizes, input_size, reg_par):
    model = Sequential()
    for l_id, ls in enumerate(layer_sizes):
        if l_id == 0:
            input_dim = input_size
        else:
            input_dim = []
        if l_id == len(layer_sizes)-1:
            activation = 'linear'
        else:
            activation = 'sigmoid'

        model.add(Dense(ls, input_dim=input_dim,
                                activation=activation,
                                kernel_regularizer=l2(reg_par)))
    return model
def _create_model(layer_sizes1, layer_sizes2, input_size1, input_size2,
                    learning_rate, reg_par, outdim_size, use_all_singular_values):
    """
    builds the whole model
    the structure of each sub-network is defined in build_mlp_net,
    and it can easily get substituted with a more efficient and powerful network like CNN
    """
    inp_1 = Input((input_size1,))
    inp_2 = Input((input_size2,))
    dense_layers1 = [Dense(i) for i in layer_sizes1]
    D1 = dense_layers1[0](inp_1)
    for d_i,d in enumerate(dense_layers1[1:]):
		D1 = d(D1)

    dense_layers2 = [Dense(i) for i in layer_sizes2]
    D2 = dense_layers2[0](inp_2)
    for d_i,d in enumerate(dense_layers2[1:]):
		D2 = d(D2)

    output = concatenate([D1, D2])
    model = Model([inp_1, inp_2], [output])

    model_optimizer = RMSprop(lr=learning_rate)
    model.compile(loss=cca_loss(outdim_size, use_all_singular_values), optimizer=model_optimizer)

    return model


