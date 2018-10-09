try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
import glob
from keras.callbacks import ModelCheckpoint, EarlyStopping
#from utils import load_data, svm_classify
from sklearn import datasets, svm, preprocessing 
from sklearn.pipeline import make_pipeline, Pipeline
#from linear_cca import linear_cca
from models import create_model
from sklearn.model_selection import *
from load_data import load_noisy_mnist_data

def run_svc_pipeline_doubleCV(X, y, dev_split=5, C_=0.015, n_splits_=10, param_search=True, n_jobs_=18): 
    # use different splits with different random states for CV-param search
    svc = svm.SVC(kernel='linear', C = C_) 
    #svc = svm.LinearSVC(C = C_) 

    pipeline_estimators = [('scale', preprocessing.StandardScaler()), ('svm',
                                                                       svc) ]
    svc_pipeline = Pipeline(pipeline_estimators)

    if param_search:
        C_search = sorted( list(np.logspace(-5,0,10)) + [0.1,5,10,20,50,100] )
        param_grid = dict( scale=[None], svm__C=C_search )

        sk_folds = StratifiedKFold(n_splits=dev_split, shuffle=False,
                                   random_state=1964)
        grid_search = GridSearchCV(svc_pipeline, param_grid=param_grid,
                                   n_jobs=n_jobs_, cv=sk_folds.split(X,y),
                                   verbose=False)
        grid_search.fit(X, y)
        # find the best C value
        which_C = np.argmax(grid_search.cv_results_['mean_test_score'])
        best_C = C_search[which_C]
    else:
        best_C = C_

    svc_pipeline.named_steps['svm'].C = best_C
    #print('estimated the best C for svm to be', best_C)
    sk_folds = StratifiedKFold(n_splits=n_splits_, shuffle=False, random_state=320)
    all_scores = []
    all_y_test = []
    all_pred = []
    for train_index, test_index in sk_folds.split(X, y):
#        print 'run -',
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        svc_pipeline.fit(X_train, y_train)
        y_pred = svc_pipeline.predict(X_test)
        score = svc_pipeline.score(X_test, y_test)
 #       print score	
        all_y_test.append(y_test)
        all_pred.append(y_pred)
        all_scores.append(score)
    return all_y_test, all_pred, all_scores

def train_model(model, data_list, epoch_num, batch_size, feature_dim):
    """
    trains the model
    # Arguments
        .... inputs?
        epoch_num: number of epochs to train the model
        batch_size: the size of batches
    # Returns
        the trained model
    """

    # Unpacking the data
    # the data_list is arranged thus:
    # [[(train_x, train_y), (val_x, val_y), (test_x, test_y) ]_(1), {}_(2),...]
    train_x_list = [i[0][0] for i in data_list]
    val_x_list = [i[1][0] for i in data_list]
    test_x_list = [i[2][0] for i in data_list]
    
    # for later
    test_y_list = [i[2][1] for i in data_list]

    # it is done to return the best model based on the validation loss
    checkpointer = ModelCheckpoint(filepath="weights_%d_dim.{epoch:02d}-{val_loss:.4f}.hdf5" % (feature_dim), 
                                        verbose=1, save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(min_delta = 1e-4, patience = 5)

    # used dummy Y because labels are not used in the loss function
    model.fit(train_x_list, np.zeros(len(train_x_list[0])),
              batch_size=batch_size, epochs=epoch_num, shuffle=True,
              validation_data=(val_x_list, np.zeros(len(val_x_list[0]))),
              callbacks=[checkpointer])

    #model_names_ = glob.glob('weights*5')
    #model.load_weights(model_names_[-1])

    results = model.evaluate(test_x_list, np.zeros(len(test_x_list[0])), batch_size=batch_size, verbose=1)

    print('loss on test data: ', results)

    results = model.evaluate(val_x_list, np.zeros(len(val_x_list[0])), batch_size=batch_size, verbose=1)
    print('loss on validation data: ', results)
    return model


def test_model(model, data_list, apply_mcca=False):
    """produce the new features by using the trained model
        outdim_size: dimension of new features
        apply_linear_cca: if to apply linear CCA on the new features
    # Returns
        new features packed like
    """

    # the data_list is arranged thus:
    # [[(train_x, train_y), (val_x, val_y), (test_x, test_y) ]_(1), {}_(2),...]
    train_x_list = [i[0][0] for i in data_list]
    val_x_list = [i[1][0] for i in data_list]
    test_x_list = [i[2][0] for i in data_list]
    
    # for later
    train_y = [i[0][1] for i in data_list][0] # since all three modalities have same labels
    val_y = [i[1][1] for i in data_list][0]
    test_y = [i[2][1] for i in data_list][0]
    # producing the new features
    train_embeddings = model.predict(train_x_list)
    val_embeddings = model.predict(val_x_list)
    test_embeddings = model.predict(test_x_list)

    return [(train_embeddings, train_y), (val_embeddings,val_y), (test_embeddings, test_y)]


if __name__ == '__main__':
############
# Parameters Section

# the path to save the final learned features
    save_to = './mcca_noisy_mnist_features.gz'

# number of modalities/datasets = n_mod
    n_mod = 3

# size of the input for view 1 and view 2
    input_shapes = [784, 784, 784]

# the size of the new space learned by the model (number of the new features)
    outdim_size = 50 # has to be same for all modalities - (TODO) change later

# layer size list to create a simple FCN
    layer_size_list = [[1024, outdim_size]] * n_mod

# the parameters for training the network
    learning_rate = 1e-3
    epoch_num = 100
    batch_size = 400

# the regularization parameter of the network
# seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5

# specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
# if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False

# if a linear CCA should get applied on the learned features extracted from the networks
# it does not affect the performance on noisy MNIST significantly
    apply_linear_cca = False

# end of parameters section
############

# the load_data function loads noisy mnist data as a list of train,val,test triples
    data_list = load_noisy_mnist_data()

# Building, training, and producing the new features by DCCA
    model = create_model(layer_size_list, input_shapes, 
                            act_='linear', learning_rate=learning_rate, n_modalities=n_mod)
    model.summary()
    model = train_model(model, data_list, epoch_num, batch_size, outdim_size)
    model.save('saved_model_sigmoid_at_last_layer.h5')
    
    data_embeddings = test_model(model, data_list)
# just test on the test set for now to assess the viability!
    pred_on_test = run_svc_pipeline_doubleCV(data_embeddings[2][0], data_embeddings[2][-1])
    print(pred_on_test[-1])
    #np.savez('saved_embeddings_sigmoid_at_last_layer_relu_model', train=data_embeddings[0], val=data_embeddings[1], test=data_embeddings[2])
