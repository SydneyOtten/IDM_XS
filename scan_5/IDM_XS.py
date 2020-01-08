import pandas as pd
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, History, EarlyStopping, CSVLogger, TensorBoard
from keras.layers import Dense, Activation, LeakyReLU, Dropout
from keras.optimizers import Adam
from keras.layers.core import Lambda
from keras.initializers import he_normal
from keras.regularizers import l2
import os
import time


data = pd.read_csv('table_IDM_scanB.csv')
data = data.values

#Inputs

X = data[:,0:5]

#Outputs

y3535, y3636, y3737, y3537, y3637, y3735, y3736, y3536 = data[:,5], data[:,6], data[:,7], data[:,8], data[:,9], data[:,10], data[:,11], data[:,12]

#Finding indices of entries with cross-sections greater than 10**(-7):

xs = [y3535, y3636, y3737, y3537, y3637, y3735, y3736, y3536]
idxs = []
for y in xs:
    idx = []
    for i in range(0,len(y)):
        if y[i] > 10**(-7):
            idx.append(i)
    idxs.append(idx)

#constructing the training data

X3535, X3636, X3737, X3537, X3637, X3735, X3736, X3536 = 0, 0, 0, 0, 0, 0, 0, 0
inputs = [X3535, X3636, X3737, X3537, X3637, X3735, X3736, X3536]

for i in range(0,len(inputs)):
    inputs[i] = X[idxs[i]]
    xs[i] = xs[i][idxs[i]]

###preprocessing: inputs are z-score transformed; target values (xs) are log transformed and then z-score transformed

#####inputs
i=0
for X in inputs:
    X = (X-np.mean(X, axis=0))/np.std(X, axis=0)
    inputs[i] = X
    i=i+1

#####cross sections
std_logs_y = np.empty(8)
mean_logs_y = np.empty(8)
def xs_preproc(z_score = False, shifted_log = True):
    i=0
    if z_score == True and shifted_log == True:
        print('Please use either z-score or shifted log preprocessing. Exiting.')
        exit()
    if z_score == True:
        for y in xs:
            y = np.log(y)
            std_logs_y[i] = np.std(y)
            mean_logs_y[i] = np.mean(y)
            y = (y-np.mean(y))/np.std(y)
            xs[i] = y
            i=i+1
        return std_logs_y, mean_logs_y, xs
    else:
        for y in xs:
            y = np.log(y)
            min_log = np.log(10**(-7))
            y = -min_log + y
            xs[i] = y
            i=i+1
        return xs
#std_logs_y, mean_logs_y, xs = xs_preproc(z_score = True, shifted_log = False)
#print(std_logs_y)
xs = xs_preproc()

###splitting data in training and validation data
train_fraction = 0.7
inputs_train = []
inputs_test = []
for X in inputs:
    X_train, X_test = X[0:int(train_fraction*len(X))], X[int(train_fraction*len(X)):len(X)]
    inputs_train.append(X_train)
    inputs_test.append(X_test)

xs_train = []
xs_test = []
for y in xs:
    y_train, y_test = y[0:int(train_fraction*len(y))], y[int(train_fraction*len(y)):len(y)]
    xs_train.append(y_train)
    xs_test.append(y_test)

#define loss functions using function closure due to dependency on external parameter: 
#standard deviation of log(y) for the different cross sections to obtain minimization 
#of the mape of the original cross-sections
def log_mape(sigma_log_y):
    def loss(y_true, y_pred):
        return 100. * K.mean(K.abs(1-K.exp(sigma_log_y*(y_pred-y_true))), axis=-1)
    return loss

def exp_xsec():
    def loss(y_true, y_pred):
        return 100. * K.mean(K.abs(1. - K.exp(y_pred-y_true)), axis=-1)
    return loss
#define permadropout

def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))

#define NN architecture

def xs_nn(layers=3, neurons=128, dropout_fraction=0.05, L2lambda=0.0):
    model = Sequential()
    model.add(Dense(neurons, input_dim=5, kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(L2lambda)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(PermaDropout(dropout_fraction))
    for i in range(0,layers-1):
        model.add(Dense(neurons, kernel_initializer = he_normal(seed=42), kernel_regularizer=l2(L2lambda)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(PermaDropout(dropout_fraction))
    model.add(Dense(1, kernel_initializer=he_normal(seed=42), activation='linear'))
    return model

#hyperparameter scan
nlayers = [6]
nneurons = [192]
ndropout_fraction = [0.01]
nL2lambda = [1e-5]

#define training procedure
def train(model, X, y, X_test, y_test, std_log_y, name, output_dir = './', exp_xsecs = True, log_mapes = False, iterations=10):
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    history = History()
    checkpointer = ModelCheckpoint(output_dir + '/' + name, monitor='val_loss', verbose=1, save_best_only=True)
    csv_logger = CSVLogger(output_dir + '/' + name + '.log')
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=256, write_graph=True, update_freq='epoch')
    lr = 0.001
    j=0
    start_time = time.time()
    time.clock()
    for i in range(0,iterations):
        csv_logger = CSVLogger(output_dir + '/' + name + '_iteration_' + str(j) + '.log')
        opt = Adam(lr = lr)
        if log_mapes == True and exp_xsec == True:
            print('You can only have one loss function. Exiting.')
            exit()
        if log_mapes == True:
            model.compile(optimizer = opt, loss = log_mape(std_log_y))
            model.fit(X, y, epochs=500, verbose=1, callbacks=[early_stopping, history, checkpointer, csv_logger, tensorboard], validation_data=(X_test, y_test))
            model = load_model(output_dir + '/' + name, compile=False, custom_objects={'log_mape': log_mape(std_log_y)})
        if exp_xsecs == True:
            model.compile(optimizer = opt, loss = exp_xsec())
            model.fit(X, y, epochs=500, verbose=1, callbacks=[early_stopping, history, checkpointer, csv_logger, tensorboard], validation_data=(X_test, y_test))
            model = load_model(output_dir + '/' + name, compile=False, custom_objects={'exp_xsec': exp_xsec})
        lr = lr/2
    end_time = time.time()
    duration = end_time - start_time
    f = open('mapes.txt', 'a')
    f.write('The training took ' + str(duration) + ' seconds.\n')
    f.close()
    if log_mapes == True and exp_xsecs == True:
        print('You can only have one loss function. Exiting.')
        exit()
    if log_mapes == True:
        model.compile(optimizer = opt, loss = log_mape(std_log_y))
    if exp_xsecs == True:
        model.compile(optimizer = opt, loss = exp_xsec())
    return model

#define postprocessing
def postprocess(preds, samples, mean, std, exp_xsecs = True, log_mapes = False):
    if exp_xsecs == True and log_mape == True:
        print('You can only postprocess according to one loss. Exiting.')
        exit()
    if log_mapes == True:
        preds = preds*std+mean
        preds = np.exp(preds)
        print(np.shape(preds))
    if exp_xsecs == True:
        preds = preds + np.log(10**(-7))
        preds = np.exp(preds)
        print(np.shape(preds))
    return preds
        
#define evaluation
def evaluate(model, X_test, y_test, mean_log_y, std_log_y, samples=100, output_dir='./'):
    preds = []
    for i in range(0, samples):
        y_pred = model.predict(X_test)
        y_pred = np.array(y_pred)
        y_pred = np.reshape(y_pred, (-1,))
        preds.append(y_pred)
    preds = np.array(preds)
    print(np.shape(preds))
    y_pred_post, y_true_post = postprocess(preds, samples, mean_log_y, std_log_y, exp_xsecs = True, log_mapes = False), postprocess(y_test, 1, mean_log_y, std_log_y, exp_xsecs = True, log_mapes = False)
    print(y_pred_post[:3,:5])
    print(y_true_post[:5])
    #perc_errors = np.empty([len(y_true_post), samples])
    mean_y_pred = np.empty(len(y_pred))
    std_y_pred = np.empty(len(y_pred))
    for i in range(0,len(y_pred)):
        mean_y_pred[i] = np.mean(y_pred_post[:,i])
        std_y_pred[i] = np.std(y_pred_post[:,i])
    print(mean_y_pred[:5])
    print(std_y_pred[:5])
    total_uncertainty = np.sum(std_y_pred)
    mean_std_ratio = std_y_pred / mean_y_pred
    perc_errors = np.empty(len(y_pred))
    tpe = 0
    for i in range(0,len(y_pred)):
        perc_errors[i] = np.abs((mean_y_pred[i] - y_true_post[i])/y_true_post[i])
        tpe += perc_errors[i]
    print(perc_errors[:100])
    print(np.shape(perc_errors))
    print(tpe)
    tpe = tpe/len(y_pred)
    mape = np.sum(perc_errors)/len(y_pred)
    print(mape)
    print(tpe)
    np.savez_compressed(str(output_dir) + 'evaluation.npz', X_test = X_test, y_test = y_true_post, y_pred = mean_y_pred, std_pred = std_y_pred, total_uncertainty = total_uncertainty, mean_std_ratio = mean_std_ratio, perc_errors = perc_errors, mape = mape)
    return mean_y_pred, y_true_post, std_y_pred, total_uncertainty, mean_std_ratio, perc_errors, mape

#execute training for 3535 on all hyperparameters, pick best hyperparameters, execute training 
mean_y_preds, y_true_posts, std_y_preds, mean_std_ratios, perc_errors, mapes = [], [], [], [], [], []

nrows=int(len(nlayers)*len(nneurons)*len(ndropout_fraction)*len(nL2lambda))
train_reps = 5
log = np.empty([nrows,6,train_reps])
i=0
criterium = np.empty(nrows)
for layers in nlayers:
    for neurons in nneurons:
        for dropout_fraction in ndropout_fraction:
            for L2lambda in nL2lambda:
                for train_rep in range(0,train_reps):
                    dir_name = str(layers) + '_layers_' + str(neurons) + '_neurons_' + str(dropout_fraction) + '_dropout_' + str(L2lambda) + '_L2lambda_3535' + '_REP_' + str(train_rep)
                    os.mkdir(str(layers) + '_layers_' + str(neurons) + '_neurons_' + str(dropout_fraction) + '_dropout_' + str(L2lambda) + '_L2lambda_3535' + '_REP_' + str(train_rep))
                    name =str(layers) + '_layers_' + str(neurons) + '_neurons_' + str(dropout_fraction) + '_dropout_' + str(L2lambda) + '_L2lambda_3535' + '_REP_' + str(train_rep) + '.hdf5'
                    model = xs_nn(layers, neurons, dropout_fraction, L2lambda)
                    model = train(model, inputs_train[0], xs_train[0], inputs_test[0], xs_test[0], std_log_y = std_logs_y[0], name = name, output_dir = dir_name, exp_xsecs = True, log_mapes = False)
                    mean_y_pred, y_true_post, std_y_pred, total_uncertainty, mean_std_ratio, perc_error, mape = evaluate(model, inputs_test[0], xs_test[0], mean_logs_y[0], std_logs_y[0], samples=100, output_dir = './' + str(layers) + '_layers_' + str(neurons) + '_neurons_' + str(dropout_fraction) + '_dropout_' + str(L2lambda) + '_L2lambda_3535_REP_' + str(train_rep) + '/')
                    mean_y_preds.append(mean_y_pred)
                    y_true_posts.append(y_true_post)
                    std_y_preds.append(std_y_pred)
                    mean_std_ratios.append(mean_std_ratio)
                    perc_errors.append(perc_error)
                    mapes.append(mape)
                    f = open('mapes.txt', 'a')
                    f.write("REP " + str(train_rep) + " LAYERS: " + str(layers) + " NEURONS: " + str(neurons) + " DROPOUT FRACTION: " + str(dropout_fraction) + " L2 lambda: " + str(L2lambda) + " MAPE: " + str(mape) + " TOTAL UNCERTAINTY: " + str(total_uncertainty) + "\n")
                    f.close()
                    log[i,0,train_rep] = layers
                    log[i,1,train_rep] = neurons
                    log[i,2,train_rep] = dropout_fraction
                    log[i,3,train_rep] = L2lambda
                    log[i,4,train_rep] = mape
                    log[i,5,train_rep] = total_uncertainty
                    K.clear_session()
                mape_mean = np.mean(log[i,4,:])
                mape_std = np.std(log[i,4,:])
                min_mape = np.min(log[i,4,:])
                max_mape = np.max(log[i,4,:])
                criterium[i] = mape_mean + mape_std
                f = open('mapes.txt', 'a')
                f.write("MEAN MAPE : " + str(mape_mean) + " MAPE STD : " + str(mape_std) + " MIN MAPE : " + str(min_mape) + " MAX MAPE : " + str(max_mape) + "\n")
                f.close()
                i = i + 1

#choosing best hyperparameters
print(log)
#best_idx = np.argmin(log[:,4,:])
best_idx = np.argmin(criterium)
best_ix_row=np.take(log,best_idx)
print(best_idx)
print(type(best_idx))
print(best_ix_row)

#f = open('mapes.txt', 'a')
#f.write("BEST PARAMS:\n" + "LAYERS: " + str(log[int(best_idx),0,0]) + " NEURONS: " + str(log[int(best_idx),1])) + " DROPOUT FRACTION: " + str(log.item((int(best_idx),2))) + " L2 lambda: " + str(log.item((int(best_idx),3))) + " MAPE: " + str(log.item((int(best_idx),4))) + " TOTAL UNCERTAINTY: " + str(log.item((int(best_idx),5))))
        
#f.close()

#training neural networks for the remaining xs on best hyperparameters for 3535
names = ['3636.hdf5', '3737.hdf5', '3537.hdf5', '3637.hdf5', '3735.hdf5', '3736.hdf5', '3536.hdf5'] 
xs_mean_y_preds, xs_y_true_posts, xs_std_y_preds, xs_mean_std_ratios, xs_perc_errors, xs_mapes = [], [], [], [], [], []
for j in range(1,8):
    name = names[j-1]
    model = xs_nn(6, 192, 0.01, 1e-5)
    #model = xs_nn(int(log.item(int(best_idx),0)),int(log.item(int(best_idx),1)), float(log.item(int(best_idx),2)), float(log.item(int(best_idx),3)))
    model = train(model, inputs_train[j], xs_train[j], inputs_test[j], xs_test[j], std_log_y = std_logs_y[j], name = name, exp_xsecs = True, log_mapes = False)
    mean_y_pred, y_true_post, std_y_pred, total_uncertainty, mean_std_ratio, perc_error, mape = evaluate(model, inputs_test[j], xs_test[j], mean_logs_y[j], std_logs_y[j], samples=100, output_dir='./' + name)
    xs_mean_y_preds.append(mean_y_pred)
    xs_y_true_posts.append(y_true_post)
    xs_std_y_preds.append(std_y_pred)
    xs_mean_std_ratios.append(mean_std_ratio)
    xs_perc_errors.append(perc_error)
    xs_mapes.append(mape)

#plots 
#def all_plots():
