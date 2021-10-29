import json
import numpy as np
import tensorflow as tf

from mpi4py import MPI
from validate import validate
from sklearn.metrics import *
from sklearn.model_selection import KFold


comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

data_path = '/p/home/jusers/rojascampos1/juwels/MyProjects/PROJECT_deepacf/deeprain/rojascampos1/precipitation_prediction/data/'
stations = ['muOsna', 'wernig', 'braunl', 'redlen']
n_runs = 20


## Master process
if my_rank == p-1:
    
    skill_muOsna = []
    skill_wernig = []
    skill_braunl = []
    skill_redlen = []
    
    
    for i in range(n_runs):
        skill_muOsna.append(comm.recv(source=i, tag=0))
        skill_wernig.append(comm.recv(source=i, tag=1))
        skill_braunl.append(comm.recv(source=i, tag=2))
        skill_redlen.append(comm.recv(source=i, tag=3))
        
    print('muOsna mdn skill', np.median(skill_muOsna))
    print('wernig mdn skill', np.median(skill_wernig))
    print('braunl mdn skill', np.median(skill_braunl))
    print('redlen mdn skill', np.median(skill_redlen))

    # Collect results in a dictionary
    results = {
        'muOsna' : {'skill' : skill_muOsna},
        'wernig' : {'skill' : skill_wernig},
        'braunl' : {'skill' : skill_braunl},
        'redlen' : {'skill' : skill_redlen}
    }

    with open('results_regression_general.json', 'w') as fp:
        json.dump(results, fp)


else: 

    ## Load files
    trn_x = []
    trn_y = []
    tst_x = []
    tst_y = []
    tst_t = []
    
    for st in stations:
        
        trn_x.append(np.loadtxt(data_path + st + '_0_4_5x5_allFeatures_created_2020-10-23/train_x.csv', delimiter=','))
        trn_y.append(np.loadtxt(data_path + st + '_0_4_5x5_allFeatures_created_2020-10-23/train_y.csv', delimiter=','))
        tst_x.append(np.loadtxt(data_path + st + '_0_4_5x5_allFeatures_created_2020-10-23/test_x.csv', delimiter=','))
        tst_y.append(np.loadtxt(data_path + st + '_0_4_5x5_allFeatures_created_2020-10-23/test_y.csv', delimiter=','))
        tst_t.append(np.load(data_path + st + '_0_4_5x5_allFeatures_created_2020-10-23/timepoints_test.npy'))

    trn_x = np.concatenate(trn_x)
    trn_y = np.concatenate(trn_y)

    ## Regression model    
    trn_y = np.exp(trn_y) ### revert log transform 
  
    batch_size = 32
    epochs = 128

    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanSquaredLogarithmicError()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)    
    
    
    ### Cross - validation
    cvalid_mses = []
    cvalid_models = []

    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(trn_x, trn_y):
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(trn_x.shape[1])))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        
        model.compile(optimizer, loss)
        
        hist = model.fit(trn_x[train_index], trn_y[train_index], batch_size, epochs, validation_data=(trn_x[test_index], trn_y[test_index]), callbacks=[early_stop], shuffle=True)
        
        pred = model(trn_x[test_index])
        pred = np.squeeze(pred)

        mse_cv = mean_squared_error(pred, trn_y[test_index])
        
        cvalid_models.append(model)
        cvalid_mses.append(mse_cv)
    
  

    index_best = np.argmin(cvalid_mses)
    best_model = cvalid_models[index_best]
    
    best_model.summary()
    
    ## Test model for each station
    for idx, sta in enumerate(stations):
        
        print(sta)
        pred = np.squeeze(best_model(tst_x[idx]))
        true_y = np.exp(tst_y[idx])
        
        leps  = validate(pred, tst_t[idx], [0], [4], False, sta)['LEPS']
        leps_cosmo = validate([], tst_t[idx], [0], [4], True, sta)['LEPS'][9]
        skill = 1 - (leps / leps_cosmo)

        comm.send(skill, dest=p-1, tag=idx)
