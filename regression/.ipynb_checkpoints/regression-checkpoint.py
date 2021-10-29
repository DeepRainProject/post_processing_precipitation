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

# data_path = '/p/home/jusers/rojascampos1/juwels/MyProjects/PROJECT_deepacf/deeprain/rojascampos1/post_processing_for_precipitation/data/'
data_path = '/p/home/jusers/rojascampos1/juwels/MyProjects/PROJECT_deepacf/deeprain/rojascampos1/precipitation_prediction/data/'
# conditions = ['specific', 'general', 'unobs']
conditions = ['specific']

archs = {'muOsna' : [1,1],
         'wernig' : [2,2],
         'braunl' : [3,3],
         'redlen' : [4,4]} 

stations = ['muOsna', 'wernig', 'braunl', 'redlen']

n_runs = 5
runs = [str(r) for r in np.arange(n_runs)]

### Calculates all possible conditions
configs = []

for condition in conditions:

    if condition == 'specific':
        for station in stations:
            for run_id in runs:
                configs.append([condition, [station], station, run_id])

    elif condition == 'general':
        for station in stations:
            for run_id in runs:
                configs.append([condition, stations, station, run_id])

    elif condition == 'unobs':
        for station in stations:
            for run_id in runs:
                configs.append([condition, [s for s in stations if s != station], station, run_id])



## Master process
if my_rank == p-1:
    
    # Collect results in a dictionary
    results = {}
    for c in conditions:

        d = {}
        for s in stations:
            d[s] = dict.fromkeys(runs)
        results[c] = d

    
    for proc in range(len(conditions) * len(stations) * len(runs)):

        cond        = comm.recv(source=proc, tag=000)
        tst_sets    = comm.recv(source=proc, tag=321)
        run_id      = comm.recv(source=proc, tag=654)
        mse         = comm.recv(source=proc, tag=543)
        skill       = comm.recv(source=proc, tag=987)
        
        print('Received', cond, tst_sets, run_id, mse, skill)
        
        results[cond][tst_sets][run_id] = {'mse': mse, 'skill': skill}

    
    with open('results_regression.json', 'w') as fp:
        json.dump(results, fp)
            


else: 

    condition = configs[my_rank][0]
    trn_sets  = configs[my_rank][1]
    tst_sets  = configs[my_rank][2]
    run_id = configs[my_rank][3]

    ## Load files
    trn_x = []
    trn_y = []
    trn_t = []
    for station in trn_sets:
        # trn_x.append(np.load(data_path + station + '/trn_x.npy'))
        # trn_y.append(np.load(data_path + station + '/trn_y.npy'))
        trn_x.append(np.loadtxt(data_path + station + '_0_4_5x5_allFeatures_created_2020-10-23/train_x.csv', delimiter=','))
        trn_y.append(np.loadtxt(data_path + station + '_0_4_5x5_allFeatures_created_2020-10-23/train_y.csv', delimiter=','))
        trn_t.append(np.load(data_path + tst_sets + '_0_4_5x5_allFeatures_created_2020-10-23/timepoints_train.npy'))

    trn_x = np.concatenate(trn_x)
    trn_y = np.concatenate(trn_y)
    trn_t = np.concatenate(trn_t)

    # tst_x = np.load(data_path + tst_sets + '/tst_x.npy')
    # tst_y = np.load(data_path + tst_sets + '/tst_y.npy')
    # tst_t = np.load(data_path + tst_sets + '/tst_t.npy')
    
    tst_x = np.loadtxt(data_path + tst_sets + '_0_4_5x5_allFeatures_created_2020-10-23/test_x.csv', delimiter=',')
    tst_y = np.loadtxt(data_path + tst_sets + '_0_4_5x5_allFeatures_created_2020-10-23/test_y.csv', delimiter=',')
    tst_t = np.load(data_path + tst_sets + '_0_4_5x5_allFeatures_created_2020-10-23/timepoints_test.npy')
    print('ID =', my_rank, condition, tst_sets,  'trn:', trn_x.shape, '--->', trn_y.shape, 'tst:', tst_x.shape, '--->', tst_y.shape)

    ## Gordon's used features
    id_predictors   = np.loadtxt('/p/project/deepacf/deeprain/rojascampos1/precipitation_prediction/data/00_gordons_regression'+'/'+tst_sets+'/'+tst_sets+ '_ID_Predictors.txt', dtype=np.int32,delimiter=',')
    id_predictors   = id_predictors - 1 ## Matlab index start in 1
    trn_x = trn_x[:, id_predictors]
    tst_x = tst_x[:, id_predictors]
    
    print('ID =', my_rank, condition, tst_sets,  'trn:', trn_x.shape, '--->', trn_y.shape, 'tst:', tst_x.shape, '--->', tst_y.shape)

    ## Regression model

    # trn_mask_rain = trn_y >= 0.1
    # tst_mask_rain = tst_y >= 0.1

    # trn_x = trn_x[trn_mask_rain]
    # trn_y = trn_y[trn_mask_rain]
    # tst_x = tst_x[tst_mask_rain]
    # tst_y = tst_y[tst_mask_rain]
    # tst_t = tst_t[tst_mask_rain]
    
    trn_y = np.exp(trn_y) ### revert log transform 
    tst_y = np.exp(tst_y)
    
    
    batch_size = 128
    epochs = 100

    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanSquaredLogarithmicError()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)    
    

    ### Cross - validation

    cvalid_mses = []
    cvalid_leps   = []
    cvalid_models = []  

    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(trn_x, trn_y):
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(trn_x.shape[1])))
        
        a = archs[station]
        
        for layer in len(a):
            model.add(tf.keras.layers.Dense(a[layer], activation='relu'))
        model.add()
       
        # model.add(tf.keras.layers.Dense(8, activation='relu'))
        # model.add(tf.keras.layers.Dense(8, activation='relu'))
        # model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        
        model.compile(optimizer, loss)
    
        hist = model.fit(trn_x[train_index], trn_y[train_index], batch_size, epochs, validation_data=(trn_x[test_index], trn_y[test_index]), callbacks=[early_stop], shuffle=True)
        
        # mse_cv = model.evaluate(tst_x)
        pred = model(trn_x[test_index])
        pred = np.squeeze(pred)
        
        print('trn_x[test_index].shape =', trn_x[test_index].shape, 'trn_y[test_index].shape =', trn_y[test_index].shape, 'trn_t[test_index].shape =', trn_t[test_index].shape, 'pred.shape =', pred.shape)
        
        leps = validate(pred, trn_t[test_index], [0], [3,4,5], False, station)['LEPS']
        mse_cv = mean_squared_error(pred, trn_y[test_index])
        cvalid_models.append(model)
        cvalid_leps.append(leps)
        cvalid_mses.append(mse_cv)
    

    index_best = np.argmin(cvalid_leps)
    best_model = cvalid_models[index_best]
    
    best_preds = best_model(tst_x)
    best_preds = np.squeeze(best_preds)
    best_leps  = validate(best_preds, tst_t, [0], [4], False, station)['LEPS']
    leps_cosmo = validate([], tst_t, [0], [4], True, station)['LEPS'][9]
    
    skill = 1 - (best_leps / leps_cosmo)
  
    best_mse = mean_squared_error(best_preds, tst_y)
    print('ID =', my_rank, condition, tst_sets, 'skill =', skill, 'mse =', best_mse)
        
    comm.send(condition, dest=p-1, tag=000)
    comm.send(tst_sets,  dest=p-1, tag=321)
    comm.send(run_id,    dest=p-1, tag=654) 
    comm.send(best_mse,       dest=p-1, tag=543)
    comm.send(skill,     dest=p-1, tag=987)
        
