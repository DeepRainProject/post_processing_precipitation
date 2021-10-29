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

data_path = '/p/home/jusers/rojascampos1/juwels/MyProjects/PROJECT_deepacf/deeprain/rojascampos1/post_processing_for_precipitation/data/'
# conditions = ['specific', 'general', 'unobs']
conditions = ['specific']
stations = ['braunl', 'muOsna', 'redlen', 'wernig']
n_runs = 20

archs = {'muOsna': [16,16,16,16,16,16],
         'wernig': [64, 64, 64, 64],
         'braunl': [16,16,16,16,16,16],
         'redlen': [64, 64, 64, 64, 64, 64]}


### Calculates all possible conditions
configs = []

for condition in conditions:
    
    if condition == 'specific':
        for station in stations:
            configs.append([condition, [station], station])
                
    elif condition == 'general':
        for station in stations:
            configs.append([condition, stations, station])
                
    elif condition == 'unobs':
        for station in stations:
            configs.append([condition, [s for s in stations if s != station], station])
        
 

## Master process
if my_rank == p-1:
    
   # Collect results in a dictionary
    results = {}
    for c in conditions:

        d = {}
        for s in stations:
            d[s] = {}
        results[c] = d

    
    for proc in range(len(conditions) * len(stations)):

        cond        = comm.recv(source=proc, tag=000)
        tst_sets    = comm.recv(source=proc, tag=123)
        runs_acc    = comm.recv(source=proc, tag=456)
        runs_fb     = comm.recv(source=proc, tag=789)
        runs_lor    = comm.recv(source=proc, tag=999)
        runs_ets    = comm.recv(source=proc, tag=888)
        
        results[cond][tst_sets] = {'accuracy':runs_acc, 'frequency_bias':runs_fb, 'log_odds_ratio':runs_lor, 'ets':runs_ets}


    with open('results_classification.json', 'w') as fp:
        json.dump(results, fp)

            
else:

    condition = configs[my_rank][0]
    trn_sets  = configs[my_rank][1]
    tst_sets  = configs[my_rank][2]

    ## Load files
    trn_x = []
    trn_y = []
    for station in trn_sets:
        trn_x.append(np.load(data_path + station + '/trn_x.npy'))
        trn_y.append(np.load(data_path + station + '/trn_y.npy'))

    trn_x = np.concatenate(trn_x)
    trn_y = np.concatenate(trn_y)

    tst_x = np.load(data_path + tst_sets + '/tst_x.npy')
    tst_y = np.load(data_path + tst_sets + '/tst_y.npy')
    tst_t = np.load(data_path + tst_sets + '/tst_t.npy')

    print('ID =', my_rank, condition, tst_sets,  'trn:', trn_x.shape, '--->', trn_y.shape, 'tst:', tst_x.shape, '--->', tst_y.shape)

    ## Feature selection
    id_predictors   = np.loadtxt('/p/project/deepacf/deeprain/rojascampos1/precipitation_prediction/data/00_gordons_regression'+'/'+tst_sets+'/'+tst_sets+ '_ID_Predictors.txt', dtype=np.int32,delimiter=',')
    id_predictors   = id_predictors - 1 ## Matlab index start in 1
    trn_x = trn_x[:, id_predictors]
    tst_x = tst_x[:, id_predictors]


    ## Classification model
    trn_y_b = np.array(trn_y >= 0.1, dtype=int)
    tst_y_b = np.array(tst_y >= 0.1, dtype=int)
    
    batch_size = 128
    epochs = 64

    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = [tf.keras.metrics.BinaryAccuracy()]
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)    
    
    
    results_runs_acc = []
    results_runs_fb = []
    results_runs_lor = []
    results_runs_ets = []

    for r in range(n_runs):
        
        ## Cross validation
        cvalid_preds = []
        cvalid_models = []
        cvalid_ets = []
        
        kf = KFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(trn_x, trn_y):
            
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=(trn_x.shape[1])))

            a = archs[station]

            for layer in range(len(a)):
                model.add(tf.keras.layers.Dense(a[layer], activation='relu'))


            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

            model.compile(optimizer, loss, metrics)

            hist = model.fit(trn_x[train_index], trn_y_b[train_index], batch_size, epochs, validation_data=(trn_x[test_index], trn_y_b[test_index]), callbacks=[early_stop], shuffle=True)

            pred = model(trn_x[test_index])
            pred = np.squeeze(pred)
            pred = np.array(pred >= 0.5, dtype=int)
            
            print('trn_y_b[test_index]', trn_y_b[test_index])
            print('pred', pred)
    
            ### pred and trn_y_b[test_index]
            cm_in = confusion_matrix(trn_y_b[test_index], pred)
            hits_in = cm_in[0][0]
            false_alarms_in = cm_in[0][1]
            misses_in = cm_in[1][0]
            correct_negatives_in = cm_in[1][1]
            
            # accuracy = (hits + correct_negatives) / np.sum(cm_in)
            # frequency_bias = (hits + false_alarms) / (hits + misses)
            # log_odd_ratio = np.log((hits * correct_negatives) / (misses * false_alarms))
            hits_random_in = ((hits_in + misses_in)*(hits_in + false_alarms_in)) / np.sum(cm_in)
            ets_in = (hits_in - hits_random_in)/(hits_in + misses_in + false_alarms_in - hits_random_in)
            
            cvalid_ets.append(ets_in)
            cvalid_preds.append(pred)
            cvalid_models.append(model)

        
        
        index_best = np.argmax(cvalid_ets)
        mod = cvalid_models[index_best]
        predictions = mod(tst_x)
        predictions_b = np.array(predictions >= 0.5, dtype=int) ## This could change
        
        cm = confusion_matrix(tst_y_b, predictions_b)
        hits = cm[0][0]
        false_alarms = cm[0][1]
        misses = cm[1][0]
        correct_negatives = cm[1][1]

        accuracy = (hits + correct_negatives) / np.sum(cm)
        frequency_bias = (hits + false_alarms) / (hits + misses)
        log_odd_ratio = np.log((hits * correct_negatives) / (misses * false_alarms))
        hits_random = ((hits + misses)*(hits + false_alarms)) / np.sum(cm)
        ets = (hits - hits_random)/(hits+misses+false_alarms-hits_random)

        results_runs_acc.append(accuracy)
        results_runs_fb.append(frequency_bias)
        results_runs_lor.append(log_odd_ratio)
        results_runs_ets.append(ets)
        
    
    comm.send(condition, dest=p-1, tag=000)
    comm.send(tst_sets,  dest=p-1, tag=123)
    comm.send(results_runs_acc,  dest=p-1, tag=456)
    comm.send(results_runs_fb,  dest=p-1, tag=789)
    comm.send(results_runs_lor,  dest=p-1, tag=999)
    comm.send(results_runs_ets, dest=p-1, tag=888)


    
