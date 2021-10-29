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
        
    print('muOsna mdn ets', np.median(skill_muOsna))
    print('wernig mdn ets', np.median(skill_wernig))
    print('braunl mdn ets', np.median(skill_braunl))
    print('redlen mdn ets', np.median(skill_redlen))

    # Collect results in a dictionary
    results = {
        'muOsna' : {'ets' : skill_muOsna},
        'wernig' : {'ets' : skill_wernig},
        'braunl' : {'ets' : skill_braunl},
        'redlen' : {'ets' : skill_redlen}
    }

    with open('results_classification_general.json', 'w') as fp:
        json.dump(results, fp)


else: 

    ## Load files
    trn_x = []
    trn_y = []
    tst_x = []
    tst_y = []
    tst_t = []
    
    for st in stations:
        
        trn_x.append(np.load(data_path + st + '/trn_x.npy'))
        trn_y.append(np.array(np.load(data_path + st + '/trn_y.npy') >= 0.1, dtype=int))
        tst_x.append(np.load(data_path + st + '/tst_x.npy'))
        tst_y.append(np.array(np.load(data_path + st + '/tst_y.npy') >= 0.1, dtype=int))
        tst_t.append(np.load(data_path + st + '/tst_t.npy'))

    trn_x = np.concatenate(trn_x)
    trn_y = np.concatenate(trn_y)
  
    batch_size = 256
    epochs = 64

    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = [tf.keras.metrics.BinaryAccuracy()]
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)    
    
    
    ### Cross - validation
    cvalid_ets = []
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
        model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
        model.compile(optimizer, loss)
        
        ## Train model
        hist = model.fit(trn_x[train_index], trn_y[train_index], batch_size, epochs, validation_data=(trn_x[test_index], trn_y[test_index]), callbacks=[early_stop], shuffle=True)
        
        pred = model(trn_x[test_index])
        pred = np.squeeze(pred)
        pred = np.array(pred >= 0.5, dtype=int)
       
        ## Evaluate k model
        cm_in = confusion_matrix(trn_y[test_index], pred)
        hits_in = cm_in[0][0]
        false_alarms_in = cm_in[0][1]
        misses_in = cm_in[1][0]
        correct_negatives_in = cm_in[1][1]

        hits_random_in = ((hits_in + misses_in) *(hits_in + false_alarms_in)) / np.sum(cm_in)
        ets_in = (hits_in - hits_random_in)/(hits_in + misses_in + false_alarms_in - hits_random_in)

        cvalid_ets.append(ets_in)
        cvalid_models.append(model)
        
    
    index_best = np.argmax(cvalid_ets)
    best_model = cvalid_models[index_best]

    best_model.summary()
    
    ## Test model for each station
    for idx, sta in enumerate(stations):
        
        print(sta)
        pred = np.squeeze(best_model(tst_x[idx]))

        predictions_b = np.array(pred >= 0.5, dtype=int) ## This could change
        
        cm = confusion_matrix(tst_y[idx], predictions_b) ## here 
        hits = cm[0][0]
        false_alarms = cm[0][1]
        misses = cm[1][0]
        correct_negatives = cm[1][1]

        accuracy = (hits + correct_negatives) / np.sum(cm)
        frequency_bias = (hits + false_alarms) / (hits + misses)
        log_odd_ratio = np.log((hits * correct_negatives) / (misses * false_alarms))
        hits_random = ((hits + misses)*(hits + false_alarms)) / np.sum(cm)
        ets = (hits - hits_random)/(hits+misses+false_alarms-hits_random)

        comm.send(ets, dest=p-1, tag=idx)
