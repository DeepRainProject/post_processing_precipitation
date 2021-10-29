import json
import numpy as np
import tensorflow as tf

from mpi4py import MPI
from validate import validate
from sklearn.metrics import *


comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

data_path = '/p/home/jusers/rojascampos1/juwels/MyProjects/PROJECT_deepacf/deeprain/rojascampos1/precipitation_prediction/data/'

# depth = [1,2,3,4,5,6,7,8,9,10]
# width = [2, 4, 8, 16, 32, 64, 128]

depth = [2, 4, 6, 8]
width = [4, 8, 16, 32, 64]

station = 'redlen' ## Falta redlen

n_runs = 5


### Calculates all possible conditions
configs = []

for layers in depth:
    for units in width:
        configs.append([layers, units])




## Master process
if my_rank == p-1:
    
    # Collect results in a dictionary
    results = np.empty([len(depth), len(width), n_runs])
    
    sender = 0
    for i, d in enumerate(depth):
        for j, u in enumerate(width):
            skills = comm.recv(source=sender, tag=000)
            print('received', skills)
            sender = sender+1
            results[i][j] = np.array(skills)
    
    np.save('arch_exploration_'+station+'.npy', results)
            
    
            
            


else: 
    layers  = configs[my_rank][0]
    neurons = configs[my_rank][1]

    ## Load datasets
    trn_x = np.loadtxt(data_path + station + '_0_4_5x5_allFeatures_created_2020-10-23/train_x.csv', delimiter=',')
    trn_y = np.loadtxt(data_path + station + '_0_4_5x5_allFeatures_created_2020-10-23/train_y.csv', delimiter=',')
    trn_t = np.load(data_path + station + '_0_4_5x5_allFeatures_created_2020-10-23/timepoints_train.npy')
    
    tst_x = np.loadtxt(data_path + station + '_0_4_5x5_allFeatures_created_2020-10-23/test_x.csv', delimiter=',')
    tst_y = np.loadtxt(data_path + station + '_0_4_5x5_allFeatures_created_2020-10-23/test_y.csv', delimiter=',')
    tst_t = np.load(data_path + station + '_0_4_5x5_allFeatures_created_2020-10-23/timepoints_test.npy')
    
    id_predictors   = np.loadtxt('/p/project/deepacf/deeprain/rojascampos1/precipitation_prediction/data/00_gordons_regression'+'/'+station+'/'+station+ '_ID_Predictors.txt', dtype=np.int32,delimiter=',')
    id_predictors   = id_predictors - 1 ### Matlab index start in 1
    trn_x = trn_x[:, id_predictors]
    tst_x = tst_x[:, id_predictors]
    
    print('ID =', my_rank,  'trn:', trn_x.shape, '--->', trn_y.shape, 'tst:', tst_x.shape, '--->', tst_y.shape)

    trn_y = np.exp(trn_y) ### revert log transform 
    tst_y = np.exp(tst_y)
    
    ## Train n models
    batch_size = 128
    epochs = 16

    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanSquaredLogarithmicError()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)    
    
    results_runs = []
    
    for r in range(n_runs):
    
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(trn_x.shape[1])))
        for l in range(layers):
            model.add(tf.keras.layers.Dense(neurons, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
    
        model.compile(optimizer, loss)

        hist = model.fit(trn_x, trn_y, batch_size, epochs, verbose=1, validation_split=0.2, shuffle=True)    

        ## Test model
        preds = model(tst_x)
        preds = np.squeeze(preds)
        leps       = validate(preds, tst_t, [0], [4], False, station)['LEPS']
        leps_cosmo = validate([], tst_t, [0], [4], True, station)['LEPS'][9]
        skill = 1 - (leps / leps_cosmo)
        results_runs.append(skill)
    
    
    print('My rank =', my_rank, 'layers =', layers, 'neurons =', neurons, 'median_skill =', np.median(results_runs))

    comm.send(results_runs, dest=p-1, tag=000)
        
