import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, grad, jacrev, hessian,jit, jacfwd
import matplotlib.pyplot as plt
import optax

import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import pickle

from functools import partial
import argparse
import os
import sys


parser = argparse.ArgumentParser(description='zhaijiayu test')
parser.add_argument('--test_case', type=str, default="Exp_4_1")
parser.add_argument('--PINN_h', type=int, default=64, help="width of the PINN model")
parser.add_argument('--PINN_L', type=int, default=3, help="depth of the PINN model")
parser.add_argument('--batch_size', type=int, default= 5000)
parser.add_argument('--epochs', type=int, default = 100000)

args = parser.parse_args()

def Accurate_Func_2dr(data):
    scalar = 3.84782606
    V = (jnp.square(data).sum() - 1)**2
    result = (1/scalar)*jnp.exp(- 2 * V)
    return result

def Accurate_Func_6duni(data):
    scalar = (1.02044)**3
    H = 3 *( (data[0]**4 - data[1])**2 + 2 * (data[1]**2) + (data[2]**4 - data[3])**2 + 2 * (data[3]**2) + (data[4]**4 - data[5])**2 + 2 * (data[5]**2) )
    result = jnp.exp(- H)
    return (1/scalar) * result

np.random.seed(0)

if args.test_case == "Exp_4_1":
    dim = 2
    num_ref = 100000
    num_train = 10**5
    center = jnp.array([-0.0056,  0.0026])
    edge_length = 2.1467
    vec_dens_accurate = vmap(Accurate_Func_2dr, 0, 0)
    H = lambda x: 2 * jnp.square(jnp.sum(jnp.square(x)) - 1)
    '''
    This should be imported from google drive data
    '''
    ref_point = jnp.asarray(np.load("/content/drive/MyDrive/Review_Experiments/Jiayu_Zhai_Method/Exp_4_1/ref_data.npy"))
    ref_dens = jnp.asarray(np.load("/content/drive/MyDrive/Review_Experiments/Jiayu_Zhai_Method/Exp_4_1/ref_dens.npy"))
    #train_data = jnp.asarray(np.load("/content/drive/MyDrive/Review_Experiments/Jiayu_Zhai_Method/Exp_4_1/train_data.npy"))
    num_test = 100000
    test_radius = 1.0
    C = 20.0


if args.test_case == "Exp_4_3":
    dim = 6
    num_ref = 200000
    num_train = 200000
    center = jnp.array([-0.0015,  0.0457,  0.0010,  0.0460,  0.0013,  0.0453])
    edge_length = 1.5192
    vec_dens_accurate = vmap(Accurate_Func_6duni, 0, 0)
    H = lambda x: 3 *( (x[0]**4 - x[1])**2 + 2 * (x[1]**2) + (x[2]**4 - x[3])**2 + 2 * (x[3]**2) + (x[4]**4 - x[5])**2 + 2 * (x[5]**2) )
    '''
    This should be imported from google drive data
    '''
    ref_point = jnp.asarray(np.load("/content/drive/MyDrive/Review_Experiments/Jiayu_Zhai_Method/Exp_4_3/ref_data.npy"))
    ref_dens = jnp.asarray(np.load("/content/drive/MyDrive/Review_Experiments/Jiayu_Zhai_Method/Exp_4_3/ref_dens.npy"))
    train_data = jnp.asarray(np.load("/content/drive/MyDrive/Review_Experiments/Jiayu_Zhai_Method/Exp_4_3/train_data.npy"))
    num_test = 500000
    test_radius = 2.0
    C = 100


Ini_layer = [dim] + [args.PINN_h] * (args.PINN_L - 1) + [1]




def init_params(layers):
    keys = jax.random.split(jax.random.PRNGKey(0),len(layers)-1)
    params = list()

    for key,n_in,n_out in zip(keys,layers[:-1],layers[1:]):
        lb, ub = -(1 / jnp.sqrt(n_in)), (1 / jnp.sqrt(n_in))
        W = lb + (ub-lb) * jax.random.uniform(key,shape=(n_in,n_out))
        B = jax.random.uniform(key,shape=(n_out,)) 
        params.append({'W':W,'B':B})

    return params

def fwd(params,x):
    '''
    neural network for the estimated u,
    input: x (dim,)
    output: number
    '''
    X = x
    *hidden,last = params
    for layer in hidden :
        X = jax.nn.tanh(X@layer['W']+layer['B'])
    return (X@last['W'] + last['B'])[0]

vec_fwd = jit(vmap(fwd, in_axes=(None, 0) ))


def Lu_Loss(params, x):
    u = fwd(params, x)
    u_x = jacrev(fwd, argnums=1)(params, x)
    u_xx = jnp.diag(jacfwd(jacrev(fwd, argnums=1), argnums=1)(params, x))
    H_x = jacrev(H, argnums=0)(x)
    H_xx = jnp.diag(jacfwd(jacrev(H, argnums=0), argnums=0)(x))
    return jnp.sum(u_xx) + u * jnp.sum(H_xx) + jnp.sum(u_x * H_x)

vec_Lu_Loss = jit(vmap(Lu_Loss, in_axes=(None, 0) ))

def Batch_Lu_Loss(params, batch):
    result = (vec_Lu_Loss(params, batch))**2
    return result.mean()


def batch_Ref_Loss(params, x_batch, ref_dens_batch):
    result = (vec_fwd(params, x_batch) - ref_dens_batch)**2
    return result.mean()

seeds = np.arange(220000, 2500000, dtype = int )
def resample(i):

    key = jax.random.PRNGKey(seeds[i])

    Data_jax = jax.random.uniform(key, 
                                  minval = center - edge_length, maxval = center + edge_length, 
                                  shape = (args.batch_size, dim))
    return Data_jax

def resample_ref(i):

    key = jax.random.PRNGKey(i)
     
    
    #dim = 2 4000

    random_indexes = jax.random.permutation(key = key, x = num_ref)[:10000]

    return random_indexes

def resample_Lu(i):

    key = jax.random.PRNGKey(i)

    random_indexes = jax.random.permutation(key = key, x = num_train)[:4000]

    return random_indexes

'''
def fit(loss_func_1, loss_func_2, optimizer, Para):
    
    losses = []
    losses_ref = []

    Opt_State = optimizer.init(Para)

    @jit
    def step(param, opt_state, batch_1, batch_2, dens_ref_points):

        loss_value_1, grads = jax.value_and_grad(loss_func_1)(param, batch_1)
        updates, opt_state = optimizer.update(grads, opt_state, param)
        param = optax.apply_updates(param, updates)


        loss_value_2, grads = jax.value_and_grad(loss_func_2)(param, batch_2, dens_ref_points)
        grads = jax.tree.map(lambda y: C * y, grads)
        updates, opt_state = optimizer.update(grads, opt_state, param)
        param = optax.apply_updates(param, updates)

        return loss_value_1, loss_value_2, param, opt_state

    Loss = []
    Loss_ref = []

    for epoch in range(args.epochs):

        #batch_1 = resample(epoch)
        batch_ref_indexes = resample_ref(epoch)
        
        
        batch_ref = ref_point[batch_ref_indexes]
        batch_1 = batch_ref
        batch_dens_ref = ref_dens[batch_ref_indexes]

        loss_value, loss_value_ref, Para, Opt_State = step(Para, Opt_State, batch_1, batch_ref, batch_dens_ref)

        losses.append(loss_value)
        Loss.append(loss_value)
        losses_ref.append(loss_value_ref)
        Loss_ref.append(loss_value_ref)

        if epoch % 200 == 0:
           print('loss value', np.mean(np.array(Loss)) )
           print('loss value', np.mean(np.array(Loss_ref)) )
           print("epoch", epoch)
           Loss = []
           Loss_ref = []

    return Para, losses, losses_ref
'''


def fit(loss_func_1, loss_func_2, optimizer, Para):
    
    losses = []
    losses_ref = []

    Opt_State = optimizer.init(Para)

    @jit
    def step(param, opt_state, batch_1, batch_2, dens_ref_points):

        loss_value_1, grads_1 = jax.value_and_grad(loss_func_1)(param, batch_1)
        loss_value_2, grads_2 = jax.value_and_grad(loss_func_2)(param, batch_2, dens_ref_points)

        combine_grads = jax.tree.map(lambda x, y: x+ C * y, grads_1, grads_2)
        updates, opt_state = optimizer.update(combine_grads, opt_state, param)

        param = optax.apply_updates(param, updates)

        return loss_value_1, loss_value_2, param, opt_state

    Loss = []
    Loss_ref = []

    for epoch in range(args.epochs):

        #batch_1 = resample(epoch)

        batch_ref_indexes = resample_ref(epoch)

        batch_ref = ref_point[batch_ref_indexes]

        batch_1 = batch_ref 
        
        batch_dens_ref = ref_dens[batch_ref_indexes]

        loss_value, loss_value_ref, Para, Opt_State = step(Para, Opt_State, batch_1, batch_ref, batch_dens_ref)

        losses.append(loss_value)
        Loss.append(loss_value)
        losses_ref.append(loss_value_ref)
        Loss_ref.append(loss_value_ref)

        if epoch % 200 == 0:
           print('loss value', np.mean(np.array(Loss)) )
           print('loss value', np.mean(np.array(Loss_ref)) )
           print("epoch", epoch)
           Loss = []
           Loss_ref = []

    return Para, losses, losses_ref


Save_Path = '/content/drive/MyDrive/Review_Experiments/Jiayu_Zhai_Method/'+ args.test_case + '/'

if not os.path.isdir(Save_Path):
    os.mkdir(Save_Path)

Schedule = optax.polynomial_schedule(init_value = 9e-4, end_value=8e-6, power = 2, transition_steps = 70000, transition_begin=10000)
opt = optax.chain(optax.lion(learning_rate= Schedule) )
Initial_param = init_params(Ini_layer)
Final_Param, losses, losses_ref = fit(Batch_Lu_Loss, batch_Ref_Loss, opt, Initial_param)

save_path_param = Save_Path + "Param.pkl"
f = open(save_path_param,"wb")
# write the python object (dict) to pickle file
pickle.dump(Final_Param,f)
f.close()

'''
This part is for Evaluation
'''

#Number of test points

#generate the testing points
np.random.seed(0)
Test_Points = np.random.uniform(low=-test_radius, high=test_radius, size=(num_test, dim))

#This three functions are for relative error computation
def Relative_Error_dist(Estimate, Accurate):
    Error = jnp.abs( (Estimate - Accurate))
    Error_result = Error/Accurate
    return Error_result

def data_filter_high_probability(my_array, Accurate, high_probability):
    result = my_array[Accurate > high_probability]
    return result


def batch_result(data, batch_size, param):
    num_batches = data.shape[0] // batch_size + (data.shape[0] % batch_size != 0)
    batches = np.array_split(data, num_batches)
    KDE_ESTIMATE = vec_fwd(param, batches[0])
    for batch in batches[1:]:
        KDE_ESTIMATE = np.concatenate((KDE_ESTIMATE, vec_fwd(param, batch)) )
    return KDE_ESTIMATE

def relative_error_high_prob(prob_area, test_points, vec_dens_accurate, param, error_description):
    kde_accurate = vec_dens_accurate(test_points)
    points_hpa = data_filter_high_probability(test_points, kde_accurate, prob_area)
    kde_hpa = batch_result(points_hpa, 5000, param)
    acc_hpa = vec_dens_accurate(points_hpa)
    rel_error_hpa = Relative_Error_dist(kde_hpa, acc_hpa)
    df_describe_error = pd.DataFrame(rel_error_hpa)
    error_data_frame = df_describe_error.describe(percentiles = error_description)
    return error_data_frame

Prob = [0.01, 0.05, 0.1]

Highprob1 = relative_error_high_prob(Prob[0], Test_Points, vec_dens_accurate,
                             Final_Param, [0.1, 0.3, 0.5, 0.7])

Highprob2 = relative_error_high_prob(Prob[1], Test_Points, vec_dens_accurate,
                              Final_Param, [0.1, 0.3, 0.5, 0.7])


Highprob3 = relative_error_high_prob(Prob[2], Test_Points, vec_dens_accurate,
                              Final_Param, [0.1, 0.3, 0.5, 0.7])


Error_Table = pd.DataFrame( {" Percentile " : ["10%", "30%",
                                               "50%", "70%", "Mean", " # Points"],

                           "P > " + str(Prob[0]): [round(Highprob1[0]["10%"], 4),
                                                   round(Highprob1[0]["30%"], 4),
                                                   round(Highprob1[0]["50%"], 4),
                                                   round(Highprob1[0]["70%"], 4),
                                                   round(Highprob1[0]["mean"],4),
                                                        str(int(Highprob1[0]["count"]))],


                           "P > " + str(Prob[1]): [round(Highprob2[0]["10%"], 4),
                                                   round(Highprob2[0]["30%"], 4),
                                                   round(Highprob2[0]["50%"], 4),
                                                   round(Highprob2[0]["70%"], 4),
                                                   round(Highprob2[0]["mean"],4),
                                                        str(int(Highprob2[0]["count"]))],

                           "P > " + str(Prob[2]): [round(Highprob3[0]["10%"], 4),
                                                   round(Highprob3[0]["30%"], 4),
                                                   round(Highprob3[0]["50%"], 4),
                                                   round(Highprob3[0]["70%"], 4),
                                                   round(Highprob3[0]["mean"],4),
                                                        str(int(Highprob3[0]["count"]))],
                          } )

Store_Error_Table = Error_Table.to_latex(index=False)

evaluation_file_name = Save_Path + "relative_error.txt"
L2 = jnp.sqrt( ( ( batch_result(Test_Points, 5000, Final_Param) - vec_dens_accurate(Test_Points) )**2 ).mean() )
def store_evaluation():
    with open(evaluation_file_name, 'w') as file:
            file.write(Store_Error_Table + "L2 error" + str(L2))

store_evaluation()









