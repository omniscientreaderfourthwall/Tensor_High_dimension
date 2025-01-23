import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, grad, jacrev, jacfwd, hessian,jit
import argparse
import os

parser = argparse.ArgumentParser(description='Reference Data Generator')
parser.add_argument('--test_case', type=str, default="Exp_4_1")
parser.add_argument('--grid_size', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=0.4)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--s_0', type=int, default=20)
args = parser.parse_args()

@jit
def func_4_1(data):
    result = -1.0 * jnp.asarray([8 * data[0]*(data[0]**2 + data[1]**2 - 1),   8 * data[1]*(data[0]**2 + data[1]**2 - 1)] )
    return result

@jit
def func_4_3(data):

    result = (- 3.0 * jnp.asarray([8 * (data[0]**4 - data[1]) * (data[0]**3),  -2 * (data[0]**4 - data[1]) + 4 * (data[1]),
                          8 * (data[2]**4 - data[3]) * (data[2]**3),  -2 * (data[2]**4 - data[3]) + 4 * (data[3]),
                          8 * (data[4]**4 - data[5]) * (data[4]**3),  -2 * (data[4]**4 - data[5]) + 4 * (data[5])])
                          )
    return result

def sde_trajectory(dim, num_steps, num_traj, step_size, initial_guess, f_func):

    '''
    This function is used to generate sde trajectories
    This is for 4.1, 4.3, 4.4, 4.5
    '''
    key = jax.random.PRNGKey(42)
    random_number = jax.random.normal(key, shape = (2 * num_steps, num_traj, dim))
    vec_f_func = jit(vmap(f_func, 0, 0))
    X = initial_guess
    trajectories = []
    step_size_noise = jnp.sqrt(step_size) * jnp.sqrt(2)

    for i in range(1, num_steps + 1):
        X = X + vec_f_func(X) * step_size  + random_number[i] * step_size_noise
        trajectories.append(X)

    return jnp.asarray(trajectories)

'''
This part is for density computation
'''

def Monte_Carlo_Density(trajectories, point, grid_size):
    dim = (point.shape)[-1]
    uni_result = (( 1 / (2 * grid_size) )**dim) * ( (point < (trajectories + grid_size)) * (point > (trajectories - grid_size)) ).prod(axis = -1)
    return uni_result.mean()

vec_Monte_Carlo_Density = jit( vmap(Monte_Carlo_Density, in_axes = (None, 0, None)) )

def vec_result_density(trajectories, ref_set, grid_size, batch_size):
    num_batches = ref_set.shape[0] // batch_size + (ref_set.shape[0] % batch_size != 0)
    batches = np.array_split(ref_set, num_batches)

    DE_ESTIMATE = vec_Monte_Carlo_Density(trajectories, batches[0], grid_size)
    for batch in batches[1:]:
        DE_ESTIMATE = np.concatenate((DE_ESTIMATE, vec_Monte_Carlo_Density(trajectories, batch, grid_size)
                                      ) )
    return DE_ESTIMATE


'''
This part is for density reference point generation
'''

def resample(i, center, edge_length):
    key = jax.random.PRNGKey(i)
    Data_jax = jax.random.uniform(key, minval = center - edge_length, maxval = center + edge_length, shape = (dim, ) )
    return Data_jax



'''
This part is for running
'''

def ref_data_generation(num_ref, s_0, center, edge_length, alpha, trajec_data):

    t = 300

    ref_data = []

    for i in range(1, num_ref+1):
        key_i = jax.random.PRNGKey(i)
        c_i = jax.random.uniform(key_i, minval = 0.0, maxval = 1.0)

        if c_i <= alpha:
            t  = t + s_0
            ref_data.append(trajec_data[t][0])
        
        else:
            x = resample(i, center, edge_length)
            ref_data.append(x)
    
    return jnp.asarray(ref_data)



key_ini = jax.random.PRNGKey(43)

if args.test_case == "Exp_4_1":
    
    num_train = 10**5
    num_ref = 10**5
    center = jnp.array([-0.0056,  0.0026])
    edge_length = 2.1467
    dim = 2


    initial_guess = 0.2 * jax.random.normal(key_ini, shape = (10, 2))
    sde_argnums = [2, 1500000, 10, 0.001, initial_guess, func_4_1]


traj_data = sde_trajectory(*sde_argnums)
traj_data_dens_est = (jnp.array(traj_data)[1000001:]).reshape((-1, dim))

traj_ref_train_generation = jnp.array(traj_data)[500001:]



'''
generate reference dataset
'''

ref_data_set = ref_data_generation(num_ref, 5, center, edge_length, 0.7, traj_ref_train_generation)
train_data_set = ref_data_set
ref_dens_set = vec_result_density(traj_data_dens_est, ref_data_set, args.grid_size, args.batch_size)

'''
create folder and save the data
'''
Save_Path = '/content/drive/MyDrive/Review_Experiments/Jiayu_Zhai_Method/'+ args.test_case + '/'

if not os.path.isdir(Save_Path):
    os.mkdir(Save_Path)


np.save(Save_Path + "ref_data", ref_data_set)
np.save(Save_Path + "ref_dens", ref_dens_set)
np.save(Save_Path + "train_data", train_data_set)
np.save(Save_Path + "traj_data", traj_data_dens_est)











