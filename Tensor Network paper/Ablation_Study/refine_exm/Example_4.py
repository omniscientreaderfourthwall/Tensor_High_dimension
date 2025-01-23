#This is for the example 2 in the paper
#I have modified Example 2 before.
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, grad, jacrev, hessian, jit, jacfwd
import optax
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import pickle

from functools import partial
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='TRBFN Training for FKP Equaions')

parser.add_argument('--test_case', type=str, default="Exp 4.4")
parser.add_argument('--Chosen_List', type=list, default= ["wendland", "wendland", "wendland"])
parser.add_argument('--rank', type = dict, default = 800)
parser.add_argument('--rbf_types', type = str, default = "three_one", help = "three_one/two")
parser.add_argument('--epochs', type = int, default = 200000)
parser.add_argument('--batches', type = int, default = 20000)
parser.add_argument('--Train', type = bool, default = True)
parser.add_argument('--m', type = int, default = 6)
parser.add_argument('--scale_r', type = float, default = 1.0)
parser.add_argument('--r_sde', type = float, default = 4.4)

args = parser.parse_args()

'''
OC: original code.
This is for the translation.
'''

'''
Part 1
Problem Definition
'''

Center_O = jnp.array([-0.03225323, 0.05983249, -0.00454075, 0.01972704, -0.00361719, 0.00547895])


def Accurate_Func_6dm(data):
    inside_exp = ( -2.0 * ( (data[:3]**2).sum() + 0.5 * (data[0]*data[1] + data[0] * data[2] + data[1] * data[2]))
                -0.5 * ( ( data[3:]**2).sum() + 0.2 *(data[3]*data[4] + data[3]* data[5] + data[4] * data[5]) ) )

    result = ( (data[:2]**2 + 0.02).prod()) * jnp.exp(inside_exp) * (1/15.9748) * (1/0.2033)
    return result

vec_dens_accurate = vmap(Accurate_Func_6dm, 0, 0)

#Function which helps the computation of loss functional
func_into = ( lambda data: ( - 1 * jnp.asarray([4 * data[0] +   1. * (data[1] + data[2]) - 2 * data[0]/(data[0]**2 + 0.02),
                                  4 * data[1] + 1 * (data[0] + data[2]) - 2 * data[1]/(data[1]**2 + 0.02),
                                  4 * data[2] + 1 * (data[0] + data[1]),

                                  1 * data[3] + 0.1 * (data[4] + data[5]) ,
                                  1 * data[4] + 0.1 * (data[3] + data[5]) ,
                                  1 * data[5] + 0.1 * (data[3] + data[4]) ,]
                           ) ) )

div_into = ( lambda data: (-1.0 * (4.0 +  2 * (data[0]**2 - 0.02)/(data[0]**2 + 0.02)**2 +
                       4.0 +  2 * (data[1]**2 - 0.02)/(data[1]**2 + 0.02)**2 +
                       4.0 +
                       1.0  +
                       1.0  +
                       1.0 )
                 )
   )


'''
Part 2
This part group the parameters of the shape of parameters
'''
#dimension of the input
dim = 6

#N in the paper,  (OC: rank_A, rank_B = args.rank["A"], args.rank["B"] )
rank = args.rank

#m_use is one third/half of the number of k_{ij}^{\ell}, depends on different types of RBFs

#Shape this is the shape used to generate scale / coeff with thin the innerest layer
#OC: Shape = (1, 3, 1, rank_A, dim) if args.structure == "structure_A" else (1, 3, m_use, rank_B, dim)

if args.rbf_types == "three_one":
   m_use = args.m//3
   Shape = (3, m_use, rank, dim)
else:
   m_use = args.m//2
   Shape = (2, m_use, rank, dim)


#radius we generate from SDE
r_base = args.r_sde

#This part controls the scale of radius
scale_r = args.scale_r

#radius we use in generate sample points
r = r_base * scale_r

#random key for initialization
key_ini = jax.random.PRNGKey(1243)

#argument list for initialization
#OC: Boundary_Shift, Boundary_Width, args_init =[1.0, 1.0, 1.0], [0.7, 0.9, 1.0], [key_s, 4, 0.9 * r, r, Shape, jnp.sqrt(r)]

args_init = [key_ini, 0.9 * r, jnp.sqrt(r), r, Shape]


#code for generating the initial value of parameters name: def Initialization_Generation

def Initialization_Generation(key, width_ini, dist_ini_shift, constraint_ini_shift, shape):
    Param_Ini = {
    }
    
    num_points = 10 * (np.array(shape)).prod()
    num_shifts = (np.array(shape[:])).prod()

    #Generation of Shift from a certain distribution
    Shift_Points = (dist_ini_shift * jax.random.normal(key, shape = (num_points, )))
    #constrain the shift within the radius
    Param_Ini["shifts"] = (Shift_Points[jnp.abs(Shift_Points) < constraint_ini_shift])[:num_shifts].reshape(shape[:]) + Center_O

    #Generate the initialization of width:
    Param_Ini["width"] = jnp.zeros(shape) + width_ini
    #Generate the initialization of alpha_1:
    Param_Ini["alpha_1"] = jnp.zeros(shape) + 0.6
    
    #Generate the initialization of alpha_2 
    Param_Ini["alpha_2"] = jnp.zeros(shape[1:]) + 0.6
    
    #Generate the coefficients c:
    Param_Ini["coeff"] = jnp.zeros((shape[2],)) + 0.6

    return Param_Ini
    
initial_param = Initialization_Generation(*args_init)


'''
This is the group of implementation of RBF function and their corresponding derivatives
'''

#Types of RBF function list
Chosen_List = args.Chosen_List

#The lower bound of the scale/bandwidth for different kinds of kernels h_{ij}^{(\ell)}
#(OC: bd ={"wendland": 3e-02,"inverse_quadratic": 0.01} )
scale_bound = {"wendland": 3e-02,"inverse_quadratic": 0.01}

#Basis Function Part
'''
version of explicit integral for three types of RBF function
'''
#gaussian integral
@jit
def gaussian_integral(sigma, shift):
    sigma_t =  1/ (5e-03 + jnp.square(sigma) )
    results = 0.5 * ( jax.lax.erf( jnp.sqrt(sigma_t/2) * (r + shift) ) - jax.lax.erf( jnp.sqrt(sigma_t/2) * (-r + shift) ) )
    return results

#inverse quadratic integral:
@jit
def inverse_quadratic_integral(epsilon, shift):
    epsilon_sq = jnp.square(jnp.abs(epsilon) + scale_bound["inverse_quadratic"])
    func = lambda insi: ( insi * ( 3.0 * epsilon_sq + 2.0 * (insi**2) ) )/(4.0 * ((epsilon_sq + insi**2)**1.5))
    results = func(r - shift) - func(-r - shift)
    return results

#wendland kernel integral
def wendland_integral(h, shift):
    return 1.0

'''
version of RBF function for evaluation
'''
#gaussian kernel
def gaussian_1_test(sigma, data, shift):

    input = data - shift
    Integral_scaling = gaussian_integral(sigma, shift)
    sigma_t =  1/ (5e-03 + jnp.square(sigma) )
    dim_each_result = (1/Integral_scaling) * ( jnp.sqrt(sigma_t)/jnp.sqrt(2 * np.pi) ) * jnp.exp( -0.5 * sigma_t * jnp.square(input))

    return dim_each_result

#inverse quadratic kernel
def inverse_quadratic_1_test(epsilon, data, shift):
    input = data - shift
    Integral_scaling = inverse_quadratic_integral(epsilon, shift)
    epsilon_sq = jnp.square(jnp.abs(epsilon) + scale_bound["inverse_quadratic"])
    dim_each_result = (1/Integral_scaling) * (0.75 * (epsilon_sq)**2) * ( 1/(epsilon_sq + jnp.square(input))**2.5 )
    return dim_each_result


def wendland_1_test(h, data, shift_w):
    shift_ = shift_w

    #h_deno = 3e-02 + jnp.abs( (r - jnp.abs(shift_)) * jnp.tanh(h) )

    h_deno = scale_bound["wendland"] + jnp.abs(h)

    h_sq = 1/h_deno

    input = data - shift_
    dim_each_result = 1.25 * h_sq * jnp.power(jax.nn.relu(1 - jnp.abs(h_sq * input)) , 3) * ( 3.0 * jnp.abs(h_sq * input) + 1 )
    return dim_each_result

'''
RBF derivatives and functions for PINN loss function
'''

def gaussian_1(sigma, data, shift):
    input = data - shift
    Integral_scaling = gaussian_integral(sigma, shift)

    sigma_t =  1/ (5e-03 + jnp.square(sigma) )

    dim_each_result = (1/Integral_scaling) * ( jnp.sqrt(sigma_t)/jnp.sqrt(2 * np.pi) ) * jnp.exp( -0.5 * sigma_t * jnp.square(input))
    grad_dim_each_result = (1/Integral_scaling) * (-sigma_t * input * dim_each_result)
    hessian_dim_each_result = (1/Integral_scaling) * (jnp.square(sigma_t) * jnp.square(input) - sigma_t) * dim_each_result

    return dim_each_result, grad_dim_each_result, hessian_dim_each_result



def inverse_quadratic_1(epsilon, data, shift):
    input = data - shift
    Integral_scaling = inverse_quadratic_integral(epsilon, shift)

    epsilon_sq = jnp.square(jnp.abs(epsilon) + scale_bound["inverse_quadratic"])

    inver_ele = 1/(epsilon_sq + jnp.square(input))

    dim_each_result = (1/ Integral_scaling) * (0.75 * (epsilon_sq)**2 * ( inver_ele**2.5 ))
    grad_dim_each_result = (1/ Integral_scaling) * inver_ele * (-5 * input) * dim_each_result
    hessian_dim_each_result = (1/ Integral_scaling) * (grad_dim_each_result * (-7.0 * input) - dim_each_result * 5.0) * inver_ele
    return dim_each_result, grad_dim_each_result, hessian_dim_each_result



def wendland_1(h, data, shift_w):

    shift_ = shift_w

    h_deno = scale_bound["wendland"] + jnp.abs(h)

    h_sq = 1/h_deno

    input = data - shift_
    abs_in = jnp.abs(h_sq * input)
    ele = jax.nn.relu(1 - abs_in)
    dim_each_result = 1.25 * h_sq * (ele**3) * ( 3.0 * abs_in + 1 )
    grad_dim_each_result = 1.25 * h_sq * (-12.0 * (h_sq**2) * (input) ) * (ele**2)
    hessian_dim_each_result = (1.25 * h_sq) * (12 * (h_sq**2)) * ele * (3 * abs_in - 1)

    return dim_each_result, grad_dim_each_result, hessian_dim_each_result

'''
This is a dictionary which includes the RBF's function's integral, function for test, and function for pinn loss function
'''
Kernel_Dic = {
    "gaussian": (gaussian_integral, gaussian_1_test, gaussian_1),
    "inverse_quadratic":(inverse_quadratic_integral,  inverse_quadratic_1_test,  inverse_quadratic_1),
    "wendland":(wendland_integral, wendland_1_test, wendland_1)
}

k1, k2, k3 = Kernel_Dic[Chosen_List[0]], Kernel_Dic[Chosen_List[1]], Kernel_Dic[Chosen_List[2]]

#This part is for evaluation
@jit 
def combine_k(bandwidth, dist_pa, data, shifts):
    result = (dist_pa[0] * k1[1](bandwidth[0], data, shifts[0]) +
              dist_pa[1] * k2[1](bandwidth[1], data, shifts[1]))

    if args.rbf_types == "three_one":
       result = result + dist_pa[2] * k3[1](bandwidth[2], data, shifts[2])

    return result


#This part is for computation of loss functional
@jit
def combine_k_no_bp(bandwidth, dist_pa, data, shift):
    r1, grad1, h1 = k1[2](bandwidth[0], data, shift[0])
    r2, grad2, h2 = k2[2](bandwidth[1], data, shift[1])

    if args.rbf_types == "three_one":
       r3, grad3, h3 = k3[2](bandwidth[2], data, shift[2])

    
    if args.rbf_types == "three_one":
       def func_comb(x1, x2, x3):
           result =dist_pa[0] * x1 + dist_pa[1] * x2 + dist_pa[2] * x3
           return result
       comb_r = func_comb(r1, r2, r3)
       comb_g = func_comb(grad1, grad2, grad3)
       comb_h = func_comb(h1, h2, h3)

    else:
       def func_comb(x1, x2):
           result =dist_pa[0] * x1 + dist_pa[1] * x2
           return result

       comb_r = func_comb(r1, r2)
       comb_g = func_comb(grad1, grad2)
       comb_h = func_comb(h1, h2)

    return comb_r, comb_g, comb_h


'''
This part is for the evaluation function and training function
'''

#Factor which multiplied by the loss function (OC: Untrain_Param = {"normalizer": 10.0} )
Factor_Loss = 10.0 

def KDE(param, data):

    alpha_1 = jnp.square(param['alpha_1'])/(jnp.square(param['alpha_1'])).sum(axis = 0)
    alpha_2 = jnp.square(param['alpha_2']) / (jnp.square(param['alpha_2'])).sum(axis = 0)
    coeff = jnp.square(param["coeff"]) /(jnp.square(param["coeff"])).sum()

    output = (     (alpha_2 * combine_k(param["width"],
                                               alpha_1,
                                               data,
                                               param["shifts"])).sum(axis = 0)
                                                  ).prod(axis = -1)
    result = ( (coeff * output ) ).sum() * Factor_Loss
    return result

#training loss functional
def KDE_no_bp(param, data):

    alpha_1 = jnp.square(param['alpha_1'])/(jnp.square(param['alpha_1'])).sum(axis = 0)
    alpha_2 = jnp.square(param['alpha_2']) / (jnp.square(param['alpha_2'])).sum(axis = 0)
    coeff = jnp.square(param["coeff"]) /(jnp.square(param["coeff"])).sum()

    KDE_, Grad_, Hessian_ = combine_k_no_bp(param["width"], alpha_1, data, param["shifts"])
    func_alpha = lambda yi: (alpha_2 * yi).sum(axis = 0)
    output_KDE = func_alpha(KDE_)
    output_KDE_grad = func_alpha(Grad_)
    output_KDE_hessian = func_alpha(Hessian_)

    Lp = 0
    fun_d_v = -1 * func_into(data)

    for i in range(dim):
        List_choose = list(set(range(dim)) - set([i]))

        KDE_choose = ( (  output_KDE[:, List_choose]).prod(axis = -1)) * Factor_Loss

        grad_value = ( ( (  output_KDE_grad[:, i] * KDE_choose) * coeff)).sum()
        Laplacian_value = ( ( ( output_KDE_hessian[:, i] * KDE_choose) * coeff )).sum() 
        Lp = grad_value * fun_d_v[i] + Laplacian_value + Lp

    Lp = Lp - ( (coeff * KDE_choose * output_KDE[:, -1]).sum() ) * div_into(data)

    return Lp


#vectorized evaluation function
vec_KDE = jit( vmap(KDE, in_axes = (None, 0) , out_axes = 0) )

#vectorized Lp
vectorize_Lp = jit( vmap(KDE_no_bp, in_axes = (None, 0) , out_axes = 0) )

'''
This part is for the real loss functional
'''

'''
This is for the KDE_no_bp part
'''

def Monte_Functional(param, data_batch):
    L_p_result = vectorize_Lp(param, data_batch)
    Output = (jnp.square(L_p_result)).mean()
    return Output

'''
boundary condition
'''


'''
penalty of constraints of parameters
'''
#OC: def H_S(param)
def penalty_constraint_param(param):
    #r_j - |s_{ij}^{(\ell)} - O_j|
    r_1_constraint_result = jnp.abs(param["shifts"] - Center_O) - r

    #|s_{ij}^{(\ell)} - O_j| < r_j
    shifts_constraint_penalty = (jax.nn.relu(r_1_constraint_result)).mean()
    
    #|h_{ij}^{(\ell)}| < |r_j - |s_{ij}^{(\ell)} - O_j||
    width_constraint_penalty =  ( jax.nn.relu(  jnp.abs(param["width"])[0] + scale_bound[Chosen_List[0]] -  jnp.abs(r_1_constraint_result[0]) ).mean()
                                 + jax.nn.relu( jnp.abs(param["width"])[1] + scale_bound[Chosen_List[1]] -  jnp.abs(r_1_constraint_result[1]) ).mean()
                                 )
    if args.rbf_types == "three_one":
        width_constraint_penalty = width_constraint_penalty + jax.nn.relu( jnp.abs(param["width"])[2] + scale_bound[Chosen_List[2]] -  jnp.abs(r_1_constraint_result[2]) ).mean()
    
    Penalty = width_constraint_penalty + shifts_constraint_penalty

    return Penalty


'''
penalty of boundary condtion
'''

def Boundary_Control(param):

    left_end = Center_O - r * jnp.array([1.0] * dim)
    right_end = Center_O + r * jnp.array([1.0] * dim)

    alpha_1 = jnp.square(param['alpha_1'])/(jnp.square(param['alpha_1'])).sum(axis = 0)
    alpha_2 = jnp.square(param['alpha_2']) / (jnp.square(param['alpha_2'])).sum(axis = 0)

    output_left =  (alpha_2 * combine_k(param["width"],
                                               alpha_1,
                                               left_end,
                                               param["shifts"])  ).sum(axis = 0)
    
    output_right =  (alpha_2 * combine_k(param["width"],
                                               alpha_1,
                                               right_end,
                                               param["shifts"])  ).sum(axis = 0)
    
    result = (jnp.abs(output_left)).mean() + (jnp.abs(output_right)).mean()
    return result

'''
Loss function
'''
Loss_Params = {"reg_bound": 100.0,"reg_diff": 1.0, "penal_param":5e4}
def Loss_Func(param, data_batch):

    loss = ( Loss_Params["reg_diff"] * Monte_Functional(param, data_batch)
             + Loss_Params["reg_bound"] * Boundary_Control(param)
             + Loss_Params["penal_param"] * penalty_constraint_param(param)
             )
    return loss


#seed generator / sample generation
seeds = np.arange(220000, 2500000, dtype = int )#starting at 200000

'''
build the saving directories
'''

bw= "rk" + str(Shape[-2]) + "_" + "r" + str(r) + "_" + 'm_use' + str(m_use) + 'rbf_types' + args.rbf_types
kn = "kl"+ Chosen_List[0][0] + Chosen_List[1][0] + Chosen_List[2][0]

Path_Name = args.test_case[-3:] + "_" + "_" + bw + "_" + kn + "/"

Save_Path = '/content/drive/MyDrive/Review_Experiments/'+ Path_Name

if not os.path.isdir(Save_Path):
    os.mkdir(Save_Path)


'''
This part is for training function
'''


#Training Function
def fit(loss_func, optimizer, resampler, Para):

    losses = []
    Opt_State = optimizer.init(Para)

    @jit
    def step(param, opt_state, batch):
        loss_value, grads = jax.value_and_grad(loss_func)(param, batch)
        updates, opt_state = optimizer.update(grads, opt_state, param)
        param = optax.apply_updates(param, updates)
        return loss_value, param, opt_state
    Loss = []
    fh = open(Save_Path + 'output.txt', 'w')
    original_stderr = sys.stderr
    sys.stderr = fh

    for epoch in tqdm.tqdm(range(args.epochs)):

        Dataset = resampler(epoch)

        loss_value, Para, Opt_State = step(Para, Opt_State, Dataset)

        losses.append(loss_value)
        Loss.append(loss_value)

        if epoch % 500 == 0:
           print('loss value', np.mean(np.array(Loss)) )
           print("epoch", epoch)
           Loss = []

    sys.stderr = original_stderr
    fh.close()

    return losses, Para


seeds = np.arange(220000, 2500000, dtype = int ) #starting at 200000

def resample(i):
    key = jax.random.PRNGKey(seeds[i])
    Data_jax = jax.random.uniform(key, minval = Center_O - r, maxval = Center_O + r, shape = ((args.batches, dim)) )
    return Data_jax

Schedule = optax.polynomial_schedule(init_value = 9e-4, end_value=8e-6, power = 2, transition_steps = 70000, transition_begin=10000)

save_path_param = Save_Path + "Param.pkl"

def training():

    opt = optax.chain(
             optax.clip(100.0),
             optax.lion(learning_rate= Schedule))
    optimizer =optax.MultiSteps(opt, every_k_schedule= 2)
    losse_, Param = fit(Loss_Func, optimizer, resample, initial_param)
    f = open(save_path_param,"wb")
    # write the python object (dict) to pickle file
    pickle.dump(Param,f)
    f.close()
    return Param, losse_

if args.Train == True:
   Final_Param, Loss = training()

else:
   Final_Param = pickle.load(open(save_path_param, "rb"))


'''
This part is for Evaluation
'''

#Number of test points
num_test = 500000
#test radius
test_radius = 2.0
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
    KDE_ESTIMATE = vec_KDE(param, batches[0])
    for batch in batches[1:]:
        KDE_ESTIMATE = np.concatenate((KDE_ESTIMATE, vec_KDE(param, batch)) )
    return KDE_ESTIMATE/10.0

def relative_error_high_prob(prob_area, test_points, vec_dens_accurate, param, error_description):
    kde_accurate = vec_dens_accurate(test_points)
    points_hpa = data_filter_high_probability(test_points, kde_accurate, prob_area)
    kde_hpa = batch_result(points_hpa, 5000, param)
    acc_hpa = vec_dens_accurate(points_hpa)
    rel_error_hpa = Relative_Error_dist(kde_hpa, acc_hpa)
    df_describe_error = pd.DataFrame(rel_error_hpa)
    error_data_frame = df_describe_error.describe(percentiles = error_description)
    return error_data_frame

Prob = [2e-04, 1e-03, 5e-03]

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

Loss_Range = range(args.epochs)
plt.plot(Loss_Range, np.log10(np.array(Loss)), color='green')
plt.xlabel('Epochs')
plt.ylabel('Losses(log_10)')
plt.savefig(Save_Path + 'Losses_Graph')

np.save(Save_Path + 'Losses', np.array(Loss))







