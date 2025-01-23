import jax
import jax.numpy as jnp
import numpy as np
import argparse
from tqdm import tqdm

import optax

from jax.example_libraries import optimizers
from functools import partial
import os, sys
import pandas as pd

from functools import partial

parser = argparse.ArgumentParser(description='PINN Training for FP Equaions')
parser.add_argument('--SEED', type=int, default=0, help="random seed")
parser.add_argument('--dataset', type=str, default="Rosenbrock2", help="Rosenbrock1 / Rosenbrock2 / Gaussian")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--epochs', type=int, default=500000, help="Adam training epochs")
parser.add_argument('--dim', type=int, default=6, help="problem dimensionality")
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate for Adam")
parser.add_argument('--PINN_h', type=int, default=128, help="width of the PINN model")
parser.add_argument('--PINN_L', type=int, default=4, help="depth of the PINN model")
parser.add_argument('--save_loss', type=bool, default=False, help="save the optimization trajectory?")
parser.add_argument('--use_sch', type=int, default=2, help="use scheduler?")
parser.add_argument('--N_f', type=int, default=8192, help="num of residual points")
parser.add_argument('--N_test', type=int, default=500000, help="num of test points")
parser.add_argument('--N_quad', type=int, default=256, help="num of quadrature points")
parser.add_argument('--cutoff', type=float, default=1.2, help="cutoff of the domain")
parser.add_argument('--K', type=int, default=5, help="Number of separable subnets")
parser.add_argument('--net_scal', type=float, default= 1.0, help="network_scaling")
parser.add_argument('--accmu_step', type=int, default=5, help="accumulation steps")
parser.add_argument('--dtype', type=str, default="float32", help="dtype, float64 is significantly better than float32")
parser.add_argument('--trunc', type=int, default=3, help="use domain truncator or not, 0 for no truncator, 1,2,3 for various truncators")
parser.add_argument('--res_dist', type=str, default="uniform", help="distribution of the residual points: uniform or normal")

args = parser.parse_args()
print(args)
np.random.seed(args.SEED)
key = jax.random.PRNGKey(args.SEED)
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

O_center = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06])

if args.dtype == "float64": 
   jax.config.update("jax_enable_x64", True)

def func_u(x):
    if args.dataset == "Rosenbrock1":
        temp = -np.sum((x[:, ::2]**2 - x[:, 1::2])**2 + x[:, 1::2]**2, axis=1)
        return temp
    elif args.dataset == "Rosenbrock2":
        temp = np.sum((x[:, ::2]**4 - x[:, 1::2])**2 + 2 * x[:, 1::2]**2, axis=1)
        return -3 * temp
    elif args.dataset == "Gaussian":
        sigma = 1 / 2 / np.pi
        temp = np.sum(x**2, axis=1) / 2
        return -temp / sigma
   
def load_data_Rosenbrock(d):
    args.output_dim = 1
    N_test = args.N_test
    test_x = np.random.uniform(low=-1, high=1, size=(N_test, d))
    test_u = func_u(test_x)
    return test_x, test_u

test_x, test_u = load_data_Rosenbrock(d=args.dim)

def quadrature_1d(N):
    if N == 1:
        coord = jnp.asarray([[0., 2.]])
    elif N == 2:
        coord = jnp.asarray([[-np.sqrt(3) / 3, 1], [np.sqrt(3) / 3, 1]])
    elif N == 3:
        coord = jnp.asarray([[-np.sqrt(15) / 5, 5 / 9],
                            [0, 8 / 9],
                            [np.sqrt(15) / 5, 5 / 9]])
    elif N == 4:
        coord = jnp.asarray([[-np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7), (18 - np.sqrt(30)) / 36],
                            [-np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7), (18 + np.sqrt(30)) / 36],
                            [np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7), (18 + np.sqrt(30)) / 36],
                            [np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7), (18 - np.sqrt(30)) / 36]])
    elif N == 5:
        coord = jnp.asarray([[-1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)), (322 - 13 * np.sqrt(70)) / 900],
                            [- 1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)), (322 + 13 * np.sqrt(70)) / 900],
                            [0, 128 / 225],
                            [1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)), (322 + 13 * np.sqrt(70)) / 900],
                            [1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)), (322 - 13 * np.sqrt(70)) / 900]])
    elif N == 6:
        coord = jnp.asarray([[-0.932469514203152, 0.171324492379170],
                            [-0.661209386466264, 0.360761573048139],
                            [-0.238619186083197, 0.467913934572691],
                            [0.238619186083197, 0.467913934572691],
                            [0.661209386466264, 0.360761573048139],
                            [0.932469514203152, 0.171324492379170]])
    elif N == 7:
        coord = jnp.asarray([[-0.949107912342758, 0.129484966168870],
                            [-0.741531185599394, 0.279705391489277],
                            [-0.405845151377397, 0.381830050505119],
                            [0, 0.417959183673469],
                            [0.405845151377397, 0.381830050505119],
                            [0.741531185599394, 0.279705391489277],
                            [0.949107912342758, 0.129484966168870]])
    elif N == 8:
        coord = jnp.asarray([[-0.960289856497536, 0.101228536290377],
                            [-0.796666477413627, 0.222381034453374],
                            [-0.525532409916329, 0.313706645877887],
                            [-0.183434642495650, 0.362683783378362],
                            [0.183434642495650, 0.362683783378362],
                            [0.525532409916329, 0.313706645877887],
                            [0.796666477413627, 0.222381034453374],
                            [0.960289856497536, 0.101228536290377]])
    elif N == 9:
        coord = jnp.asarray([[-0.968160239507626, 0.0812743883615744],
                            [-0.836031107326636, 0.180648160694858],
                            [-0.613371432700590, 0.260610696402936],
                            [-0.324253423403809, 0.312347077040003],
                            [0.0, 0.330239355001260],
                            [0.324253423403809, 0.312347077040003],
                            [0.613371432700590, 0.260610696402936],
                            [0.836031107326636, 0.180648160694858],
                            [0.968160239507626, 0.0812743883615744]])
    elif N == 10:
        coord = jnp.asarray([[-0.973906528517172, 0.0666713443086881],
                            [-0.865063366688985, 0.149451349150581],
                            [-0.679409568299024, 0.219086362515982],
                            [-0.433395394129247, 0.269266719309997],
                            [-0.148874338981631, 0.295524224714753],
                            [0.148874338981631, 0.295524224714753],
                            [0.433395394129247, 0.269266719309997],
                            [0.679409568299024, 0.219086362515982],
                            [0.865063366688985, 0.149451349150581],
                            [0.973906528517172, 0.0666713443086881]])
    elif N == 11:
        coord = jnp.asarray([[-0.978228658146057, 0.0556685671161737],
                            [-0.887062599768095, 0.125580369464904],
                            [-0.730152005574049, 0.186290210927734],
                            [-0.519096129206812, 0.233193764591991],
                            [-0.269543155952345, 0.262804544510247],
                            [0.0, 0.272925086777901],
                            [0.269543155952345, 0.262804544510247],
                            [0.519096129206812, 0.233193764591991],
                            [0.730152005574049, 0.186290210927734],
                            [0.887062599768095, 0.125580369464904],
                            [0.978228658146057, 0.0556685671161737]])
    elif N == 12:
        coord = jnp.asarray([[-0.981560634246719, 0.0471753363865118],
                            [-0.904117256370475, 0.106939325995318],
                            [-0.769902674194305, 0.160078328543345],
                            [-0.587317954286617, 0.203167426723066],
                            [-0.367831498998180, 0.233492536538356],
                            [-0.125233408511469, 0.249147045813403],
                            [0.125233408511469, 0.249147045813403],
                            [0.367831498998180, 0.233492536538356],
                            [0.587317954286617, 0.203167426723066],
                            [0.769902674194305, 0.160078328543345],
                            [0.904117256370475, 0.106939325995318],
                            [0.981560634246719, 0.0471753363865118]])
    elif N == 13:
        coord = jnp.asarray([[-0.984183054718588, 0.0404840047653159],
                            [-0.917598399222978, 0.0921214998377285],
                            [-0.801578090733310, 0.138873510219789],
                            [-0.642349339440340, 0.178145980761946],
                            [-0.448492751036447, 0.207816047536889],
                            [-0.230458315955135, 0.226283180262898],
                            [0.0, 0.232551553230874],
                            [0.230458315955135, 0.226283180262898],
                            [0.448492751036447, 0.207816047536889],
                            [0.642349339440340, 0.178145980761946],
                            [0.801578090733310, 0.138873510219789],
                            [0.917598399222978, 0.0921214998377285],
                            [0.984183054718588, 0.0404840047653159]])
    elif N == 14:
        coord = jnp.asarray([[-0.986283808696812, 0.0351194603317519],
                            [-0.928434883663574, 0.0801580871597603],
                            [-0.827201315069765, 0.121518570687902],
                            [-0.687292904811685, 0.157203167158193],
                            [-0.515248636358154, 0.185538397477937],
                            [-0.319112368927890, 0.205198463721295],
                            [-0.108054948707344, 0.215263853463158],
                            [0.108054948707344, 0.215263853463158],
                            [0.319112368927890, 0.205198463721295],
                            [0.515248636358154, 0.185538397477937],
                            [0.687292904811685, 0.157203167158193],
                            [0.827201315069765, 0.121518570687902],
                            [0.928434883663574, 0.0801580871597603],
                            [0.986283808696812, 0.0351194603317519]])
    elif N == 15:
        coord = jnp.asarray([[-0.987992518020485, 0.0307532419961174],
                            [-0.937273392400706, 0.0703660474881081],
                            [-0.848206583410427, 0.107159220467172],
                            [-0.724417731360170, 0.139570677926155],
                            [-0.570972172608539, 0.166269205816993],
                            [-0.394151347077563, 0.186161000015562],
                            [-0.201194093997435, 0.198431485327112],
                            [0.0, 0.202578241925561],
                            [0.201194093997435, 0.198431485327112],
                            [0.394151347077563, 0.186161000015562],
                            [0.570972172608539, 0.166269205816993],
                            [0.724417731360170, 0.139570677926155],
                            [0.848206583410427, 0.107159220467172],
                            [0.937273392400706, 0.0703660474881081],
                            [0.987992518020485, 0.0307532419961174]])
    elif N == 16:
        coord = jnp.asarray([[-0.989400934991650, 0.0271524594117540],
                            [-0.944575023073233, 0.0622535239386481],
                            [-0.865631202387832, 0.0951585116824914],
                            [-0.755404408355003, 0.124628971255535],
                            [-0.617876244402644, 0.149595988816578],
                            [-0.458016777657227, 0.169156519395002],
                            [-0.281603550779259, 0.182603415044923],
                            [-0.0950125098376374, 0.189450610455069],
                            [0.0950125098376374, 0.189450610455069],
                            [0.281603550779259, 0.182603415044923],
                            [0.458016777657227, 0.169156519395002],
                            [0.617876244402644, 0.149595988816578],
                            [0.755404408355003, 0.124628971255535],
                            [0.865631202387832, 0.0951585116824914],
                            [0.944575023073233, 0.0622535239386481],
                            [0.989400934991650, 0.0271524594117540]])
    else:
        raise ValueError('This quadrature scheme is not implemented now!')
    return coord[:,0], coord[:,1]

def composite_quadrature_1d(N, a, b, M):
    h = (b - a) / M
    x, w = quadrature_1d(N)
    x = jnp.repeat((x+1)/2, M)*h + jnp.tile(jnp.linspace(a,b,M+1)[:-1], N)
    w_quad = jnp.repeat(w / 2, M) * h

    x_re = np.array([x] * args.dim)

    x_quad = x_re.T + O_center

    return x_quad, w_quad

def estimate_integral_rosenbrock():
    # Here, we generate the two-dimensiosn quadrature for estimating the integral of the Rosenbrock function.
    quad_x, quad_w = composite_quadrature_1d(args.N_quad // 16, -args.cutoff, args.cutoff, 16)
    quad_x = jnp.meshgrid(quad_x, quad_x)
    quad_w = jnp.meshgrid(quad_w, quad_w)
    quad_x, quad_w = jnp.stack(quad_x, axis=-1), jnp.stack(quad_w, axis=-1)
    quad_w = jnp.prod(quad_w, axis=-1, keepdims=True)

    # Computing the Integral of the Rosenbrock function for Normalization of Likelihood
    if args.dataset == "Rosenbrock1":
        temp = (quad_x[:, :, 0:1]**2 - quad_x[:, :, 1:2])**2 + quad_x[:, :, 1:2]**2
    elif args.dataset == "Rosenbrock2":
        temp = (quad_x[:, :, 0:1]**4 - quad_x[:, :, 1:2])**2 + 2 * quad_x[:, :, 1:2]**2
        temp *= 3
    norm_sol = jnp.sum(quad_w * jnp.exp(-temp))
    print("Integral of the Rosenbrock function: ", norm_sol)
    return norm_sol

norm_sol = np.array([(1.02044)**3]) #estimate_integral_rosenbrock() if "Rosenbrock" in args.dataset else jnp.ones(1)
test_u = test_u - jnp.log(norm_sol)# normalize the log-likelihood

def data_filter_high_probability(my_array, Accurate, high_probability):
    result = my_array[Accurate > high_probability]
    return result

print("Maximal value of the PDF: ", np.max(np.exp(test_u)))
if args.dataset == "Rosenbrock1":
    x = data_filter_high_probability(test_x, np.exp(test_u), 0.01)
elif args.dataset == "Rosenbrock2":
    x = data_filter_high_probability(test_x, np.exp(test_u), 0.05)
elif args.dataset == "Gaussian":
    x = data_filter_high_probability(test_x, np.exp(test_u), 0.01)

u = func_u(x) - np.log(norm_sol.item())

print("Minimum value of the PDF on the test points: ", np.exp(u).min())
print("Shape of test points and labels: ", x.shape, u.shape)
u = np.exp(u)

quad_x, quad_w = composite_quadrature_1d(args.N_quad // 16, -args.cutoff, args.cutoff, 16) 
# This quadrature is for estimating the integral of the model

def Multi_TNN(layers, dim, K):
    def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = jax.random.split(key)
          glorot_stddev = 1.0 / np.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * jax.random.normal(k1, (K, dim, d_out, d_in))
          b = jnp.zeros((dim, d_out, 1))
          return W, b
      key, *keys = jax.random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params
    def apply(params, H):
        outputs = H.T.reshape(1, args.dim, -1, 1)

        for W, b in params[:-1]:
            # print(W.shape, H.shape)
            outputs = W @ outputs + b
            outputs = jnp.tanh(outputs)
        W, b = params[-1]
        outputs = W @ outputs + b # K, dim, 1, 1
        outputs = jax.nn.softplus(outputs)
        
        H = H - O_center
        if args.trunc == 1:
            outputs *= ( args.cutoff**2 - jnp.square(H.reshape(1, args.dim, 1, 1)) )
            outputs *= ( args.cutoff**2 >= jnp.square(H.reshape(1, args.dim, 1, 1)) )
        elif args.trunc == 2:
            outputs *= ( 1 - jnp.sum(jnp.square(H.reshape(1, args.dim, 1, 1)), 1) / args.cutoff**2 )**3
            outputs *= ( args.cutoff**2 >= jnp.sum(jnp.square(H.reshape(1, args.dim, 1, 1)), 1) )
        elif args.trunc == 3:
            outputs *= ( 1 - jnp.square(H.reshape(1, args.dim, 1, 1)) / args.cutoff**2 )**3
            outputs *= ( args.cutoff**2 >= jnp.square(H.reshape(1, args.dim, 1, 1)) )
            
        outputs = jnp.prod(outputs, 1) # K, 1, 1
        outputs = jnp.sum(outputs)
        return outputs
    def integral(params, H):
        outputs = H.T.reshape(1, args.dim, -1, 1)
        for W, b in params[:-1]:
            # print(W.shape, H.shape)
            outputs = W @ outputs + b
            outputs = jnp.tanh(outputs)
        W, b = params[-1]
        outputs = W @ outputs + b
        outputs = jax.nn.softplus(outputs) # K, dim, 1, 1
        
        H = H - O_center
        if args.trunc == 1:
            outputs *= ( args.cutoff**2 - jnp.square(H.reshape(1, args.dim, 1, 1)) )
            outputs *= ( args.cutoff**2 >= jnp.square(H.reshape(1, args.dim, 1, 1)) )
        elif args.trunc == 2:
            outputs *= ( 1 - jnp.sum(jnp.square(H.reshape(1, args.dim, 1, 1)), 1) / args.cutoff**2 )**3
            outputs *= ( args.cutoff**2 >= jnp.sum(jnp.square(H.reshape(1, args.dim, 1, 1)), 1) )
        elif args.trunc == 3:
            outputs *= ( 1 - jnp.square(H.reshape(1, args.dim, 1, 1)) / args.cutoff**2 )**3
            outputs *= ( args.cutoff**2 >= jnp.square(H.reshape(1, args.dim, 1, 1)) )

        return outputs[:, :, 0, 0]
    return init, apply, integral

class PINN_TNN:
    def __init__(self):
        self.epoch = args.epochs
        self.dim = args.dim
        self.x = x
        self.u = u.reshape(-1)
        self.quad_x, self.quad_w = quad_x, quad_w

        layers = [1] + [args.PINN_h] * (args.PINN_L - 1) + [args.output_dim]
        self.init, self.apply, self.integral = Multi_TNN(layers, args.dim, args.K)
        params = self.init(rng_key = key)
        self.saved_info = []

        #lr = optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9)
        init_value = args.lr
        Schedule = optax.polynomial_schedule(init_value = init_value, end_value=4e-6, power = 2, transition_steps = int(args.epochs * 0.8/args.accmu_step), transition_begin=0)
        opt = optax.lion(learning_rate= Schedule)
        self.opt_func = optax.MultiSteps(opt, every_k_schedule= args.accmu_step)
        self.opt_state = self.opt_func.init(params)
        self.param_in = params
        #self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr)
        

        self.predict_pinn = jax.vmap(self.neural_net, (None, 0))
        self.predict_integral = jax.vmap(self.integral_net, (None, 0))
        self.predit_residual = jax.vmap(self.Rosenbrock, (None, 0))

        self.loss = self.get_loss_pinn_multi

    def neural_net(self, params, x):
        outputs = self.apply(params, x)
        return outputs
    
    def integral_net(self, params, x):
        outputs = self.integral(params, x)
        return outputs

    def multi_norm(self, params, x):
        ret = self.predict_integral(params, x) # N_grid, K, dim
        # print(ret.shape)
        ret = ret * self.quad_w.reshape(-1, 1, 1)
        ret = jnp.sum(ret, 0) # K, dim
        ret = jnp.prod(ret, 1)
        ret = jnp.sum(ret)
        return ret
    
    def Rosenbrock(self, params, x): 
        u = self.neural_net(params, x)
        u_x = jax.jacrev(self.neural_net, argnums=1)(params, x)
        u_xx = jnp.diag(jax.jacfwd(jax.jacrev(self.neural_net, argnums=1), argnums=1)(params, x))
        H_x = jax.jacrev(self.H, argnums=0)(x)
        H_xx = jnp.diag(jax.jacfwd(jax.jacrev(self.H, argnums=0), argnums=0)(x))
        return jnp.sum(u_xx) + u * jnp.sum(H_xx) + jnp.sum(u_x * H_x)
    
    # Define a compiled update step
    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, opt_state, batch):
        #params = self.get_params(opt_state)
        # g = jax.grad(self.loss)(params, batch)

        loss, grads = jax.value_and_grad(self.loss)(params, batch)
        updates, opt_state = self.opt_func.update(grads, opt_state, params)
        params= optax.apply_updates(params, updates)

        return loss, params, opt_state
    
    
    def Resample(self): # sample random points at the begining of each iteration
        self.xf = 1.0 * np.random.uniform(low=O_center-args.cutoff, high=O_center + args.cutoff, size=(args.N_f, args.dim))
        return
    
    def H(self, x):
        if args.dataset == "Rosenbrock1":
            raise NotImplementedError("This function has not been implemented yet")
        elif args.dataset == "Rosenbrock2":
            x1 = x[::2]
            x2 = x[1::2]
            return 3 * jnp.sum(jnp.square(jnp.square(jnp.square(x1)) - x2) + 2 * jnp.square(x2))
        elif args.dataset == "Gaussian":
            sigma = 1 / 2 / np.pi
            return jnp.sum(x**2 / sigma) / 2
    
    def get_loss_pinn_multi(self, params, x):
        f = self.predit_residual(params, x)
        norm = self.multi_norm(params, self.quad_x)
        f = f / norm
        mse_f = jnp.mean(f**2)
        #f = self.Rosenbrock()
        #mse_f = f.square().mean()
        """u_pred = self.predict_pinn(params, x)
        norm = self.multi_norm(params, self.quad_x)
        u_pred /= norm
        mse_f = jnp.mean(jnp.abs((u_pred -  self.u) / self.u)) # sanity check for the model expressiveness"""
        return mse_f
    
    def train(self):
        opt_state = self.opt_state
        params = self.param_in

        for n in tqdm(range(self.epoch)):
            self.Resample()
            #loss, self.opt_state = self.step(n, self.opt_state, self.xf)
            loss, params, opt_state = self.step(params, opt_state, self.xf)
            if n % 5000 == 0:
                #params = self.get_params(self.opt_state)
                L1 = self.L1_pinn(params)
                print('epoch %d, loss: %e, l1: %e'%(n, loss, L1))
                Error_Table = self.error_table(params)

        Store_Error_Table = Error_Table.to_latex(index=False)
        Save_Path = "" #whatever path you like
        file_name = Save_Path + args.dataset + "cutoff" + str(args.cutoff) +"relative_error" + "h" + str(args.PINN_h) + "L" + str(args.PINN_L) + "K" + str(args.K) + ".txt"
        with open(file_name, 'w') as file:
            file.write(Store_Error_Table)

    #@partial(jax.jit, static_argnums=(0,))
    def L1_pinn(self, params):
        pred_u = self.predict_pinn(params, self.x)
        norm = self.multi_norm(params, self.quad_x)
        pred_u /= norm
        print("Mean of pred_u and u on the test points (should be similar for sanity check): ", pred_u.mean(), self.u.mean())
        prediction, label = self.u - pred_u, self.u
        L1 = np.mean(np.abs(prediction / label))
        return L1
    def relative_error_high_prob(self, params, high_prob, error_description):
        idx = np.where(self.u > high_prob)[0]
        x, u = self.x[idx], self.u[idx]

        pred_u = self.predict_pinn(params, x)
        norm = self.multi_norm(params, self.quad_x)
        pred_u /= norm
        prediction, label = u - pred_u, u
        rel_error_hpa = np.abs(prediction / label)
        df_describe_error = pd.DataFrame(rel_error_hpa)
        error_data_frame = df_describe_error.describe(percentiles = error_description)
        return error_data_frame

    def error_table(self, params):
        Highprob1 = self.relative_error_high_prob(params, 0.05, [0.1, 0.3, 0.5, 0.7])
        Highprob2 = self.relative_error_high_prob(params, 0.25, [0.1, 0.3, 0.5, 0.7])
        Highprob3 = self.relative_error_high_prob(params, 0.5, [0.1, 0.3, 0.5, 0.7])

        Prob = [0.05, 0.25, 0.5]
        Percentile = ["10%", "30%", "50%", "70%"]

        Error_Table = pd.DataFrame( {" Percentile " : ["10%", "30%",
                                                    "50%", "70%", "Mean", "Number of Points"],

                            "Prob > " + str(Prob[0]): [Highprob1[0]["10%"],
                                                        Highprob1[0]["30%"],
                                                        Highprob1[0]["50%"],
                                                        Highprob1[0]["70%"],
                                                        Highprob1[0]["mean"],
                                                        str(int(Highprob1[0]["count"]))],


                         "Prob > " + str(Prob[1]): [Highprob2[0]["10%"],
                                                        Highprob2[0]["30%"],
                                                        Highprob2[0]["50%"],
                                                        Highprob2[0]["70%"],
                                                        Highprob2[0]["mean"],
                                                        str(int(Highprob2[0]["count"]))],

                         " Prob > " + str(Prob[2]): [Highprob3[0]["10%"],
                                                     Highprob3[0]["30%"],
                                                     Highprob3[0]["50%"],
                                                     Highprob3[0]["70%"],
                                                     Highprob3[0]["mean"],
                                                     str(int(Highprob3[0]["count"])) ],
                          } )

        print(Error_Table)
        return Error_Table

model = PINN_TNN()
model.train()


