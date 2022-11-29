import torch
import numpy as np

def Griewank(x):
  # evaluates function
  # inputs:
  #   x     = feature vector of size (n_samples x n_features)
  # outputs: 
  #   f_val = scalar function evaluated at input features of size (n_samples x 1)
  n_features        = x.shape[1]
  n_samples         = x.shape[0]
  temp_vec          = torch.FloatTensor(range(1,n_features+1))
  denom             = torch.sqrt(temp_vec)
  cos_vec           = torch.cos(x/denom)
  f_val             = 1 + torch.sum(x*x, dim=1)/4000 - torch.prod(cos_vec, dim=1)
  return f_val


def Drop_Wave(x):
  n_features        = x.shape[1]
  n_samples         = x.shape[0]
  f_val             = torch.zeros(n_samples)

  squared_sum = x[:,0]**2 + x[:,1]**2
  f_val = ( 1 + torch.cos(12*torch.sqrt(squared_sum)) ) / ( 0.5 * squared_sum + 2 )
  f_val = -f_val
  return f_val

def AlpineN1(x):
  n_features        = x.shape[1]
  n_samples         = x.shape[0]
  f_val             = torch.zeros(n_samples)

  f_val = torch.sum(torch.abs(x * torch.sin(x) + 0.1*x), dim=1)
  # for i in range(n_features):
  #   f_val = f_val + torch.abs(x[:,i] * torch.sin(x[:,i]) + 0.1*x[:,i])
  return f_val

def Ackley(x):
  n_features        = x.shape[1]
  n_samples         = x.shape[0]
  f_val             = torch.zeros(n_samples)

  a = 20
  b = 0.2
  c = 2*np.pi

  cos_sum_term = (1/n_features) * torch.sum( torch.cos(c*x), dim=1)
  quad_sum_term = (1/n_features) * torch.sum( x*x, dim=1)

  f_val = -a * torch.exp( -b * torch.sqrt( quad_sum_term) ) - torch.exp(cos_sum_term) + a + np.exp(1)

  return f_val

def Levy(x):
  n_features        = x.shape[1]
  n_samples         = x.shape[0]
  f_val             = torch.zeros(n_samples)

  w = 1 + (x-1)/4
  init_term = torch.sin(np.pi*w[:,0]) ** 2
  fin_term  = (w[:,n_features-1] - 1)**2 * (1 + torch.sin(2*np.pi*w[:,n_features-1]))

  for i in range(n_features-1):
    f_val = f_val + (w[:,i]-1)**2 * ( 1 + 10*torch.sin( np.pi*w[:,i] + 1)**2 )

  f_val = f_val + init_term + fin_term

  return f_val

def Rastrigin(x):
  n_features        = x.shape[1]
  n_samples         = x.shape[0]
  f_val             = torch.zeros(n_samples)

  f_val = torch.sum(x**2 - 10*torch.cos(2*np.pi*x), dim=1)
  f_val = f_val + 10*n_features

  return f_val
    
    
# ----------------------------------------------------------------------
# Numpy Versions of functions above for built-in python solvers
# ----------------------------------------------------------------------
def Griewank_numpy(x):

  dim = x.shape[0]
  # temp_vec          = np.asarray(range(1,dim+1))
  # denom             = np.sqrt(temp_vec)
  # cos_vec           = np.cos(x/denom)

  cos_vec = np.zeros(dim)
  for i in range(dim):
    cos_vec[i] = np.cos(x[i]/np.sqrt(i+1))

  f_val             = 1 + np.sum(x*x)/4000 - np.prod(cos_vec)
  return f_val

def Drop_Wave_numpy(x):
  # must be 2D
  sum_squared_term = x[0]**2 + x[1]**2
  f_val = -( 1 + np.cos(12 * np.sqrt(sum_squared_term)) )/( 0.5 * sum_squared_term + 2)
  return f_val

def Rastrigin_numpy(x):
  n_features = len(x)

  f_val = np.sum(x**2 - 10*np.cos(2*np.pi*x))
  f_val = f_val + 10*n_features

  return float(f_val)

def Levy_numpy(x):
  n_features        = x.shape[0]
  f_val             = 0.0

  w = 1 + (x-1)/4
  init_term = np.sin(np.pi*w[0]) ** 2
  fin_term  = (w[n_features-1] - 1)**2 * (1 + np.sin(2*np.pi*w[n_features-1]))

  for i in range(n_features-1):
    f_val = f_val + (w[i]-1)**2 * ( 1 + 10*np.sin( np.pi*w[i] + 1)**2 )

  f_val = f_val + init_term + fin_term

  return f_val

def AlpineN1_numpy(x):
  n_features        = x.shape[0]
  f_val             = 0.0

  f_val = np.sum(np.abs(x * np.sin(x) + 0.1*x))
  # for i in range(n_features):
  #   f_val = f_val + torch.abs(x[:,i] * torch.sin(x[:,i]) + 0.1*x[:,i])
  return f_val

def Ackley_numpy(x):
  n_features        = x.shape[0]
  f_val             = 0.0

  a = 20
  b = 0.2
  c = 2*np.pi

  cos_sum_term = (1/n_features) * np.sum( np.cos(c*x))
  quad_sum_term = (1/n_features) * np.sum( x*x)

  f_val = -a * np.exp( -b * np.sqrt( quad_sum_term) ) - np.exp(cos_sum_term) + a + np.exp(1)

  return f_val