from statistics import NormalDist
from typing import Callable
from scipy.stats import norm

import numpy as np

def genereate_pdf_function(x:list):
  gen = DensityGenerator(x)
  return gen.generate_density_func()
  
class DensityGenerator:
  """
  creates a density from given array of values, x using the Chinese Restaurant process to
  implement a Dirichlet Process Mixture Model (DPMM).
  All of x are initially assigned to same cluster, then n_max_rounds of iteration over
  all points are performed to reposition the points to either the highest probability cluster
  or to a new cluster.
  
  see also  the predictive distribution of
  the Dirichlet Process as a Bayesian model
  (eqn 10, Teh, Y.W., 2017. Dirichlet process. In Encyclopedia of machine learning and data mining (pp. 361-370). Springer, Boston, MA.
  Teh YW. Dirichlet process. InEncyclopedia of machine learning and data mining 2017 (pp. 361-370). Springer, Boston, MA.)
  """
  def __init__(self, x:list, n_max_rounds:int=10):
    self.x = np.array(x)
    self.unit_std_norm = NormalDist(mu=0, sigma=1)
    self.n_max_rounds = n_max_rounds
    self.x.sort()
    max_diff = self.x[-1] - self.x[0]
    #prior_G0 is the base distribution.  it's a prior.
    self.prior_G0 = NormalDist(mu=0, sigma=0.8*max_diff)
    #alapha is concentration parameter governing how likely to start a new cluster
    self.alpha = 1.
    self.sumx = []     # list of scalars of sums of items
    self.n = []        # list of scalars
    #cluster assignments for each index of x:
    self.assign = [-1 for _ in range(len(self.x))] #1-D array
    self.members = [] #list of sets of cluster memberships.  first set is for cluster 0, ...

    #put all in same group
    self.sumx.append(np.sum(self.x))
    self.n.append(len(self.x))
    for i in range(len(self.x)):
       self.assign[i] = 0
    self.members.append(set([i for i in range(len(self.x))]))
    
    self.built = False

  def _remove_from_cluster(self, point_idx, cluster_idx):
    self.n[cluster_idx] -= 1
    self.sumx[cluster_idx] -= self.x[point_idx]
    self.assign[point_idx] = -1
    self.members[cluster_idx].remove(point_idx) 

  def _add_to_cluster(self, point_idx, cluster_idx):
    x = self.x[point_idx]
    if cluster_idx == len(self.n):
      #new cluster
      self.n.append(1)
      self.sumx.append(x)
      self.members.append(set([point_idx]))
    else:
      self.n[cluster_idx] += 1
      self.sumx[cluster_idx] += x
      self.members[cluster_idx].add(point_idx)
    self.assign[point_idx] = cluster_idx
    
  def _calc_stdev(self,cluster_idx) -> np.float64:
    """
    calculate the standard deviation for the cluster
    :param cluster_idx: cluster index, that is, the index of self.members array
    :return: the standard deviation
    """
    members = [self.x[i] for i in self.members[cluster_idx]]
    return np.std(members) + 1E-9

  def _reposition(self, point_idx) -> None:
    """
    repositions a point using Gibb's sampling via Chinese Restaurant Process.

    takes point_idx out of its current cluster, calculates the probabilities that it belongs to
    each cluster, and calculates a probability for it being in its own new cluster.
    the largest probability decides which cluster or new cluster to assign point_idx to.
    :param point_idx: the index in self.x and self.assign for the point.
    """
    cluster_idx = self.assign[point_idx]
    self._remove_from_cluster(point_idx, cluster_idx)
    k_clusters = len(self.n)
    x = self.x[point_idx]
    max_p_cluster = float('-inf')
    max_p_cluster_idx = -1
    for k in range(k_clusters):
      cluster_mean = self.sumx[k] / self.n[k]
      cluster_sigma = self._calc_stdev(k)
      z = -1.*np.abs((x - cluster_mean)/cluster_sigma)
      #e.g. signma=1.5, lookup left side and double it for percent that it belongs to distr
      p_i = 2.*self.unit_std_norm.cdf(z)
      print(f'point:{x} cluster:{k} {[self.x[i].item() for i in self.members[k]]}, mu={cluster_mean}, sigma={cluster_sigma}')
      print(f'  z={z}, p[{k}]={p_i}')
      if p_i > max_p_cluster:
        max_p_cluster = p_i
        max_p_cluster_idx = k
    #calc probabllity of new cluster:
    z = -1.*np.abs((x - self.prior_G0.mean)/self.prior_G0.stdev)
    p_new = self.alpha*(self.prior_G0.cdf(z))
    print(f'pnew={p_new}')
    if p_new > max_p_cluster:
      self._add_to_cluster(point_idx, k_clusters)
    else:
      self._add_to_cluster(point_idx, max_p_cluster_idx)
 

  def _determine_clusters(self) -> None:
    """
    invoked only once after construction to form clusters from the given data.
    invokes n_max_rounds of assignments over all points:
    the array x given to the constructor is reordered by largest distances from nearest cluster.
    then for each point, it's removed from its cluster and assigned to the cluster it most
    likely belongs to or to a new cluster.
    """
    if self.built:
      return
    #TODO: could optionally offer the simpler Ewan's sampling.   draw random cluster.
    # p_new_cluster = (alpha/(alpha+n)), p_stay_in_current=(n/(alpha+n))
    # but the results are not as good.  e.g. for x=[3,5,6,12], unless we randomly choose '12' first and then p_new with change 50%,
    # the resulting distr likely will not be best representation
    for _ in range(self.n_max_rounds):
      #rearrange point order to visit those with largest stdev from their cluster means
      reordered = np.arange(len(self.x))
      #NOTE: _sort is O(len(indices) * len(self.sumx)) so can make a faster random selection at expense of accuracy if needed.
      reordered = self._sort(reordered)
      for point_idx in reordered:
        if self.n[self.assign[point_idx]] <= 2:
          continue
        self._reposition(point_idx)
      #TODO: add early exit for no changes in distr
    self.built = True
    
  def generate_density_func(self) -> Callable:
    '''
    returns a method to generate probability densities.
    :return: pdf function, f(X)
    '''
    self._determine_clusters()
    def prob_density(X):
      """
      given array X, return probability densities for those points.
      :param X: an array of input data
      :return: an array of probabilitiy densities
      """
      k_clusters = len(self.n)
      weights = np.array([w for w in self.n])
      weights = weights / np.sum(weights)
      f_y = 0.
      for k in range(k_clusters):
        mu_k = self.sumx[k] / self.n[k]
        sigma_k = self._calc_stdev(k)
        z = -1.*np.abs((X - mu_k) / sigma_k)
        p_k = 2*norm.cdf(z, loc=0., scale=1.0)
        f_y += weights[k] * p_k
      return f_y
    return prob_density
  
  def _sort(self, indices):
    """
    descending sort of the point indices by the closest distance of their points to a cluster.
    :param indices: array of point indices.
    Runtime comolexity is O(len(indices) * len(self.sumx))
    :return: indices reoreded by decreasing distance to nearest cluster
    """
    cluster_means = [self.sumx[i]/self.n[i] for i in range(len(self.n))]
    #for each index, find minimum distance from a cluster and store that,
    # then sort the indices by the largest distances
    d = []
    for i, point_idx in enumerate(indices):
      d.append(float('inf'))
      for cluster_idx in range(len(self.n)):
        di = np.abs(self.x[point_idx] - cluster_means[cluster_idx])
        if di < d[i]:
          d[i] = di
    return indices[np.argsort(-np.array(d))]
  