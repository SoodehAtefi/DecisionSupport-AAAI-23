from random import Random
import numpy as np

DATATYPE_BOOL = np.intc
DATATYPE_INTEGER = np.intc
DATATYPE_REAL = np.double


# calculates the empirical probability of each technique given (1) techniques that are known to have been employed, (2) techniques that are known not to have been employed, (3) prior incidents in the form of a (number of other incidents) by (number of technique matrix) matrix with 1s and 0s
def kNN(employed: list[int], not_employed: list[int], other_incidents: np.ndarray, k:int):
  # calculate difference between every prior incident and current incident
  employed_vector = np.zeros(other_incidents.shape[1], dtype=DATATYPE_BOOL)
  employed_vector[employed] = 1 # 1 for each technique that is known to have been used
  difference = np.bitwise_xor(other_incidents, employed_vector, dtype=DATATYPE_BOOL) # matrix with 1s where prior incident differs
  # consider only techniques that are either known to have been employed or known not to have been employed
  mask_vector = employed_vector # first part of the mask
  mask_vector[not_employed] = 1 # second part of the mask
  masked_difference = np.multiply(difference, mask_vector, dtype=DATATYPE_BOOL) # matrix with 1s only where prior incident differs from what is known
  distance = np.sum(masked_difference, axis=1, dtype=DATATYPE_INTEGER) # sum up 1s for each prior incident to get distance from current incident
  # select the k nearest prior incidents
  sorted_indices = distance.argsort() # order the indices of prior incidents by their distance
  k_nearest_indices = sorted_indices[:k] # select first k indices
  k_nearest = other_incidents[k_nearest_indices] # select k nearest prior incidents
  # calculate mean of each column to get the empirical probability of each technique
  probabilities = np.mean(k_nearest, axis=0, dtype=DATATYPE_REAL)
  return probabilities # one probability value for each technique


class Environment:
  def __init__(self, benefits: list[float], costs: list[float],d,cat):
    self.benefits = np.array(benefits, dtype=DATATYPE_REAL)
    self.benefits.setflags(write=False)
    self.costs = np.array(costs, dtype=DATATYPE_REAL)
    self.costs.setflags(write=False)
    assert(self.benefits.shape == self.costs.shape)
    self.d=d
    self.cat=cat
    
class Policy:
  def __init__(self, environment: Environment, other_incidents: np.ndarray,parameters: dict[str, float]):
    self.environment = environment
    self.other_incidents = other_incidents
    if 'random_seed' in parameters:
      self.random = Random(parameters['random_seed'])
    else:
      self.random = Random()

  def next_technique(self, employed: list[int], not_employed: list[int]):
    raise "Not implemented"


class RandomPolicy(Policy):
  def __init__(self, environment: Environment, other_incidents: np.ndarray,parameters: dict[str, float]):
    super().__init__(environment, other_incidents, parameters)
    
  def next_technique(self, employed: list[int], not_employed: list[int]):
    employed = set(employed)
    not_employed = set(not_employed)
    actions = [t for t in range(len(self.environment.benefits)) if t not in employed and t not in not_employed]
    return self.random.choice(actions)


class StaticPolicy(Policy):
  def __init__(self, environment: Environment, other_incidents: np.ndarray,parameters: dict[str, float]):
    super().__init__(environment, other_incidents,parameters)
    probabilities = np.mean(other_incidents, axis=0, dtype=DATATYPE_REAL)
    self.expectation = np.divide(np.multiply(probabilities, environment.benefits, dtype=DATATYPE_REAL), environment.costs, dtype=DATATYPE_REAL)
    
  def next_technique(self, employed: list[int], not_employed: list[int]):
    self.expectation[employed] = 0.0
    self.expectation[not_employed] = 0.0
    action = np.argmax(self.expectation)
    return action



class MyopicPolicy(Policy):
  def __init__(self, environment: Environment, other_incidents: np.ndarray,parameters: dict[str, float]):
    super().__init__(environment, other_incidents, parameters)
    self.beta1 = parameters['beta1']
    self.beta2 = parameters['beta2']
    self.distance=np.array([0 for z in range(len(other_incidents))])
    self.investigated = set()

  def next_technique(self, employed: list[int], not_employed: list[int]):
    k = round(self.beta1 + self.beta2 * (len(employed) + len(not_employed)))

    if len(employed) + len(not_employed) > 0:
      previous_action = None
      previous_employed = None
      for t in employed:
        if t not in self.investigated:
          self.investigated.add(t)
          previous_action = t
          previous_employed = True
          break
      if previous_action is None:
        for t in not_employed:
          if t not in self.investigated:
            self.investigated.add(t)
            previous_action = t
            previous_employed = False
            break

      additional_distance = self.other_incidents[:,previous_action]
      if previous_employed:
        additional_distance = np.subtract(np.ones(self.other_incidents.shape[0], dtype=DATATYPE_BOOL), additional_distance, dtype=DATATYPE_BOOL)
      self.distance = np.add(self.distance, additional_distance, dtype=DATATYPE_REAL)
   
    sorted_indices = self.distance.argsort()
    k_nearest_indices = sorted_indices[:k]
    k_nearest = self.other_incidents[k_nearest_indices]
    probabilities = np.mean(k_nearest, axis=0, dtype=DATATYPE_REAL)
    probabilities[employed] = -1.0 # uninvestigated techniques may all have 0 probability
    probabilities[not_employed] = -1.0
    expectation = np.divide(np.multiply(probabilities, self.environment.benefits, dtype=DATATYPE_REAL), self.environment.costs, dtype=DATATYPE_REAL)
    action = np.argmax(expectation)
    return action


class MCTSPolicy(Policy):
  def __init__(self, environment: Environment, other_incidents: np.ndarray,parameters: dict[str, float]):
    super().__init__(environment, other_incidents, parameters)
    self.beta1 = parameters['beta1']
    self.beta2 = parameters['beta2']
    self.gamma = parameters['gamma']
    self.K = parameters['K']
    self.D = parameters['D']
    self.M = parameters['M']
    self.probabilities = {}
    self.state_value = {} # R[Y, N]
    self.state_action_values = {} # R[Y, N, a]
    self.exploration_counts = {} # N[Y, N, a]

  def get_probabilities(self, state: tuple[tuple[int]]):
    if state in self.probabilities:
      return self.probabilities[state]
    employed = state[0]
    not_employed = state[1]
    t = len(employed) + len(not_employed)
    k = round(self.beta1 + self.beta2 * t)
    probabilities = kNN(list(employed),  list(not_employed), self.other_incidents, k)
    self.probabilities[state] = probabilities
    return probabilities
    
  def exploration_decision(self, state: tuple[tuple[int]], uninvestigated: list[int]):


    benefits = self.environment.benefits
    costs = self.environment.costs
    employed = list(state[0])
    not_employed  = list(state[1])
    probabilities = self.get_probabilities(state)
    probabilities[employed] = -1.0
    probabilities[not_employed] = -1.0
    expectation = np.divide(np.multiply(probabilities, benefits, dtype=DATATYPE_REAL), costs, dtype=DATATYPE_REAL)
    sorted_indices = np.flip(expectation.argsort())
    myopic_choices=sorted_indices[:3]
    idxes_myopic_choices=np.argwhere(np.isin(list(uninvestigated), list(myopic_choices))).ravel()

    if state not in self.state_action_values:
      self.state_action_values[state] = np.zeros(len(uninvestigated), dtype=DATATYPE_REAL)
      assert(state not in self.exploration_counts)
      self.exploration_counts[state] = np.ones(len(uninvestigated), dtype=DATATYPE_INTEGER)

    log_total_exploration = np.log(np.sum(self.exploration_counts[state], dtype=DATATYPE_REAL), dtype=DATATYPE_REAL)
    exploration_factor = np.multiply(np.sqrt(np.divide(log_total_exploration, self.exploration_counts[state], dtype=DATATYPE_REAL), dtype=DATATYPE_REAL), self.M, dtype=DATATYPE_REAL)

    preference = np.add(self.state_action_values[state], exploration_factor, dtype=DATATYPE_REAL)
    return idxes_myopic_choices[np.argmax(preference[idxes_myopic_choices])] #search over Myopic choices and pick the maximum

  def get_state_value(self, state: tuple[tuple[int]]):
    if state in self.state_value:
      return self.state_value[state]
    
    base_estimate = 0
    probabilities = self.get_probabilities(state)
    employed = state[0]
    not_employed = state[1]
    probabilities[list(employed)] = 0.0
    probabilities[list(not_employed)] = 0.0
    
    
    expectation = np.divide(np.multiply(probabilities, self.environment.benefits, dtype=DATATYPE_REAL), self.environment.costs, dtype=DATATYPE_REAL)

    #add base estimation according to the formula
    remaining_actions_expectations=expectation[np.nonzero(expectation)]
    remaining_actions_inx=np.arange(len(remaining_actions_expectations))
    multiply_by_gamma=np.array([[self.gamma**j for i in range(remaining_actions_expectations.shape[0])] for j in range(remaining_actions_expectations.shape[0]-1)])
    estimation_each_action=np.divide(np.multiply(remaining_actions_expectations,multiply_by_gamma,dtype=DATATYPE_REAL),remaining_actions_expectations.shape[0],dtype=DATATYPE_REAL)
    estimation=np.sum(np.sum(estimation_each_action, axis = 0,dtype=DATATYPE_REAL))
    base_estimate = estimation
    self.state_value[state] = base_estimate
    return base_estimate

  def next_technique(self, employed: list[int], not_employed: list[int]):
    # root state
    root_employed = tuple(sorted(employed))
    root_not_employed = tuple(sorted(not_employed))
    root_uninvestigated = [t for t in range(len(self.environment.benefits)) if t not in root_employed and t not in root_not_employed]

    for k in range(self.K):

      trajectory = []
      employed = root_employed
      not_employed = root_not_employed
      uninvestigated = list(root_uninvestigated) # create copy
    
      while uninvestigated and len(trajectory) < self.D:
      
        state = (employed, not_employed)
        action = self.exploration_decision(state, uninvestigated)
        self.exploration_counts[state][action] += 1
        trajectory.append((state, tuple(uninvestigated), action))
        if self.random.random() < 0.5:
          employed = tuple(sorted(employed + (uninvestigated[action],)))
        else:
          not_employed = tuple(sorted(not_employed + (uninvestigated[action],)))
        del uninvestigated[action]
      for (state, uninvestigated, action) in reversed(trajectory):
        technique = uninvestigated[action]
        probability = self.get_probabilities(state)[technique]
        cost = self.environment.costs[technique]
        benefit = self.environment.benefits[technique]
        employed = state[0]
        not_employed = state[1]
        next_state_employed = (tuple(sorted(employed + (technique,))), not_employed)
        next_state_not_employed = (employed, tuple(sorted(not_employed + (technique,))))
        self.state_action_values[state][action] =   probability * (benefit / cost + self.gamma * self.get_state_value(next_state_employed)) \
                                                  + (1.0 - probability) * self.gamma * self.get_state_value(next_state_not_employed)

        self.state_value[state] = np.max(self.state_action_values[state])

    action = np.argmax(self.state_action_values[(root_employed, root_not_employed)])
    return root_uninvestigated[action]

      
class TwoStepExhaustivePolicy(Policy):#look two steps ahead
  def __init__(self, environment: Environment, other_incidents: np.ndarray,parameters: dict[str, float]):
    super().__init__(environment, other_incidents,parameters)
    self.beta1 = parameters['beta1']
    self.beta2 = parameters['beta2']
    self.alpha = parameters['alpha']
    self.gamma = parameters['gamma']

    
  def next_technique(self, employed: list[int], not_employed: list[int]):
    if len(employed) + len(not_employed) == len(self.environment.benefits) - 1:
      for t in range(len(self.environment.benefits)):
        if t not in employed and t not in not_employed:
          return t

    benefits = self.environment.benefits
    costs = self.environment.costs
    k = round(self.beta1 + self.beta2 * (len(employed) + len(not_employed)))
   
    root_probabilities = kNN(employed, not_employed, self.other_incidents, k)
    root_probabilities[employed] = -1.0
    root_probabilities[not_employed] = -1.0
    root_expectation = np.divide(np.multiply(root_probabilities, benefits, dtype=DATATYPE_REAL), costs, dtype=DATATYPE_REAL)
    sorted_indices = np.flip(root_expectation.argsort())
    employed = set(employed)
    not_employed = set(not_employed)
    slopes = np.full(len(benefits), -1.0, dtype=DATATYPE_REAL)
    for root_action in sorted_indices[:3]: #uninvestigated:
      # find optimal sub action given that root action was employed
      employed.add(root_action)
      sub_emp_probs = kNN(list(employed), list(not_employed), self.other_incidents, k,self.other_incidents_features_Y,self.other_incidents_features_N,self.environment.techniques_features,self.alpha)
      sub_emp_probs[list(employed)] = -1.0
      sub_emp_probs[list(not_employed)] = -1.0
      sub_emp_exps = np.divide(np.multiply(sub_emp_probs, benefits, dtype=DATATYPE_REAL), costs, dtype=DATATYPE_REAL)
      sub_emp_action = np.argmax(sub_emp_exps)
      employed.remove(root_action)
      # find optimal sub action given that root action was not employed
      not_employed.add(root_action)
      sub_nemp_probs = kNN(list(employed), list(not_employed), self.other_incidents, k,self.other_incidents_features_Y,self.other_incidents_features_N,self.environment.techniques_features,self.alpha)
      sub_nemp_probs[list(employed)] = -1.0
      sub_nemp_probs[list(not_employed)] = -1.0
      sub_nemp_exps = np.divide(np.multiply(sub_nemp_probs, benefits, dtype=DATATYPE_REAL), costs, dtype=DATATYPE_REAL)
      sub_nemp_action = np.argmax(sub_nemp_exps)
      not_employed.remove(root_action)
      # calculate expected slope
      root_pr = root_probabilities[root_action]
      root_slope = root_pr * benefits[root_action] / costs[root_action]
      sub_emp_pr = sub_emp_probs[sub_emp_action]
      sub_emp_slope = sub_emp_pr * benefits[sub_emp_action]  / costs[sub_emp_action]
      sub_nemp_pr = sub_nemp_probs[sub_nemp_action]
      sub_nemp_slope = sub_nemp_pr * benefits[sub_nemp_action]  / costs[sub_nemp_action]
      factor=self.gamma
      expected_slope = root_slope + factor * (root_pr * sub_emp_slope + (1 - root_pr) * sub_nemp_slope)
      slopes[root_action] = expected_slope
    action = np.argmax(slopes)
    return action
