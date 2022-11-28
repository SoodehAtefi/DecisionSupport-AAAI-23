import numpy as np

from policies import Environment, Policy, DATATYPE_REAL, DATATYPE_BOOL

class DISCLOSEPolicy(Policy):
  def __init__(self, environment: Environment, other_incidents: np.ndarray,parameters: dict[str, float]):
    super().__init__(environment, other_incidents,parameters)
    self.calculate_probabilities()
    self.gamma = np.zeros(len(environment.benefits), dtype=DATATYPE_REAL)
    self.conf= np.zeros(len(environment.benefits), dtype=DATATYPE_REAL)
    self.investigated = set()

  def update_gamma(self, employed: list[int], not_employed: list[int], previous_action: int):
    assert(previous_action in employed)
   
    if previous_action==employed[0]:
      
        self.conf= [self.environment.d[self.environment.cat[previous_action][0]][self.environment.cat[i][0]] for i in range(0, len(self.gamma))]
        self.gamma = self.T[previous_action]
       
    
    for j in range(len(self.gamma)):
      if (j in employed) or (j in not_employed):
        self.gamma[j] = -1 # never choose this technique
#      else:
#        self.gamma[j] = max(self.gamma[j], self.T[previous_action,j])
#
      elif any(self.environment.d[x][self.environment.cat[j][0]] < self.conf[j] for x in self.environment.cat[previous_action]):
        
        self.gamma[j] =  self.T[previous_action,j]
        self.conf[j] = min(self.environment.d[x][self.environment.cat[j][0]] for x in self.environment.cat[previous_action])

      elif any(self.environment.d[x][self.environment.cat[j][0]] == self.conf[j] for x in self.environment.cat[previous_action]):
      
        self.gamma[j] = max(self.gamma[j], self.T[previous_action,j])
      else:
        pass
     
        
  def calculate_probabilities(self):
    num_techniques = len(self.environment.benefits)
    self.T = np.zeros((num_techniques, num_techniques), dtype=DATATYPE_REAL)
    for i in range(num_techniques):
      occurrences_i = self.other_incidents[:,i]
      num_occurrences_i = np.sum(occurrences_i)
      for j in range(num_techniques):
        occurrences_j = self.other_incidents[:,j]
        intersection = np.multiply(occurrences_i, occurrences_j, dtype=DATATYPE_BOOL)
        cond_prob = np.sum(intersection) / num_occurrences_i
        self.T[i,j] = cond_prob
  
  def selection(self):
    expectation = np.divide(np.multiply(self.gamma, self.environment.benefits, dtype=DATATYPE_REAL), self.environment.costs, dtype=DATATYPE_REAL)
    action = np.argmax(expectation)
    return action
    
  def next_technique(self, employed: list[int], not_employed: list[int]):
    
    if employed or not_employed:
      # find previous action
      
      previous_action = None
      for t in employed + not_employed:
        if t not in self.investigated:
          previous_action = t
          break
      self.investigated.add(previous_action)
      # update gamma (if previous action was employed)
      if previous_action in employed:
        
        self.update_gamma(employed, not_employed, previous_action)
      else:
        self.gamma[previous_action]=-1

      action = self.selection()
      self.gamma[action] = -1 # never choose this techniques
      return action
    else:
      # myopic selection of first action
      probabilities = np.mean(self.other_incidents, axis=0, dtype=DATATYPE_REAL)
      expectation = np.divide(np.multiply(probabilities, self.environment.benefits, dtype=DATATYPE_REAL), self.environment.costs, dtype=DATATYPE_REAL)
      action = np.argmax(expectation)
      self.gamma[action] = -1
      return action

