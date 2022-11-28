import logging
import numpy as np
from random import Random
MULTIPROCESSING = True
import pandas as pd
import pickle
if MULTIPROCESSING:
  from multiprocessing import Pool,cpu_count

from policies import Environment, Policy, DATATYPE_BOOL,DATATYPE_REAL


MAX_NUM_INVESTIGATIONS = 5000
INITIAL_TECHNIQUE = True

def _evaluate_policy(investigation: dict):

  environment = investigation['environment']
  incidents = investigation['incidents']
  policy_class = investigation['policy_class']
  policy_parameters = investigation['policy_parameters']
  INVESTIGATION_BUDGET=investigation['INVESTIGATION_BUDGET']
  i = investigation['i']
  logging.info(f"Starting investigation {i}")
  # select incident
  incident = incidents[i]
  #remove current incident from the dataset (leave-one-out cross-validation)
  other_incidents = np.delete(incidents, i, axis=0)
  # initialize policy
  policy_parameters['random_seed'] = i
  policy = policy_class(environment, other_incidents, policy_parameters)
  # initial state
  employed = []
  not_employed = []
  #choose first action randomly
  if INITIAL_TECHNIQUE:
        initial_choices = [t for t in range(incidents.shape[1]) if incident[t]]
        assert(len(initial_choices) > 1)
        initial_technique = Random(i).choice(initial_choices)
        employed = [initial_technique]
        
  AUC = 0
  sum_benefit = 0.0
  sum_cost = 0.0
  benefit_per_cost=np.array([0 for z in range(INVESTIGATION_BUDGET+1)])

  
  # statistics

  for t in range(incidents.shape[1]-1):
    technique = policy.next_technique(employed, not_employed)
    assert(technique not in employed)
    assert(technique not in not_employed)
    cost = environment.costs[technique]
    if incident[technique]:
      employed.append(technique)
#      print('employed',technique)
      benefit = environment.benefits[technique]
    else:
      not_employed.append(technique)
#      print('not_employed',technique)
      benefit = 0.0
    if sum_cost+ cost> INVESTIGATION_BUDGET:
        AUC+=(INVESTIGATION_BUDGET- sum_cost) * sum_benefit
        benefit_per_cost[int(sum_cost+(INVESTIGATION_BUDGET- sum_cost))]=sum_benefit
        break
    AUC += cost * sum_benefit
    sum_cost += cost
    benefit_per_cost[int(sum_cost)]=sum_benefit
    sum_benefit += benefit

  #fill zeros
  previous = np.arange(len(benefit_per_cost))
  previous[benefit_per_cost == 0] = 0
  previous = np.maximum.accumulate(previous)
  benefit_per_cost=benefit_per_cost[previous]

  return AUC,benefit_per_cost

def evaluate_policy(environment: Environment, incidents: list[list[bool]], policy_class: type, policy_parameters: dict[str, float],INVESTIGATION_BUDGET:int):
  incidents = np.array(incidents, dtype=DATATYPE_BOOL)
  incidents.setflags(write=False)
  assert(incidents.shape[1] == environment.benefits.shape[0])
  assert(issubclass(policy_class, Policy))
  AUC_values = []
  benefit_all_incidents = []
  num_investigations = min(incidents.shape[0], MAX_NUM_INVESTIGATIONS)
  investigations = [{'environment': environment, 'incidents': incidents,'policy_class': policy_class, 'policy_parameters': policy_parameters, 'INVESTIGATION_BUDGET':INVESTIGATION_BUDGET,'i': i} for i in range(num_investigations)]

  if MULTIPROCESSING:
    with Pool(processes=cpu_count()) as pool:
     AUC_values,benefit_all_incidents=zip(*pool.map(_evaluate_policy, investigations))
  else:
     AUC_values= [_evaluate_policy(investigation) for investigation in investigations]
  AUC_mean = np.mean(AUC_values)


#  avg_benefit_obtained=np.mean(benefit_all_incidents, axis = 0)
  print(f"Policy: {policy_class} / AUCBE: {AUC_mean}")
  return AUC_mean,benefit_all_incidents


