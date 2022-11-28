import logging
import pickle
import csv
import numpy as np
from policies import Environment, RandomPolicy, StaticPolicy, MyopicPolicy, MCTSPolicy, TwoStepExhaustivePolicy
from DISCLOSE import DISCLOSEPolicy
from evaluation import evaluate_policy
# modules for testing
from random import seed, randint, random
from timeit import timeit
import pandas as pd
from hyperopt import fmin, tpe, Trials, hp,STATUS_OK

logging.basicConfig(level=logging.WARN)
seed(0)

path_to_dataset="./v6" # can be changed

with open(path_to_dataset+"/"+"incidents.pkl", 'rb') as fin:
  incidents = pickle.load(fin)

cat = {}
with open(path_to_dataset+"/"+"categories.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        cat[int(row[0])] = row[1:]
for x in cat:
    cat[x] = [i for i in cat[x] if i != '']
# read trust values per category pair
d={}
with open(path_to_dataset+"/"+"trust.csv") as csvDataFile:
    data1=csv.reader(csvDataFile, delimiter=',')
    headers=next(data1)[1:]
    for row in data1:
        d[row[0]]={key: int(value) for key, value in zip(headers,row[1:])}
        
benefits = {}
costs = {}
with open(path_to_dataset+"/"+"Benefit_Cost.csv", 'rt') as fin:
  fin.readline() # skip header
  for line in fin:
    fields = line.strip().split(",")
    index = int(fields[0])
    benefits[index] = fields[2]
    costs[index] = fields[3]
benefits = [benefits[t] for t in range(len(benefits))]
costs = [costs[t] for t in range(len(costs))]

environment = Environment(benefits, costs,d,cat)

policy_parameters = { #can be changed for each investigation budget and each dataset
    'beta1':40.0,
    'beta2': 1.5,
    'gamma':0.0,
    'K': 500, # number of MCTS iterations
     'D':2.0, #depth of the tree-search
    'M': 29.0, # focus exploration (exploration factor)
}
INVESTIGATION_BUDGET =45


def MCTS_parameters_search(policy_parameters):
    AUC_result,_=evaluate_policy(environment, incidents,policy_class, policy_parameters,INVESTIGATION_BUDGET)
    score = 1-AUC_result
    file_results = open(f_out, 'a')
    writer = csv.writer(file_results)
    writer.writerow([score,policy_parameters['D'], policy_parameters['M'],policy_parameters['gamma']])
    file_results.close()
    return {'loss': score, 'policy_parameters': policy_parameters, 'status': STATUS_OK}
    
if __name__ == '__main__':
        
    df_benefits=pd.DataFrame() # avg benefit attained to the cost (e.g., for figures 1 to 4)
    for policy_class in [DISCLOSEPolicy,StaticPolicy,MCTSPolicy]:

            AUC_result,benefit_all_incidents =evaluate_policy(environment, incidents,policy_class, policy_parameters,INVESTIGATION_BUDGET)
            avg_benefit_obtained=np.mean(benefit_all_incidents, axis = 0) #average benefit obtain over all incidents at each budget
            df_benefits['Budget']=np.arange(0,INVESTIGATION_BUDGET+1)
            name_policy=str(policy_class).split('.')[1].strip(".>'")
            df_benefits[name_policy+'_'+'Benefit'+'_'+str(INVESTIGATION_BUDGET)]=avg_benefit_obtained
            for quantile in [.25,.50,.75]:
                df_benefits[name_policy+'_'+str(INVESTIGATION_BUDGET)+'_'+str(quantile)]= np.quantile(benefit_all_incidents, quantile,axis = 0)
    df_benefits.to_csv(path_to_dataset+ '/plots/'+'budget'+'_'+str(INVESTIGATION_BUDGET)+'.csv',index=False)

##################exhaustive hyper-parameter search for Beta1 and Beta2 parameters (KNN algorithm)
#    policy_class = MyopicPolicy
#    beta1s=[]
#    beta2s = []
#    AUC_results = []
#    for beta1 in np.arange(1,131):
#        for beta2 in np.arange(0.0,6.1,0.1):
#                policy_parameters['beta1']=beta1
#                policy_parameters['beta2']=beta2
#                AUC_result,_=evaluate_policy(environment, incidents,policy_class, policy_parameters,INVESTIGATION_BUDGET)
#                AUC_results.append(AUC_result)
#                beta1s.append(beta1)
#                beta2s.append(beta2)
#    df=pd.DataFrame()
#    df['AUC']=AUC_results
#    df['Beta1']=beta1s
#    df['Beta2']=beta2s
#    df.to_csv(path_to_dataset+'/parameter_search_results/parameter_search_budget'+'_'+str(INVESTIGATION_BUDGET)+'.csv',index=False)
#    print(df.loc[df['AUC'].idxmax()].reset_index(name = 'Max_Params'))
     #use beta1, and beta2 that provides maximum AUCBE to run the optimization for MCTS below
     
#####################hyper-parameter search for M (exploration factor), D (depth of the tree-search), and gamma (discount factor)
#    policy_class = MCTSPolicy
#    params = {
#    'beta1': 40.0,
#    'beta2': 1.5,
#    'K': 500, # number of MCTS iterations
#    'D':hp.quniform('D', 2, 5, 1), #depth of the tree-search
#    'M':hp.quniform('M',1,100,1), # focus exploration (exploration factor)
#    'gamma':hp.quniform('gamma', 0.0, 1.0, 0.01) #discount factor
#    }
#    tpe_algorithm = tpe.suggest
#    num_eval = 500
#    trials = Trials()
#    f_out = path_to_dataset+ '/parameter_search_results/MCTS_parameter_search'+'_'+str(INVESTIGATION_BUDGET)+'.csv'
#    final_res = open(f_out, 'w')
#    writer = csv.writer(final_res)
#    best_param = fmin(MCTS_parameters_search, params, algo=tpe.suggest, max_evals=num_eval, trials=trials)
#    final_res.close()
#

