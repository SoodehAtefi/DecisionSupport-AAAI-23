# Principled Data-Driven Decision Support for Cyber-Forensic Investigations
This project contains the source code and dataset for the paper 'Principled Data-Driven Decision Support for Cyber-Forensic Investigations' accepted in the 37th AAAI Conference on Artificial Intelligence (AAAI 2023).

## Structure
```
.
├── main.py              #  file for running different policies (produces results of the evaluation of each approach which is reported in figures 1 to 4)
├── evaluation.py        #  evaluation code for all of the incidents
├── policies.py          #  algorithms implementations 
├── DISCLOSE.py          #  DISCLOSE algorithm
├── v6                   #  contains dataset,files required for runnig evaluations based on this dataset, and results of optimizations using dataset version 6.3 
├── v10                  #  contains dataset,files required for runnig evaluations based on this dataset, and results of optimizations using dataset version 10.1
├── v11                  #  contains dataset,files required for runnig evaluations based on this dataset, and results of optimizations using dataset version 11.3
├── optimal_parameters   #  file contains optimal parameters for different datasets and different budget limitations for all approaches (MCTS, Static, and DISCLOSE)
parameter_search_results #  results of hyper-parameter optimization for Myopic and MCTS approaches for each dataset (each dataset has a 'parameter_search_results' folder)
plots                    #  results of the evaluation of each approach which is reported in figure1 to 4 of the paper (each dataset has a 'plots' folder)
```                            
## Datasets
- Experiments performed on three different versions of the CTI dataset (V6.3 and V10.1, and V11.3 (latest)). 
- For v6 we use 31 techniques and for v10.1, and v11.3 we use 29 techniques.
- Datasets are list of binary values for each incident. Employed techniques in an incident(from sets of 29 or 31) are 1's, and the rests are 0's.

## Parameter Search

- We performed hyper-parameter search for each budget limitation and each dataset seperately. For replicating results with different datasets and different budgets, the path to the dataset, optimal hyper-parameters ('policy_parameters') using optimal_parameters file, and invetsigation budget ('INVESTIGATION_BUDGET') in the code should be changed to achieve the required results.
- In order to do the hyper-parameter search for each optimization (Myopic or MCTS), it can be uncommented in the main.py
- Note that for running experiments with no budget limitation, we put 183 (sum of benefit of all 31 techniques for v6.3) and 171 (sum of benefit of all 31 techniques for v6.3) as investigation budget. Since an incident investigation starts with a technique (we do not consider the benefit of the first action), maximum cost/budget of the investigation never reaches the numbers above.
## Python version: 3.9.10
