# python train.py   train.gs_epochs=30000   train.no_densify=True   gs.dataset.source_path=/home/moog-2/Downloads/360_v2/stump   gs.dataset.model_path=./outputs/test   init_wC.matches_per_ref=10000   init_wC.nns_per_ref=3 init_wC.num_refs=500 gs.vgs.is_probabilistic=True gs.vgs.vanilla=False gs.vgs.num_models=5 gs.vgs.top_K=4


import subprocess
from itertools import product
num_matches = [5000,2000,1000]
num_nns = [3]
num_refs = 1
train_densify = [True]
prob = [ True, False]
n_models = [2,5,10]
top_K = [2,4,5]

exp_list = product(num_matches, num_nns, train_densify, prob, n_models, top_K)

# fin_exp_list = []
fin_exp_list = [(0,0,False,False,True,0,0)]
for exp in exp_list:
    nm, nn, td, pr, nmodel, k = exp
    if pr and k > nmodel:
        continue

    exp_t = (nm, nn, td, pr, False, nmodel, k)
    fin_exp_list.append(exp_t)
    
    
        
exp_list = fin_exp_list
print(f"Total experiments to run: {len(exp_list)}")

for (nm, nn, td, pr, va, nmodel, k) in exp_list:
    # if pr and k > nmodel:
    #     continue
    # if va and pr:
    #     continue
    # if td and not va:
    #     continue
    cmd = f"python train.py   train.gs_epochs=2000   train.no_densify={td}   gs.dataset.source_path=/home/christoa/Downloads/360_v2/stump   gs.dataset.model_path=./outputs/exp_nm{nm}_nn{nn}_td{td}_pr{pr}_va{va}_nmodel{nmodel}_k{k}   init_wC.matches_per_ref={nm}   init_wC.nns_per_ref={nn} init_wC.num_refs={num_refs} gs.vgs.is_probabilistic={pr} gs.vgs.vanilla={va} gs.vgs.num_models={nmodel} gs.vgs.top_K={k}"
    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True)

# for nm in num_matches:
#     for nn in num_nns:
#         for pr in prob:
#             for nmodel in n_models:
#                 for k in top_K:
#                     if pr and k > nmodel:
#                         continue
#                     cmd = f"python train.py   train.gs_epochs=16000   train.no_densify=True   gs.dataset.source_path=/home/moog-2/Downloads/360_v2/stump   gs.dataset.model_path=./outputs/test_nm{nm}_nn{nn}_pr{pr}_nmodel{nmodel}_k{k}   init_wC.matches_per_ref={nm}   init_wC.nns_per_ref={nn} init_wC.num_refs={num_refs} gs.vgs.is_probabilistic={pr} gs.vgs.num_models={nmodel} gs.vgs.top_K={k}"
#                     print(f"Running command: {cmd}")
#                     subprocess.run(cmd, shell=True)