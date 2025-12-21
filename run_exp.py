# python train.py   train.gs_epochs=30000   train.no_densify=True   gs.dataset.source_path=/home/moog-2/Downloads/360_v2/stump   gs.dataset.model_path=./outputs/test   init_wC.matches_per_ref=10000   init_wC.nns_per_ref=3 init_wC.num_refs=500 gs.vgs.is_probabilistic=True gs.vgs.vanilla=False gs.vgs.num_models=5 gs.vgs.top_K=4


import subprocess
from itertools import product
import os
# models = bicycle  bonsai  counter  flowers.txt  garden  kitchen  room  stump



class ExpParams:
    def __init__(self, num_matches, num_nns, num_refs, train_densify, is_edgs, is_vgs, is_vanilla, n_models, top_K, model_name):
        self.num_matches      = num_matches
        self.num_nns          = num_nns
        self.num_refs         = num_refs
        self.train_nodensify    = train_densify
        self.is_edgs          = is_edgs
        self.is_vgs           = is_vgs
        self.is_vanilla = is_vanilla
        self.n_models         = n_models
        self.top_K            = top_K
        self.model_name       = model_name

    def __repr__(self) -> str:
        return f"ExpParams(num_matches={self.num_matches}, num_nns={self.num_nns}, num_refs={self.num_refs}, train_nodensify={self.train_nodensify}, is_edgs={self.is_edgs}, is_vgs={self.is_vgs}, is_vanilla={self.is_vanilla}, n_models={self.n_models}, top_K={self.top_K}, model_name={self.model_name})"

models        = ["bicycle"]
num_matches   = [2000,1000]
num_nns       = [2]
num_refs      = [500]       # if you want this in the config tuple
train_densify = [True, False]
prob_vals     = [False]
n_models_vals = [2]
top_K_vals    = [2]


# fin_exp_list = []
vanilla_config = (0,0,False,False,False,0,0)
fin_exp_list = []

for m in models:
    nm, nn, nr, td, p, n_m, k = vanilla_config
    is_edgs = False
    is_vgs  = False
    is_vanilla = True
    t = ExpParams(num_matches=nm, num_nns=nn, num_refs=nr, train_densify=td, is_edgs=is_edgs, is_vgs=is_vgs, is_vanilla=is_vanilla, n_models=n_m, top_K=k, model_name=m)
    fin_exp_list.append(t)

for i in [(True,True), (True, False), (False, True) ]:
    is_edgs = i[0]
    is_vgs  = i[1]
    is_vanilla = False
    for m in models:
        for nm, nn, nr, td, p, n_m, k in product(num_matches, num_nns, num_refs, train_densify, prob_vals, n_models_vals, top_K_vals):
            if is_edgs and td:
                # EDGS with densify training is not supported
                continue
            t = ExpParams(num_matches=nm, num_nns=nn, num_refs=nr, train_densify=td, is_edgs=is_edgs, is_vgs=is_vgs, is_vanilla=is_vanilla, n_models=n_m, top_K=k, model_name=m)
            fin_exp_list.append(t)


import random
# random.shuffle(fin_exp_list)

for exp in fin_exp_list:
    print(exp)

exp_list = fin_exp_list
print(f"Total experiments to run: {len(exp_list)}")


# for (nm, nn, td, pr, va, nmodel, k) in exp_list:
for exp in exp_list[1:]:
    nm     = exp.num_matches
    nn     = exp.num_nns
    td     = exp.train_nodensify
    ed    = exp.is_edgs
    pr     = exp.is_vgs
    va     = exp.is_vanilla
    nmodel = exp.n_models
    k      = exp.top_K
    model_name = exp.model_name
    downloads_dir = os.path.expanduser("~/Downloads/360_v2")
    model_path = os.path.join(downloads_dir, model_name)
    if pr and va:
        print("Invalid config: both probabilistic and vanilla cannot be true simultaneously.")
        continue
    if va and ed:
        print("Invalid config: vanilla and edgs cannot be true simultaneously.")
        continue


    # output_path = f"./outputs/exp_{model_name}_nm{nm}_nn{nn}_td{td}_pr{pr}_va{va}_nmodel{nmodel}_k{k}"
    # cmd = f"python train.py   train.gs_epochs=1500  train.no_densify={td}   gs.dataset.source_path={model_path}   gs.dataset.model_path={output_path}   init_wC.matches_per_ref={nm}   init_wC.nns_per_ref={nn} init_wC.num_refs={exp.num_refs} gs.vgs.is_probabilistic={pr} gs.vgs.vanilla={va} gs.vgs.num_models={nmodel} gs.vgs.top_K={k}"
    exp_name = f'exp_{model_name}_nm{nm}_nn{nn}_td{td}_ed{ed}_pr{pr}_va{va}_nmodel{nmodel}_k{k}'
    output_path = f"./outputs/{exp_name}"
    cmd = f"python train.py   train.gs_epochs=16000  train.no_densify={td}   gs.dataset.source_path={model_path}   gs.dataset.model_path={output_path}   init_wC.matches_per_ref={nm}   init_wC.nns_per_ref={nn} init_wC.num_refs={exp.num_refs} gs.vgs.is_probabilistic={pr} gs.vgs.vanilla={va} gs.vgs.num_models={nmodel} gs.vgs.top_K={k} init_wC.exp_name={exp_name}"
    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True)
    

