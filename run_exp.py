# python train.py   train.gs_epochs=30000   train.no_densify=True   gs.dataset.source_path=/home/moog-2/Downloads/360_v2/stump   gs.dataset.model_path=./outputs/test   init_wC.matches_per_ref=10000   init_wC.nns_per_ref=3 init_wC.num_refs=500 gs.vgs.is_probabilistic=True

import subprocess
num_matches = [15000, 10000, 5000, 1000]
num_nns = [3]
num_refs = 500
prob = [True, False]
n_models = [10,5]
top_K = [4,2]

for nm in num_matches:
    for nn in num_nns:
        for pr in prob:
            for nmodel in n_models:
                for k in top_K:
                    if pr and k > nmodel:
                        continue
                    cmd = f"python train.py   train.gs_epochs=18000   train.no_densify=True   gs.dataset.source_path=/home/moog-2/Downloads/360_v2/stump   gs.dataset.model_path=./outputs/test_nm{nm}_nn{nn}_pr{pr}_nmodel{nmodel}_k{k}   init_wC.matches_per_ref={nm}   init_wC.nns_per_ref={nn} init_wC.num_refs={num_refs} gs.vgs.is_probabilistic={pr} gs.vgs.num_models={nmodel} gs.vgs.top_K={k}"
                    print(f"Running command: {cmd}")
                    subprocess.run(cmd, shell=True)