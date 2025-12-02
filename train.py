import os
from source.trainer import EDGSTrainer
from source.utils_aux import set_seed
import omegaconf
import wandb
import hydra
from argparse import Namespace
from omegaconf import OmegaConf
import time

@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: omegaconf.DictConfig):
    num_matches = cfg.init_wC.matches_per_ref
    num_nns = cfg.init_wC.nns_per_ref
    is_prob = cfg.gs.vgs.is_probabilistic
    n_models = cfg.gs.vgs.num_models
    top_K = cfg.gs.vgs.top_K
    exp_name = f"EDGS_prob{is_prob}_nm{num_matches}_nn{num_nns}_nmodel{n_models}_k{top_K}"
    _ = wandb.init(entity=cfg.wandb.entity,
                   project=cfg.wandb.project,
                   config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                   tags=[cfg.wandb.tag],
                   name=exp_name,
                   mode=cfg.wandb.mode)
    omegaconf.OmegaConf.resolve(cfg)
    # set_seed(cfg.seed)

    # Init output folder
    print("Output folder: {}".format(cfg.gs.dataset.model_path))
    os.makedirs(cfg.gs.dataset.model_path, exist_ok=True)
    with open(os.path.join(cfg.gs.dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        params = {
                "sh_degree": 3,
                "source_path": cfg.gs.dataset.source_path,
                "model_path": cfg.gs.dataset.model_path,
                "images": cfg.gs.dataset.images,
                "depths": "",
                "resolution": -1,
                "_white_background": cfg.gs.dataset.white_background,
                "train_test_exp": False,
                "data_device": cfg.gs.dataset.data_device,
                "eval": False,
                "convert_SHs_python": False,
                "compute_cov3D_python": False,
                "debug": False,
                "antialiasing": False   
                    }
        cfg_log_f.write(str(Namespace(**params)))
    
    print(cfg.gs)
    # Init both agents
    gs = hydra.utils.instantiate(cfg.gs) 

    # Init trainer and launch training
    trainer = EDGSTrainer(GS=gs,
        training_config=cfg.gs.opt,
        device=cfg.device)
    
    trainer.load_checkpoints(cfg.load)
    trainer.timer.start()
    start_time = time.perf_counter()
    trainer.init_with_corr(cfg.init_wC,cfg.gs.opt)   
    total_init_time = time.perf_counter() - start_time
    wandb.log({"init/time": total_init_time})   
    # trainer.GS.gaussians.training_setup(cfg.gs.opt)
    trainer.train(cfg.train)
    
    # All done
    wandb.finish()
    print("\nTraining complete.")

if __name__ == "__main__":
    main()

