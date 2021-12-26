from omegaconf import DictConfig, OmegaConf
import logging
import torch
import hydra

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from pcfg.util import seed_all_rng, setup_logger
from pcfg.monitor import *


def _distributed_worker(local_rank, main_func, cfg, ddp):
    assert torch.cuda.is_available(), "CUDA is not available"
    global_rank = 0 + local_rank
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=cfg.dist_url,
            world_size=cfg.num_gpus,
            rank=global_rank,
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(cfg.dist_url))
        raise e
    dist.barrier()
    torch.cuda.set_device(local_rank)
    pg = dist.new_group(range(cfg.num_gpus))
    device = torch.device('cuda', local_rank)
    main_func(cfg, local_rank, ddp, pg, device, DDPMonitor)


def main(cfg, rank, ddp, pg, device, manager):
    cfg.rank = rank
    seed_all_rng(cfg.seed) # + rank)
    
    fname = "eval" if cfg.eval else "train"
    output_dir = f"{cfg.alias_root}/{cfg.alias_name}"
    logger = setup_logger(
        output_dir=output_dir, rank=rank, output=output_dir, fname=fname
    )
    
    if cfg.verbose or not cfg.eval:
        cfg_str = OmegaConf.to_yaml(cfg)
        logger.info(f"\n\n{cfg_str}")

    ngpu = torch.cuda.device_count()
    logger.info("World size: {}; rank: {}".format(ngpu, rank))

    torch.backends.cudnn.benchmark=True
    
    if isinstance(manager, str):
        monitor = eval(manager)(cfg, logger.info, device)
    else:
        monitor = manager(cfg, logger.info, device)
    monitor.learn()


@hydra.main(config_path="configs", config_name="default")
def train(cfg: DictConfig) -> None:
    if cfg.mode == "dp":
        cfg.rank = 0
        torch.cuda.set_device(0)
        main(cfg, 0, False, False, torch.device('cuda', 0), cfg.monitor)
    elif cfg.mode == "ddp":
        try:
            mp.spawn(
                _distributed_worker, 
                nprocs = cfg.num_gpus, 
                args = (main, cfg, False), 
                daemon = False,
            )
        except KeyboardInterrupt as e:
            dist.destroy_process_group()
    else:
        cfg.rank = 0
        torch.cuda.set_device(0)
        main(cfg, 0, False, False, torch.device('cuda', 0), None)

if __name__ == "__main__":
    train()
