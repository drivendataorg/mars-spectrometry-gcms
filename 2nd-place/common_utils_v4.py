import importlib
import time, sys
from contextlib import contextmanager
import numpy as np
import warnings
import math
import torch
import yaml
import torch.nn.functional as F
import torch.utils.data
from typing import Sized, Optional
import click
import pandas as pd
import os, sys
import random
from os.path import join, isfile
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR


# try:
#     # sys.path.insert(0, expanduser('~/dev/y22m05gh/'))
#     sys.path.insert(0, '..')
#     sys.path.insert(0, '../..')
#     sys.path.insert(0, '../../..')
#     from v22src.dk3_common import set_dbg, dbg, get_dbg
# except:
#     def dbg(a=None, b=None): pass
#     def set_dbg(a=None): return False
#     def get_dbg(a=None): return False


def load_module_from_rel_path(file_rel_path):
    fpath = os.path.abspath(file_rel_path)
    module_name = file_rel_path.replace('.py', '')
    module_name = os.path.split(module_name)[-1]
    spec = importlib.util.spec_from_file_location(module_name, fpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

    # RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`,
    # but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case,
    # you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8.
    # For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    # set the debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8" (may limit overall performance) or ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB).
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def load_dir_names(cfg):
    # model_params = cfg.model_params
    # model_type = model_params.model_type
    # model_tag = f'{cfg.model_str}_fold{cfg.fold}_seed{cfg.seed}'
    model_tag = f'{cfg.version}'
    model_tag = f'{model_tag}/{cfg.model_str}/{cfg.timm_name}_fold{cfg.fold}_seed{cfg.seed}'
    cfg.model_tag = model_tag
    cfg.checkpoints_dir = join(f"{cfg.OUTPUT_DIR}", 'checkpoints', model_tag)
    cfg.tensorboard_dir = join(f"{cfg.OUTPUT_DIR}", 'tensorboard', model_tag)
    cfg.oof_dir = join(f"{cfg.OUTPUT_DIR}", 'oof', model_tag)
    cfg.orig_val_dir = join(f"{cfg.OUTPUT_DIR}", 'orig_val', model_tag)
    cfg.test_val_dir = join(f"{cfg.OUTPUT_DIR}", 'test_val', model_tag)
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(cfg.tensorboard_dir, exist_ok=True)
    os.makedirs(cfg.oof_dir, exist_ok=True)
    os.makedirs(cfg.orig_val_dir, exist_ok=True)
    os.makedirs(cfg.test_val_dir, exist_ok=True)
    print(cfg.model_tag)


def safe_clone(src):
    assert isinstance(src, dict), 'TODO: fix src via, eg src = vars(src)'
    ret = {}
    for k, v in src.items():
        if isinstance(v, dict):
            v = safe_clone(v)
        ret[k] = v
    ret = toDotDict(ret)
    return ret


def safe_merge_cfg(*, dst, src: dict):
    # https://stackoverflow.com/questions/52783883/how-to-initialize-a-dict-from-a-simplenamespace
    assert isinstance(src, dict), 'TODO: fix src via, eg src = vars(src)'
    for k, v in src.items():
        # assert k not in dst, f'key={k} in dst'
        dst[k] = v


def normalize_experiment_name(run_name: str):
    # if run_name.startswith("experiments/"):
    #     run_name = run_name[len("experiments/"):]
    if run_name.endswith(".yaml"):
        run_name = run_name[: -len(".yaml")]
    return run_name


def load_config_data(*, cfg_name, main_cfg, timm_name, fold, seed, version) -> dict:
    cfg_name = normalize_experiment_name(cfg_name)
    with open(f"{cfg_name}.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = toDotDict(cfg)
    if 'base_model_name' in cfg.model_params:  # e.g. missing in LSTM
        cfg.model_params.base_model_name = timm_name
    cfg.timm_name = timm_name
    cfg.model_str = cfg_name  # from v1
    cfg.cfg_name = cfg_name  # new in v2
    cfg.fold = fold
    cfg.seed = seed
    cfg.version = version

    safe_merge_cfg(dst=cfg, src=main_cfg)
    return cfg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def toDotDict(src: dict):
    ret = {}
    for k, v in src.items():
        if isinstance(v, dict):
            v = toDotDict(v)
        ret[k] = v
    ret = DotDict(ret)
    return ret


class DotDict(dict):
    """dot.notation access to dictionary attributes

    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """  # NOQA

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@contextmanager
def timeit_context(name, enabled=True):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    if enabled:
        print(f"[{name}] finished in {elapsedTime:0.3f}s")


def print_stats(title, array):
    if len(array):
        print(
            "{} shape:{} dtype:{} min:{} max:{} mean:{} median:{}".format(
                title,
                array.shape,
                array.dtype,
                np.min(array),
                np.max(array),
                np.mean(array),
                np.median(array),
            )
        )
    else:
        print(title, "empty")


class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (float, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1.0, eta_min=0, last_epoch=-1, verbose=False, first_epoch_lr_scale=None):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1.0:
            raise ValueError("Expected T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.first_epoch_lr_scale = first_epoch_lr_scale

        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch, verbose)

        self.T_cur = self.last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        lr_scale = 1.0
        if self.last_epoch == 0 and self.first_epoch_lr_scale is not None:
            lr_scale = self.first_epoch_lr_scale

        return [
            lr_scale * (self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2)
            for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>         scheduler.step(epoch + i / iters)

        This function can be called in an interleaved way.

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = int(self.T_i * self.T_mult)
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1.0:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


# def seed_worker(worker_id, worker_seeding='all'):
#     from timm.data.loader import _worker_init
#     _worker_init(worker_id, worker_seeding=worker_seeding)
#     # fix imgaug
#     worker_info = torch.utils.data.get_worker_info()
#     new_seed = worker_info.seed % (2 ** 32 - 1)
#     # np.random.seed(new_seed)  # already done in _worker_init
#     import imgaug
#     imgaug.random.RNG(new_seed)


def seed_worker(worker_id):
    # https://pytorch.org/docs/stable/notes/randomness.html
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def print_metrics(*, labels, preds, verb=True):
    accuracy = ((labels > 0.5) == (preds > 0.5)).mean()
    if verb:
        print('accuracy', accuracy)
    best_loss = None
    search_grid = [0] + list(np.logspace(-5, -1, num=20))
    # for clip in [0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]:
    for clip in search_grid:
        loss_clip = F.binary_cross_entropy(torch.from_numpy(preds * (1 - clip * 2) + clip),
                                           torch.from_numpy(labels))
        if verb:
            print(f'loss clip {clip}: {loss_clip:0.4f}')
        if best_loss is None or best_loss > loss_clip:
            best_loss = loss_clip
            best_clip = clip
    return best_loss, best_clip


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, verbose=False):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch, verbose)


def get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1,
        verbose=False,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch, verbose)


def get_scheduler(cfg, optimizer, *, num_train_steps, num_warmup_steps, verbose=False):
    scheduler = None
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
            verbose=verbose,
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
            verbose=verbose,
        )
    # elif cfg.scheduler=='ReduceLROnPlateau':
    #     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.factor,
    #                                   patience=cfg.patience, verbose=True, eps=cfg.eps)
    # elif cfg.scheduler=='CosineAnnealingLR':
    #     scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr,
    #                                   last_epoch=-1)
    # elif cfg.scheduler=='CosineAnnealingWarmRestarts':
    #     scheduler = CosineAnnealingWarmRestarts(
    #         optimizer, T_0=cfg.T_0, T_mult=1, eta_min=cfg.min_lr, last_epoch=-1)
    # elif cfg.scheduler=='MultiStepLR':
    #     scheduler = MultiStepLR(optimizer, milestones=cfg.milestones, gamma=0.1, last_epoch=-1)
    return scheduler


# ------------------------------- DBG
# __DBG__ = True
__DBG__ = False


def set_dbg(val):
    global __DBG__
    old_val = __DBG__
    __DBG__ = val
    return old_val


def get_dbg():
    global __DBG__
    return __DBG__


def dbg_stats(x):
    bad_idx = np.isnan(x)
    count_nans = np.sum(bad_idx)
    if count_nans > 0:
        dbg('count_nans')
        x = x[~bad_idx]
    dbg('np.mean(x)', 'np.std(x)')
    dbg('np.min(x)', 'np.max(x)')
    dbg('x.shape', 'x.dtype')


def dbg_np(x):
    try:
        idx = np.isnan(x)
        count = np.sum(idx)
        if np.all(idx):
            print(x.shape, x.dtype, 'all_nans_count=', count)
            return
        if count == 0:
            print(x.shape, x.dtype, 'min=', np.min(x), 'max=', np.max(x))
        else:
            print(x.shape, x.dtype,
                  'min=', np.nanmin(x), 'max=', np.nanmax(x), 'nan_count=', count)
    except:
        print(x.shape, x.dtype)


def dbgT(df, n=1):
    # dbg('df.head(n).T', 'df.tail(n).T', 'len(df)')
    dbg('list(df.columns)[:100]', 'list(df.dtypes)[:100]')
    dbg('df.tail(n).T', 'len(df)')


def display(a=None, **kwargs):
    print(a, **kwargs)


def dbg(x, x2=None, exp3=None, text=None, on=None):
    global __DBG__
    if on is None and not __DBG__:
        return __DBG__
    if on is not None and not on:
        return __DBG__
    if x is None:
        print('x=None')
    if isinstance(x, torch.HalfTensor) or isinstance(x, torch.Tensor):
        return dbg_pt(x)
    if isinstance(x, np.ndarray) and isinstance(x2, np.ndarray):
        dbg_np(x)
        dbg_np(x2)
        return
    if isinstance(x, np.ndarray):
        dbg_np(x)
        return
    if isinstance(x, pd.DataFrame):
        print(list(x.columns)[:100])
        print(list(x.dtypes)[:100])
        # print(x.head(2))
        print(x.tail(2))
        print(x.tail(1).T)
        print(len(x))
        return

    if hasattr(x, 'items'):  # new 30-march-2021
        for k, v in x.items():
            print(f'{k}: {v}')
        return

    # if isinstance(x, list):
    #     print('list len=', len(x), x)
    #     return
    if isinstance(x, list):
        dbg('len(x)')
        for xi in x:
            dbg(xi)
        return

    if text:
        print(text)
    frame = sys._getframe(1)
    print(x, '=', repr(eval(x, frame.f_globals, frame.f_locals)))
    if x2:
        print(x2, '=', repr(eval(x2, frame.f_globals, frame.f_locals)))
    if exp3:
        print(exp3, '=', repr(eval(exp3, frame.f_globals, frame.f_locals)))

    return __DBG__


def dbg_pt(x, x2_USE_LIST=None):
    if not get_dbg():
        return
    if x2_USE_LIST is not None:
        dbg_pt(x2_USE_LIST)
    if x is None:
        print('None')
    if isinstance(x, torch.HalfTensor):
        print(x.shape, x.dtype)
        return
    if isinstance(x, torch.Tensor):
        print(x.shape, x.dtype, 'min=', torch.min(x), 'max=', torch.max(x))
        return
    if isinstance(x, list):
        dbg('len(x)')
        for xi in x:
            dbg_pt(xi)
        return
    # dbg(x, x2)  # try dbg for others


def check_CosineAnnealingWarmRestarts():
    import matplotlib.pyplot as plt

    optimizer = torch.optim.SGD([torch.tensor(1)], lr=1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=16, T_mult=1.41421)

    lrs = []
    for _ in range(2222):
        optimizer.step()
        lrs.append(scheduler.get_lr())
        scheduler.step()

    # 251: 77
    # 371: 49
    # 536: 37
    # 771: 27
    # 1101: 17
    # 1536: 13

    plt.plot(lrs, label='Relative learning rate')
    plt.scatter([251, 371, 536, 771, 1101, 1536], [0] * 6, c='r', label='False positive mining')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


# main_cfg = toDotDict(main_cfg)


if __name__ == "__main__":
    check_CosineAnnealingWarmRestarts()
