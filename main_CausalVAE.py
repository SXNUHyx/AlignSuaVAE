import argparse
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from lib.data import SyntheticDataset, DataLoaderGPU, create_if_not_exist_dataset
from lib.metrics import mean_corr_coef as mcc
from lib.models import CausalVAE
from lib.utils import Logger, checkpoint
from lib.distributions import Normal
from lib.mlp import MLP

LOG_FOLDER = 'log/'
TENSORBOARD_RUN_FOLDER = 'runs/'
TORCH_CHECKPOINT_FOLDER = 'ckpt/'


class WarmupKLLoss:
    def __init__(self, init_weights, steps,
                 M_N=0.005,
                 eta_M_N=1e-5,
                 M_N_decay_step=3000):
        self.init_weights = init_weights
        self.M_N = M_N
        self.eta_M_N = eta_M_N
        self.M_N_decay_step = M_N_decay_step
        self.speeds = [(1. - w) / s for w, s in zip(init_weights, steps)]
        self.steps = np.cumsum(steps)
        self.stage = 0
        self._ready_start_step = 0
        self._ready_for_M_N = False
        self._M_N_decay_speed = (self.eta_M_N - self.M_N) / self.M_N_decay_step
        self.mn = M_N

    def _get_stage(self, step):
        while True:
            if self.stage > len(self.steps) - 1:
                break
            if step <= self.steps[self.stage]:
                return self.stage
            else:
                self.stage += 1
        return self.stage

    def get_loss(self, step, losses):
        loss = 0.
        stage = self._get_stage(step)
        for i, l in enumerate(losses):
            if i == stage:
                speed = self.speeds[stage]
                t = step if stage == 0 else step - self.steps[stage - 1]
                w = min(self.init_weights[i] + speed * t, 1.)
            elif i < stage:
                w = 1.
            else:
                w = self.init_weights[i]
            if self._ready_for_M_N is False and i == len(losses) - 1 and w == 1.:
                self._ready_for_M_N = True
                self._ready_start_step = step
            l = losses[i] * w
            loss += l
        if self._ready_for_M_N:
            self.mn = min(self.M_N + self._M_N_decay_speed * (step - self._ready_start_step), self.eta_M_N)
        else:
            self.mn = self.M_N
        return self.mn * loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='./data/real_athlete_session_data.npz', help='path to data file in .npz format.')
    parser.add_argument('-x', '--data-args', type=str, default=None)
    parser.add_argument('-b', '--batch-size', type=int, default=512)
    parser.add_argument('-e', '--epochs', type=int, default=400)
    parser.add_argument('-m', '--max-iter', type=int, default=None)
    parser.add_argument('-g', '--hidden-dim', type=int, default=128)
    parser.add_argument('-d', '--depth', type=int, default=4)
    parser.add_argument('-l', '--lr', type=float, default=5e-3)
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-c', '--cuda', action='store_true', default=True)
    parser.add_argument('-p', '--preload-gpu', action='store_true', default=True, dest='preload')
    parser.add_argument('-a', '--anneal', action='store_true', default=False)
    parser.add_argument('-n', '--no-log', action='store_true', default=False)
    parser.add_argument('-q', '--log-freq', type=int, default=25)
    parser.add_argument('-et', '--edge-threshold', type=float, default=0.1)
    parser.add_argument('-rate', '--edge-rate', type=float, default=0.7)
    parser.add_argument('--kl_anneal_portion', type=float, default=0.2)
    parser.add_argument('--kl_const_portion', type=float, default=0.001)
    parser.add_argument('--kl_const_coeff', type=float, default=0.001)
    parser.add_argument('--dag-lambda', type=float, default=1.0, help='DAG penalty weight lambda')

    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    st = time.time()
    if args.file is None:
        args.file = create_if_not_exist_dataset(root='data/', arg_str=args.data_args)
    metadata = vars(args).copy()
    del metadata['no_log'], metadata['data_args']

    device = torch.device('cuda' if args.cuda else 'cpu')
    print('training on {}'.format(torch.cuda.get_device_name(device) if args.cuda else 'cpu'))

 
    if not args.preload:
        dset = SyntheticDataset(args.file, 'cpu')
        loader_params = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader = DataLoader(dset, shuffle=True, batch_size=args.batch_size, **loader_params)
        data_dim, latent_dim, aux_dim = dset.get_dims()
        args.N = len(dset)
        metadata.update(dset.get_metadata())
    else:
        train_loader = DataLoaderGPU(args.file, shuffle=True, batch_size=args.batch_size)
        data_dim, latent_dim, aux_dim = train_loader.get_dims()
        args.N = train_loader.dataset_len
        metadata.update(train_loader.get_metadata())
    if args.max_iter is None:
        args.max_iter = len(train_loader) * args.epochs

 
    model = CausalVAE(
        latent_dim=latent_dim,
        data_dim=data_dim,
        n_layers=args.depth,
        hidden_dim=args.hidden_dim,
        activation='lrelu',
        slope=0.1,
        device=device
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-4)

    ste = time.time()
    print('setup time: {}s'.format(ste - st))

 
    logger = Logger(path=LOG_FOLDER)
    exp_id = logger.get_id()
    tensorboard_run_name = TENSORBOARD_RUN_FOLDER + 'exp' + str(exp_id) + '_'.join(
        map(str, ['dag_lambda', args.dag_lambda, args.batch_size, args.max_iter, args.lr, args.hidden_dim, args.depth]))
    writer = SummaryWriter(logdir=tensorboard_run_name)
    logger.add('elbo')
    logger.add('perfz')
    logger.add('perfn')
    logger.add('dag_loss')
    print('Beginning training for exp: {}'.format(exp_id))

    it = 0
    model.train()
    warmup_kl = WarmupKLLoss(
        init_weights=[1., 1., 1., 1., 1.],
        steps=[100, 100, 100, 100, 100],
        M_N=len(train_loader) / args.batch_size,
        eta_M_N=1.,
        M_N_decay_step=10000
    )
    print('M_N=', warmup_kl.M_N, 'ETA_M_N=', warmup_kl.eta_M_N)

    while it < args.max_iter:
        est = time.time()
        for _, (x, u, z, sn) in enumerate(train_loader):
            optimizer.zero_grad()
            if args.cuda and not args.preload:
                x = x.cuda(device=device, non_blocking=True)
                u = u.cuda(device=device, non_blocking=True)

            elbo, z_est, noise_est, weight_sample, weights_u0 = model.elbo(x, u)

    
            dag_penalty = model.dag_loss()
            total_loss = -elbo + args.dag_lambda * dag_penalty

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            logger.update('elbo', -elbo.item())
            logger.update('dag_loss', dag_penalty.item())

            if it % args.log_freq == 0:
                logger.log()
                writer.add_scalar('data/neg_elbo', logger.get_last('elbo'), it)
                writer.add_scalar('data/dag_penalty', logger.get_last('dag_loss'), it)
            if it % int(args.max_iter / 5) == 0 and not args.no_log:
                checkpoint(
                    TORCH_CHECKPOINT_FOLDER, exp_id, it, model, optimizer,
                    logger.get_last('elbo'), logger.get_last('perfz'), metadata
                )
            it += 1

 
        model.eval()
        logger.log()
        eet = time.time()
        print(
            'epoch {} done in: {:.3f}s, Neg ELBO Loss: {:.3f}, DAG Penalty: {:.3f}'.format(
                int(it / len(train_loader)) + 1, eet - est,
                logger.get_last('elbo'), logger.get_last('dag_loss')
            )
        )
        model.train()

    et = time.time()
    print('training time: {}s'.format(et - ste))
    writer.close()
    if not args.no_log:
        logger.add_metadata(**metadata)
        logger.save_to_json()
        logger.save_to_npz()
    print('total time: {}s'.format(et - st))