import argparse
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from lib.data import SyntheticDataset, DataLoaderGPU, create_if_not_exist_dataset
from lib.metrics import mean_corr_coef as mcc
from lib.models import AlignDagVAE
from lib.utils import Logger, checkpoint

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

            if self._ready_for_M_N == False and i == len(losses) - 1 and w == 1.:
                self._ready_for_M_N = True
                self._ready_start_step = step
            l = losses[i] * w
            loss += l

        if self._ready_for_M_N:
            self.mn = min(self.M_N + self._M_N_decay_speed *
                          (step - self._ready_start_step), self.eta_M_N)
        else:
            self.mn = self.M_N

        return self.mn * loss


def find_baseline_environment(X, U):
    n_seg = U.shape[1]
    variances = []
    for i in range(n_seg):
        x_cond = X[U[:, i] == 1]
        if len(x_cond) > 0:
            var_u = np.var(x_cond, axis=0).sum()
            variances.append(var_u)
        else:
            variances.append(float('inf'))
    u0_idx = np.argmin(variances)
    print(f"Detected baseline environment (u_0) index: {u0_idx}, minimal conditional variance: {variances[u0_idx]:.4f}")
    return u0_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='./data/real_athlete_session_data.npz', help='path to data file in .npz format. (default None)')
    parser.add_argument('-x', '--data-args', type=str, default=None,
                        help='argument string to generate a dataset.')
    parser.add_argument('-b', '--batch-size', type=int, default=512, help='batch size (default 64)')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='number of epochs (default 20)')
    parser.add_argument('-m', '--max-iter', type=int, default=None, help='max iters, overwrites --epochs')
    parser.add_argument('-g', '--hidden-dim', type=int, default=128, help='hidden dim of the networks (default 50)')
    parser.add_argument('-d', '--depth', type=int, default=4, help='depth (n_layers) of the networks (default 3)')
    
  
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='learning rate (default 1e-3)')
    
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed (default 1)')
    parser.add_argument('-c', '--cuda', action='store_true', default=True, help='train on gpu')
    parser.add_argument('-p', '--preload-gpu', action='store_true', default=True, dest='preload',
                        help='preload data on gpu for faster training.')
    parser.add_argument('-a', '--anneal', action='store_true', default=False, help='use annealing in learning')
    parser.add_argument('-n', '--no-log', action='store_true', default=False, help='run without logging')
    parser.add_argument('-q', '--log-freq', type=int, default=25, help='logging frequency (default 25).')

    parser.add_argument('-et', '--edge-threshold', type=float, default=0.1, help='threshold to cut edge.')
    parser.add_argument('-rate', '--edge-rate', type=float, default=0.7, help='rate to determine edge accoros u')

    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.2, help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.001, help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.001, help='The constant value used for min KL coeff')

  
    parser.add_argument('--flow-type', type=str, default='realnvp', choices=['affine', 'realnvp'], help="Inv Align")
    parser.add_argument('--flow-layers', type=int, default=4, help="RealNVP Number")
    parser.add_argument('--flow-hidden', type=int, default=128, help="MLP")
    parser.add_argument('--gamma-align', type=float, default=1.2, help="L_align")

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

    # load data
    if not args.preload:
        dset = SyntheticDataset(args.file, 'cpu')
        loader_params = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader = DataLoader(dset, shuffle=True, batch_size=args.batch_size, **loader_params)
        data_dim, latent_dim, aux_dim = dset.get_dims()
        args.N = len(dset)
        metadata.update(dset.get_metadata())
        full_X_np = dset.X.numpy()
        full_U_np = dset.U.numpy()
    else:
        train_loader = DataLoaderGPU(args.file, shuffle=True, batch_size=args.batch_size)
        data_dim, latent_dim, aux_dim = train_loader.get_dims()
        args.N = train_loader.dataset_len
        metadata.update(train_loader.get_metadata())
        full_X_np = train_loader.x.cpu().numpy()
        full_U_np = train_loader.u.cpu().numpy()

    u0_idx = find_baseline_environment(full_X_np, full_U_np)

    if args.max_iter is None:
        args.max_iter = len(train_loader) * args.epochs

    # 模型实例化
    model = AlignDagVAE(
        latent_dim=latent_dim,
        data_dim=data_dim,
        aux_dim=aux_dim,
        flow_type=args.flow_type,
        flow_layers=args.flow_layers,
        flow_hidden=args.flow_hidden,
        n_layers=args.depth,
        activation='lrelu',
        hidden_dim=args.hidden_dim,
        device=device,
        anneal=args.anneal
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-4)

    ste = time.time()
    print('setup time: {}s'.format(ste - st))

    # setup loggers
    logger = Logger(path=LOG_FOLDER)
    exp_id = logger.get_id()
    tensorboard_run_name = TENSORBOARD_RUN_FOLDER + 'exp' + str(exp_id) + '_'.join(
        map(str, ['', args.batch_size, args.max_iter, args.lr, args.hidden_dim, args.depth, args.anneal, args.flow_type, args.gamma_align]))
    writer = SummaryWriter(logdir=tensorboard_run_name)
    logger.add('elbo')
    logger.add('perfz')
    logger.add('perfn')
    logger.add('loss_align')

    print('Beginning training for exp: {}'.format(exp_id))

    it = 0
    model.train()
    warmup_kl = WarmupKLLoss(init_weights=[1., 1., 1., 1., 1.],
                             steps=[100, 100, 100, 100, 100],
                             M_N=len(train_loader) / args.batch_size,
                             eta_M_N=1.,
                             M_N_decay_step=10000)
    print('M_N=', warmup_kl.M_N, 'ETA_M_N=', warmup_kl.eta_M_N)

    while it < args.max_iter:
        est = time.time()

        for _, (x, u, z, sn) in enumerate(train_loader):
            model.anneal(args.N, args.max_iter, it)
            optimizer.zero_grad()

            if args.cuda and not args.preload:
                x = x.cuda(device=device, non_blocking=True)
                u = u.cuda(device=device, non_blocking=True)

            elbo, z_est, noise_est, weight_sample, weights_u0 = model.elbo(x, u)
            loss_align = model.compute_align_penalty(u0_idx, device)
            elbo = elbo - args.gamma_align*loss_align
            
    
            if torch.isnan(elbo) or torch.isinf(elbo):
                print(f"[Warning] Iter {it}: NaN/Inf detected in loss! Skipping this batch.")
                optimizer.zero_grad() 
                continue

            elbo.mul(-1).backward()
            
      
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=5.0)
            optimizer.step()

            logger.update('elbo', -elbo.item())
            logger.update('loss_align', loss_align.item())

            if it % args.log_freq == 0:
                logger.log()
                writer.add_scalar('data/elbo', logger.get_last('elbo'), it)
                writer.add_scalar('data/loss_align', logger.get_last('loss_align'), it)

            if it % int(args.max_iter / 5) == 0 and not args.no_log:
                checkpoint(TORCH_CHECKPOINT_FOLDER, exp_id, it, model, optimizer,
                           logger.get_last('elbo'), logger.get_last('perfz'), metadata)

            it += 1

        # evaluate
        model.eval()
        Allsz = np.empty((200, latent_dim))
        Allsn = np.empty((200, latent_dim))
        Allez = np.empty((200, latent_dim))
        Allen = np.empty((200, latent_dim))
        
        with torch.no_grad():
            for _, (x, u, z, sn) in enumerate(train_loader):
                if args.cuda and not args.preload:
                    x = x.cuda(device=device, non_blocking=True)
                    u = u.cuda(device=device, non_blocking=True)
                
                elbo, z_est, noise_est, weight_sample, weights_u0 = model.elbo(x, u)
                Allsz = np.append(Allsz, z.cpu().detach().numpy(), axis=0)
                Allsn = np.append(Allsn, sn.cpu().detach().numpy(), axis=0)
                Allez = np.append(Allez, z_est.cpu().detach().numpy(), axis=0)
                Allen = np.append(Allen, noise_est.cpu().detach().numpy(), axis=0)

        estn = Allen[200:, :]
        estz = Allez[200:, :]
        truez = Allsz[200:, :]
        truen = Allsn[200:, :]

  
        if np.isnan(estn).any() or np.isnan(estz).any():
            print(f"[Warning] Epoch {int(it / len(train_loader)) + 1}: NaN detected in model outputs. Skipping MCC evaluation.")
            perf, perfn = 0.0, 0.0
        else:
            perf, assign_z = mcc(truez, estz, permute=True)
            perfn, assign_n = mcc(truen, estn, permute=True)

        logger.update('perfz', perf)
        logger.update('perfn', perfn)
        logger.log()
        
        writer.add_scalar('metric/perf_z', logger.get_last('perfz'), it)
        writer.add_scalar('metric/perf_n', logger.get_last('perfn'), it)

        eet = time.time()
        print('epoch {} done in: {:.3f}s, elbo: {:.3f}, L_align: {:.3f}, Latent_z MCC: {:.3f}, Latent_n MCC: {:.3f}'.format(
            int(it / len(train_loader)) + 1,
            eet - est,
            logger.get_last('elbo'),
            logger.get_last('loss_align'),
            logger.get_last('perfz'),
            logger.get_last('perfn')
        ))

        model.train()

    et = time.time()
    print('training time: {}s'.format(et - ste))

    writer.close()
    if not args.no_log:
        logger.add_metadata(**metadata)
        logger.save_to_json()
        logger.save_to_npz()

    print('total time: {}s'.format(et - st))