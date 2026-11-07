import copy
import datetime
import glob
import json
import os
import re
import shutil
import signal
import time

import numpy as np
import torch

from . import data as data_mod
from . import losses, networks, official, phema_tools, presets
from .utils import KI, MI, Logger, Timer, code_fingerprint, crossed, device_info, format_nimg, now_iso, parse_nimg, to_builtin, write_json

STATE_FORMAT = 'fastgen-teacher-state-v1'
EARLY_SAMPLE_PAIRS = [(int(i * 151 / 8), i) for i in range(8)]


def _state_name(nimg):
    return f'training-state-{int(nimg):010d}.pt'

def snapshot_interval(sched, nimg):
    if nimg <= sched['dense_until']:
        return sched['dense']
    if nimg <= sched.get('mid_until', 0):
        return sched['mid']
    return sched['sparse']

def count_snapshots(sched, start, end):
    b1 = sched['dense_until']
    b2 = max(b1, sched.get('mid_until', 0))
    n = 0
    for lo, hi, interval in [(0, b1, sched['dense']), (b1, b2, sched.get('mid', 0)), (b2, None, sched['sparse'])]:
        a, b = max(start, lo), (end if hi is None else min(end, hi))
        if interval > 0 and b > a:
            n += b // interval - a // interval
    return n

def find_latest_state(run_dir):
    paths = glob.glob(os.path.join(run_dir, 'training-state-*.pt'))
    best = None
    for p in paths:
        try:
            nimg = int(torch.load(p, map_location='cpu', weights_only=True, mmap=True)['state']['cur_nimg'])
        except Exception as e:
            print(f'Warning: skipping unreadable training state {p}: {type(e).__name__}: {str(e)[:300]}', flush=True)
            continue
        if best is None or nimg > best[0]:
            best = (nimg, p)
    return None if best is None else best[1]

def _link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def _is_link_or_copy(a, b):
    if not (os.path.isfile(a) and os.path.isfile(b)):
        return False
    if os.path.samefile(a, b):
        return True
    sa, sb = os.stat(a), os.stat(b)
    return sa.st_size == sb.st_size and int(sa.st_mtime) == int(sb.st_mtime)

RUN_LOGS = ('stats.jsonl', 'fid.jsonl', 'guard.jsonl')
DERIVED_DIRS = ('phema', 'posttrain')
_RUN_FILE_RE = re.compile(r'(training-state|ema|early|fid)-(\d+)(?:-\d+\.\d+)?\.(?:pt|png)')

def _file_nimg(name):
    m = _RUN_FILE_RE.fullmatch(name)
    return None if m is None else int(m.group(2))

def _run_files(run_dir):
    out = []
    for sub in ('', 'samples'):
        d = os.path.join(run_dir, sub)
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            n = _file_nimg(name)
            if n is not None and os.path.isfile(os.path.join(d, name)):
                out.append((os.path.join(sub, name) if sub else name, n))
    return out

def _read_log(path):
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f.read().split('\n'):
            if line.strip():
                try:
                    rec = json.loads(line)
                    out.append((line, int(rec.get('nimg', 0)) if isinstance(rec, dict) else 0))
                except (ValueError, TypeError):
                    out.append((line, None))
    return out

class _SetAside:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.dir = None
        self.moved = 0
        self.dropped_records = 0

    def _dst(self, rel):
        if self.dir is None:
            base = os.path.join(self.run_dir, 'abandoned', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
            d, i = base, 1
            while os.path.exists(d):
                d, i = f'{base}-{i}', i + 1
            os.makedirs(d)
            self.dir = d
        dst = os.path.join(self.dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        return dst

    def move(self, rel):
        shutil.move(os.path.join(self.run_dir, rel), self._dst(rel))
        self.moved += 1

    def truncate_log(self, rel, max_nimg):
        path = os.path.join(self.run_dir, rel)
        records = _read_log(path)
        keep = [line for line, n in records if n is not None and n <= max_nimg]
        if len(keep) == len(records):
            return
        shutil.copy2(path, self._dst(rel))
        with open(path + '.tmp', 'w') as f:
            f.write(''.join(line + '\n' for line in keep))
        os.replace(path + '.tmp', path)
        self.dropped_records += len(records) - len(keep)


class TeacherTraining:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.log = Logger(args.run_dir)
        self.stop_requested = None
        self.pending_events = []
        resume_path = self._resume_path()
        loaded = torch.load(resume_path, map_location='cpu', weights_only=True) if resume_path else None
        self._check_args(loaded)
        self._prepare_run_dir(loaded, resume_path)
        self._setup_backends()
        self.prep = data_mod.load_prepared(args.prep_dir)
        self.stats = self.prep['stats']
        self._build(loaded, resume_path)
        self._install_signal_handlers()


    def _setup_backends(self):
        a = self.args
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = bool(a.tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        self.vram_limit = float(a.vram_limit_bytes)
        if self.device.type == 'cuda':
            total = torch.cuda.get_device_properties(self.device).total_memory
            frac = min(1.0, self.vram_limit / total)
            torch.cuda.set_per_process_memory_fraction(frac, self.device)
            self.log.print(f'VRAM cap: {self.vram_limit / 1e9:.2f} GB of {total / 1e9:.2f} GB (allocator fraction {frac:.4f})')

    def _resume_path(self):
        r = self.args.resume
        if r in (None, 'none', ''):
            return None
        if r == 'auto':
            return find_latest_state(self.args.run_dir)
        if not os.path.isfile(r):
            raise FileNotFoundError(r)
        here = find_latest_state(self.args.run_dir)
        if here is not None and os.path.abspath(here) != os.path.abspath(r) and not self.args.force:
            raise ValueError(f'{self.args.run_dir} already has its own training state {here}. Continue it with --resume auto '
                             f'(the default), or pass --force to restart this run directory from {r} (files it will not '
                             f'continue from are moved to abandoned/).')
        return r

    def _check_args(self, loaded):
        a = self.args
        override = getattr(a, 'override_ref_lr', None)
        if loaded is None:
            if override is not None:
                raise ValueError('--override-ref-lr changes alpha_ref when resuming; a new run takes --ref-lr')
            return
        if loaded['network']['preset'] != a.preset:
            raise ValueError(f"The resumed state is a {loaded['network']['preset']} network, not --preset {a.preset}")
        if a.ref_lr is not None and override is None and float(a.ref_lr) != float(loaded['recipe']['ref_lr']):
            raise ValueError(f"--ref-lr {a.ref_lr} differs from the resumed run's {loaded['recipe']['ref_lr']}; "
                             f"--override-ref-lr changes it on resume")
        if loaded['state']['subset'] != a.subset and not a.allow_data_switch:
            raise ValueError(f"Resumed run trained on '{loaded['state']['subset']}', not '{a.subset}'. "
                             f"Pass --allow-data-switch to continue on the new set.")

    def _prepare_run_dir(self, loaded, resume_path):
        a = self.args
        run_dir = a.run_dir
        start = 0 if loaded is None else int(loaded['state']['cur_nimg'])
        self.src_dir = None if loaded is None else os.path.dirname(os.path.abspath(resume_path))
        self.same_dir = self.src_dir is not None and os.path.samefile(self.src_dir, run_dir)
        files = _run_files(run_dir)
        logs = {name: [n for _, n in _read_log(os.path.join(run_dir, name))] for name in RUN_LOGS}
        derived = [d for d in DERIVED_DIRS + ('lineage',) if os.path.isdir(os.path.join(run_dir, d))]
        if self.same_dir:
            foreign = []
        elif loaded is None:
            foreign = [rel for rel, _ in files] + [name for name, ns in logs.items() if ns] + derived
        else:
            foreign = [rel for rel, n in files if n <= start and not (rel.startswith('ema-') and
                                                                     _is_link_or_copy(os.path.join(self.src_dir, rel), os.path.join(run_dir, rel)))]
            foreign += [rel for rel, _ in files if rel.startswith('training-state-')]
            foreign += [name for name, ns in logs.items() if any(n is not None and n <= start for n in ns)]
        aside = _SetAside(run_dir)
        if foreign:
            if not a.force:
                what = 'snapshots or logs but no training state' if loaded is None else 'files of another run'
                raise ValueError(f'{run_dir} already holds {what} ({", ".join(foreign[:3])}{", ..." if len(foreign) > 3 else ""}). '
                                 f'Use a new --run-dir, or pass --force to move them to {run_dir}/abandoned/ and start over.')
            for name in sorted(os.listdir(run_dir)):
                if name != 'abandoned':
                    aside.move(name)
        else:
            later = [rel for rel, n in files if n > start]
            for rel in later:
                aside.move(rel)
            for name in RUN_LOGS:
                aside.truncate_log(name, start)
            if any(rel.startswith('ema-') for rel in later):
                for d in DERIVED_DIRS:
                    if os.path.isdir(os.path.join(run_dir, d)):
                        aside.move(d)
            for name in sorted(os.listdir(run_dir)):
                if name.endswith('.tmp') and os.path.isfile(os.path.join(run_dir, name)):
                    aside.move(name)
        if aside.dir is not None:
            what = (f'the previous contents ({aside.moved} files/directories)' if foreign else
                    f'{aside.moved} files/directories and {aside.dropped_records} log records after {format_nimg(start)}')
            self.log.print(f'Moved {what} to {aside.dir}')
            self.pending_events.append(dict(kind='set_aside', nimg=start, dir=aside.dir, files=aside.moved,
                                            records=aside.dropped_records, time=now_iso()))
        if loaded is not None and not self.same_dir:
            self._adopt_lineage(start)

    def _adopt_lineage(self, start):
        src, run_dir = self.src_dir, self.args.run_dir
        for p in sorted(glob.glob(os.path.join(src, 'ema-*.pt'))):
            n = _file_nimg(os.path.basename(p))
            if n is not None and n <= start:
                _link_or_copy(p, os.path.join(run_dir, os.path.basename(p)))
        lineage_dir = os.path.join(run_dir, 'lineage', os.path.basename(src))
        os.makedirs(lineage_dir, exist_ok=True)
        for name in RUN_LOGS + ('log.txt', 'config.json', 'training_time.json'):
            if os.path.isfile(os.path.join(src, name)):
                shutil.copy2(os.path.join(src, name), os.path.join(lineage_dir, name))
        if os.path.isdir(os.path.join(src, 'samples')):
            shutil.copytree(os.path.join(src, 'samples'), os.path.join(lineage_dir, 'samples'), dirs_exist_ok=True)
        if os.path.isdir(os.path.join(src, 'lineage')):
            shutil.copytree(os.path.join(src, 'lineage'), os.path.join(run_dir, 'lineage'), dirs_exist_ok=True)
        self.log.print(f'Adopted the lineage of {src} up to {format_nimg(start)}: snapshots linked, logs in {lineage_dir}')

    def _build(self, loaded, resume_path):
        a = self.args
        override = getattr(a, 'override_ref_lr', None)
        self.recipe = presets.recipe(a.preset, a.ref_lr) if loaded is None else loaded['recipe']
        if loaded is None and a.batch_size is not None:
            self.recipe['batch_size'] = int(a.batch_size)
        self.ref_lr_change = None
        if override is not None and float(override) != float(self.recipe['ref_lr']):
            self.ref_lr_change = dict(kind='ref_lr_override', nimg=int(loaded['state']['cur_nimg']), old=float(self.recipe['ref_lr']),
                                      new=float(override), time=now_iso())
            self.recipe['ref_lr'] = float(override)
        self.batch_size = self.recipe['batch_size']
        sigma_data = float(self.stats['sigma_data'])
        self.sigma_data = sigma_data
        self.P_mean = presets.p_mean_for(self.recipe['family'], sigma_data)
        self.P_std = self.recipe['P_std']

        if loaded is None:
            planned = parse_nimg(a.planned_images) if a.planned_images else a.total_steps * self.batch_size
            p = presets.dropout_for_planned_images(planned) if a.dropout == 'auto' else float(a.dropout)
            self.net_cfg = presets.network_config(a.preset, dropout=p, precision=a.precision, sigma_data=sigma_data)
            self.state = dict(cur_nimg=0, step=0, data_offset_nimg=0, subset=a.subset, dropout=p, planned_images=int(planned),
                              micro_batch=int(a.micro_batch), events=[], fid_history=[], decision_history=[], guard_history=[],
                              guard_raised=False, small_improvements=0, best_decision=None, saved_snapshots=[],
                              time=dict(wall_sec=0.0, train_sec=0.0, eval_sec=0.0, ckpt_sec=0.0, segments=[]), lineage=[],
                              stop_reason=None)
        else:
            self.net_cfg = loaded['network']
            self.state = st = loaded['state']
            entry = None if self.same_dir else dict(run_dir=self.src_dir, subset=st['subset'], end_nimg=st['cur_nimg'])
            if st['subset'] != a.subset:
                entry = entry or dict(run_dir=self.src_dir, subset=st['subset'], end_nimg=st['cur_nimg'])
                self._switch_data(entry)
            if entry is not None:
                st['lineage'].append(entry)
            if a.micro_batch_override:
                st['micro_batch'] = int(a.micro_batch_override)
        self.state['events'] += self.pending_events
        if self.ref_lr_change is not None:
            self.state['events'].append(self.ref_lr_change)

        torch.manual_seed(a.seed)
        self.net = networks.build_network(self.net_cfg)
        with torch.no_grad():
            self.net.mean_image.copy_(self.prep['mean_image'])
        self.net.set_dropout(self._dropout_table(self.state['dropout']))
        if loaded is not None:
            self.net.load_state_dict(loaded['net'], strict=True)
        self.net.set_attention(getattr(a, 'attention', 'einsum'))
        self.net.to(self.device)
        if a.channels_last:
            self.net.to(memory_format=torch.channels_last)
        self.net.train()
        self.params = [p for p in self.net.parameters() if p.requires_grad]
        fused = bool(a.fused_adam and self.device.type == 'cuda')
        self.optimizer = torch.optim.Adam(self.params, lr=0.0, betas=tuple(self.recipe['betas']), eps=self.recipe['eps'], fused=fused)
        self.ema = official.PowerFunctionEMA(self.net, stds=list(self.recipe['ema_stds']))
        if loaded is not None:
            self.optimizer.load_state_dict(loaded['optimizer'])
            self.ema.load_state_dict(loaded['ema'])
        for e in self.ema.emas:
            e.eval()
        self.train_net = torch.compile(self.net) if a.compile else self.net
        self.augment = official.AugmentPipe(**self.recipe['augment']) if self.recipe['augment'] else None
        self.loss_fn = losses.LOSSES[self.recipe['loss']]

        subset_idx = self.prep['splits'][f"{self.state['subset']}_train"]
        if getattr(a, 'limit_images', None):
            subset_idx = sorted(np.random.RandomState(0).choice(subset_idx, a.limit_images, replace=False).tolist())
        self.train_indices = subset_idx
        self.stream = data_mod.TrainStream(self.prep['images'][subset_idx], self.prep['labels'][subset_idx], seed=a.seed, device=self.device)

        counts = networks.count_parameters(self.net)
        self.log.print(f"Network {self.net_cfg['preset']} ({self.net_cfg['family']}): {counts['trainable']:,} trainable + {counts['frozen']:,} frozen = {counts['total']:,} parameters")
        if counts['total'] > 100_000_000:
            raise ValueError('Network exceeds the 100M parameter limit')
        self.state_bytes = 4 * 5 * counts['total']
        self.snapshot_pair_bytes = len(self.recipe['ema_stds']) * (2 * counts['trainable'] + 4 * counts['frozen'])
        self.disk_warned = False
        self.log.print(f"Data: {self.state['subset']} set, {self.stream.n} images; sigma_data {sigma_data:.5f}; P_mean {self.P_mean:.4f}, P_std {self.P_std}")
        self.log.print(f"Recipe: ref_lr {self.recipe['ref_lr']}, batch {self.batch_size} = {self.state['micro_batch']} x {self.batch_size // self.state['micro_batch']}, dropout {self.state['dropout']}, precision {self.net_cfg['kwargs'].get('precision')}")
        if loaded is not None:
            self.log.print(f"Resumed from {resume_path} at {format_nimg(self.state['cur_nimg'])}, step {self.state['step']}")
        if self.ref_lr_change is not None:
            c = self.ref_lr_change
            self.log.print(f"alpha_ref {c['old']} -> {c['new']} from {format_nimg(c['nimg'])} (--override-ref-lr)")

        os.makedirs(os.path.join(a.run_dir, 'samples'), exist_ok=True)
        cfg_path = os.path.join(a.run_dir, 'config.json')
        new_cfg = loaded is None or not self.same_dir or not os.path.isfile(cfg_path)
        if new_cfg:
            cfg = dict(network=self.net_cfg, recipe=self.recipe, stats=self.stats, args=vars(a), P_mean=self.P_mean, P_std=self.P_std,
                       created=now_iso(), code_sha256=code_fingerprint())
        else:
            with open(cfg_path) as f:
                cfg = json.load(f)
            cfg['recipe']['ref_lr'] = self.recipe['ref_lr']
        if self.ref_lr_change is not None:
            cfg.setdefault('ref_lr_overrides', []).append(self.ref_lr_change)
        if new_cfg or self.ref_lr_change is not None:
            write_json(cfg_path, cfg)
        self.segment = dict(start=now_iso(), start_nimg=self.state['cur_nimg'], end=None, end_nimg=None,
                            args=vars(a), device=device_info(self.device), code_sha256=code_fingerprint())
        self.state['time']['segments'].append(self.segment)
        self._account_lost_time(loaded is not None)
        self._wall_base = self.state['time']['wall_sec']
        self._wall_start = time.time()
        self.fid = None
        self.guard = None
        self.sched = dict(status=parse_nimg(a.status_every), dense=parse_nimg(a.snapshot_dense), dense_until=parse_nimg(a.snapshot_dense_until),
                          mid=parse_nimg(getattr(a, 'snapshot_mid', 0)), mid_until=parse_nimg(getattr(a, 'snapshot_mid_until', 0)),
                          sparse=parse_nimg(a.snapshot_every), guard=parse_nimg(a.guard_every), quick=parse_nimg(a.quick_fid_every),
                          decision=parse_nimg(a.decision_fid_every), state=parse_nimg(a.state_every),
                          early=[parse_nimg(e) for e in a.early_samples.split(',') if e.strip()])
        self._disk_report()

    def _account_lost_time(self, resumed):
        t = self.state['time']
        t.setdefault('lost_wall_sec', 0.0)
        path = os.path.join(self.args.run_dir, 'training_time.json')
        if resumed and os.path.isfile(path):
            with open(path) as f:
                last = json.load(f)
            logged = float(last.get('segment_wall_sec_total', 0.0))
            if logged > t['wall_sec'] + 1.0:
                lost = logged - t['wall_sec']
                t['lost_wall_sec'] += lost
                t['wall_sec'] = logged
                self.log.print(f'  counted {lost / 3600:.2f} h of wall-clock logged after the last training state')

    def _dropout_table(self, p):
        return presets.dropout_table(self.net_cfg['family'], p, self.net_cfg['kwargs']['channel_mult'])

    def _switch_data(self, entry):
        a, st = self.args, self.state
        self.log.print(f"Data switch: '{st['subset']}' -> '{a.subset}' at {format_nimg(st['cur_nimg'])}; previous run {entry['run_dir']}")
        entry.update(fid_history=st['fid_history'], decision_history=st['decision_history'])
        st['events'].append(dict(kind='data_switch', nimg=st['cur_nimg'], src=entry['run_dir'], old=st['subset'], new=a.subset))
        st['subset'] = a.subset
        st['data_offset_nimg'] = st['cur_nimg']
        st['fid_history'], st['decision_history'], st['small_improvements'], st['best_decision'] = [], [], 0, None

    def _install_signal_handlers(self):
        def handler(signum, _frame):
            self.stop_requested = f'signal {signal.Signals(signum).name}'
            print(f'\nReceived {signal.Signals(signum).name}; stopping after this step and saving the training state.', flush=True)
        for s in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(s, handler)
            except ValueError:
                pass


    def train_step(self, cur_nimg, bucket_stats):
        a, r = self.args, self.recipe
        official.set_random_seed(a.seed, 0, cur_nimg)
        batch = self.batch_size
        mb = self.state['micro_batch']
        assert batch % mb == 0
        lr = float(official.learning_rate_schedule(cur_nimg=cur_nimg, batch_size=batch, ref_lr=r['ref_lr'], ref_batches=r['ref_batches'],
                                                   rampup_Mimg=r['rampup_steps'] * batch / 1e6))
        self.optimizer.zero_grad(set_to_none=True)
        sigma_all = losses.sample_sigma(batch, self.P_mean, self.P_std, self.device, stratified=r['stratified_sigma'])
        data_pos = cur_nimg - self.state['data_offset_nimg']
        infos = []
        for k in range(batch // mb):
            images, labels = self.stream.get(data_pos + k * mb, mb)
            if self.augment is not None:
                images, aug_labels = self.augment(images)
            else:
                aug_labels = None
            drop = torch.rand([mb], device=self.device) < r['label_dropout']
            labels = torch.where(drop, torch.full_like(labels, self.net.num_classes), labels)
            onehot = self.net.one_hot(labels)
            if a.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            loss, info = self.loss_fn(self.train_net, images, onehot, sigma_all[k * mb : (k + 1) * mb], self.sigma_data, augment_labels=aug_labels)
            loss.sum().mul(1.0 / batch).backward()
            infos.append(info)
        if bucket_stats is not None:
            for info in infos:
                bucket_stats.add(info)
        grads = [p.grad for p in self.params if p.grad is not None]
        grad_norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
        nonfinite = 0
        if not bool(torch.isfinite(grad_norm)):
            nonfinite = int(sum(int((~torch.isfinite(g)).sum()) for g in grads))
        for g in grads:
            torch.nan_to_num(g, **r['nan_to_num'], out=g)
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.optimizer.step()
        return float(grad_norm), nonfinite, lr

    def finish_step(self, cur_nimg):
        self.ema.update(cur_nimg=cur_nimg, batch_size=self.batch_size)
        self.net.post_step_()


    def _ema_by_std(self, std):
        for ema_net, suffix in self.ema.get():
            if abs(float(suffix[1:]) - std) < 1e-9:
                return ema_net
        raise KeyError(std)

    def _fid_refs(self):
        return ['holdout'] if self.state['subset'] == 'pilot' else ['train0', 'train1', 'train2']

    def _get_fid(self):
        if self.fid is None:
            from . import fid as fid_mod
            self.fid = fid_mod.ProxyFID(self.prep, self.device, num_workers=self.args.fid_workers, work_dir=self.args.run_dir)
        return self.fid

    def _get_guard(self):
        if self.guard is None:
            from . import guard as guard_mod
            self.guard = guard_mod.MemorisationGuard(self.prep, self.train_indices, self.device, self.sigma_data,
                                                     sample_steps=self.args.fid_steps, sample_batch=self.args.fid_batch,
                                                     max_images=self.args.guard_max_images)
        return self.guard

    def _sampler_kwargs(self):
        return dict(num_steps=self.args.fid_steps)

    def run_fid(self, std, seed, refs, grid=None):
        net = self._ema_by_std(std)
        while True:
            try:
                res = self._get_fid().evaluate(net, refs, seed=seed, per_class=self.args.fid_per_class, batch_size=self.args.fid_batch,
                                               sampler_kwargs=self._sampler_kwargs(), grid_path=grid)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.args.fid_batch <= 8:
                    raise
                self.args.fid_batch //= 2
                self.log.print(f'  out of memory while sampling; FID batch -> {self.args.fid_batch}')
        res.update(nimg=self.state['cur_nimg'], std=std)
        return res

    def quick_check(self):
        cur = self.state['cur_nimg']
        grid = os.path.join(self.args.run_dir, 'samples', f'fid-{cur:010d}.png')
        res = self.run_fid(0.100, 0, self._fid_refs(), grid=grid)
        res['kind'] = 'quick'
        self.state['fid_history'].append(res)
        self.log.record('fid', res)
        self.log.print(f"  quick FID @ {format_nimg(cur)}: {res['fid']:.3f} (KID {res['kid'] * 1e3:.2f}e-3) vs {list(res['per_ref'])}; "
                       f"sample brightness {res['brightness']['mean']:+.3f} vs train {res['train_brightness']['mean']:+.3f}")
        return res

    def decision_check(self, quick=None):
        cur = self.state['cur_nimg']
        refs = self._fid_refs()
        first = {}
        for std in self.recipe['ema_stds']:
            reuse = quick is not None and abs(quick['std'] - std) < 1e-9 and list(quick['per_ref']) == refs
            first[std] = quick if reuse else self.run_fid(std, 0, refs)
        best_std = min(first, key=lambda s: first[s]['fid'])
        runs = [first[best_std]] + [self.run_fid(best_std, seed, refs) for seed in (1, 2)]
        fids = [r_['fid'] for r_ in runs]
        res = dict(kind='decision', nimg=cur, std=best_std, fids=fids, mean=float(np.mean(fids)), sd=float(np.std(fids, ddof=1)),
                   seed0_by_std={str(s): first[s]['fid'] for s in first}, refs=refs)
        prev_best = self.state['best_decision']
        if prev_best is None:
            res['improvement'] = None
            small = False
        else:
            res['improvement'] = prev_best['mean'] - res['mean']
            small = res['improvement'] < self.args.converge_sd * res['sd']
        self.state['small_improvements'] = self.state['small_improvements'] + 1 if small else 0
        if prev_best is None or res['mean'] < prev_best['mean']:
            self.state['best_decision'] = dict(nimg=cur, mean=res['mean'], sd=res['sd'], std=best_std)
        res['small_improvements'] = self.state['small_improvements']
        self.state['decision_history'].append(res)
        self.log.record('fid', res)
        self.log.print(f"  decision FID @ {format_nimg(cur)}: {res['mean']:.3f} +- {res['sd']:.3f} (std {best_std}); "
                       f"best {self.state['best_decision']['mean']:.3f} @ {format_nimg(self.state['best_decision']['nimg'])}; "
                       f"small improvements in a row: {self.state['small_improvements']}")
        if (self.state['small_improvements'] >= self.args.converge_patience and cur >= parse_nimg(self.args.converge_min)):
            self.stop_requested = self.stop_requested or 'converged'
        return res

    def guard_check(self):
        g = self._get_guard()
        res = g.check(self._ema_by_std(0.100))
        res['nimg'] = self.state['cur_nimg']
        res = json.loads(json.dumps(res, default=float))
        self.state['guard_history'].append(res)
        self.log.record('guard', res)
        self.log.print(f"  guard @ {format_nimg(res['nimg'])}: canary-probe gap (sigma<=1) {res['gap_low']:.5f} "
                       f"(rel {res['rel_gap_low']:+.3f}); copy rate {res['copy_rate']:.3f}")
        hist = [dict(gap_low=h['gap_low'], copy_rate=h['copy_rate']) for h in self.state['guard_history']]
        if self.args.guard_action and not self.state['guard_raised'] and g.should_raise_dropout(hist):
            new_p = round(self.state['dropout'] + 0.05, 4)
            self.log.print(f"  guard action: canary gap widened over {g.widen_checks} checks and copy rate > {g.copy_rate_threshold}; dropout {self.state['dropout']} -> {new_p}")
            self.state['events'].append(dict(kind='dropout_raise', nimg=res['nimg'], old=self.state['dropout'], new=new_p))
            self.state['dropout'] = new_p
            self.state['guard_raised'] = True
            self.net.set_dropout(self._dropout_table(new_p))
            self._set_cfg_dropout(new_p)
        return res

    def _set_cfg_dropout(self, p):
        kw = self.net_cfg['kwargs']
        if self.net_cfg['family'] == 'edm2':
            kw['dropout_res'] = self._dropout_table(p)
        else:
            kw['dropout'] = float(p)

    @torch.no_grad()
    def early_samples(self):
        from . import sampler as sampler_mod
        net = self._ema_by_std(0.100)
        classes = [c for c, _ in EARLY_SAMPLE_PAIRS]
        seeds = [s for _, s in EARLY_SAMPLE_PAIRS]
        x = sampler_mod.sample(net, classes, seeds, device=self.device, sampler_kwargs=self._sampler_kwargs())
        path = os.path.join(self.args.run_dir, 'samples', f"early-{self.state['cur_nimg']:010d}.png")
        sampler_mod.save_grid(x, path, nrow=8)
        return path


    def save_snapshots(self):
        cur = self.state['cur_nimg']
        if cur in self.state['saved_snapshots']:
            return
        meta = dict(subset=self.state['subset'], step=self.state['step'], dropout=self.state['dropout'], sigma_data=self.sigma_data,
                    P_mean=self.P_mean, P_std=self.P_std, null_class_index=self.net.num_classes)
        for ema_net, suffix in self.ema.get():
            std = float(suffix[1:])
            phema_tools.save_snapshot(os.path.join(self.args.run_dir, phema_tools.snapshot_name(cur, std)), ema_net, cur, std,
                                      self.net_cfg, meta=meta)
        self.state['saved_snapshots'].append(cur)

    def save_state(self):
        self._update_time()
        path = os.path.join(self.args.run_dir, _state_name(self.state['cur_nimg']))
        obj = dict(format=STATE_FORMAT, network=to_builtin(self.net_cfg), recipe=to_builtin(self.recipe), state=to_builtin(self.state),
                   net={k: v.detach().cpu().contiguous() for k, v in self.net.state_dict().items()},
                   optimizer=to_builtin(self.optimizer.state_dict()), ema=to_builtin(self.ema.state_dict()))
        tmp = path + '.tmp'
        torch.save(obj, tmp)
        os.replace(tmp, path)
        keep = max(1, int(self.args.keep_states))
        states = sorted(glob.glob(os.path.join(self.args.run_dir, 'training-state-*.pt')))
        for old in states[:-keep]:
            os.remove(old)
        return path

    def _update_time(self):
        t = self.state['time']
        t['wall_sec'] = self._wall_base + (time.time() - self._wall_start)
        self.segment['end'] = now_iso()
        self.segment['end_nimg'] = self.state['cur_nimg']

    def write_time(self):
        self._update_time()
        t = self.state['time']
        write_json(os.path.join(self.args.run_dir, 'training_time.json'), dict(
            total_wall_hours=t['wall_sec'] / 3600, train_hours=t['train_sec'] / 3600, eval_hours=t['eval_sec'] / 3600,
            checkpoint_hours=t['ckpt_sec'] / 3600, lost_hours=t.get('lost_wall_sec', 0.0) / 3600,
            segment_wall_sec_total=t['wall_sec'], peak_memory_allocated=self.state.get('peak_memory_allocated'),
            vram_limit_bytes=self.vram_limit, nimg=self.state['cur_nimg'], steps=self.state['step'],
            mimg=self.state['cur_nimg'] / 1e6, stop_reason=self.state.get('stop_reason'),
            note='Sums over every segment of this run, including resumed and pilot segments (lineage), and time lost after the last saved state.',
            lineage=[dict(run_dir=l['run_dir'], subset=l['subset'], end_nimg=l['end_nimg']) for l in self.state['lineage']],
            segments=t['segments']))


    def _due(self, prev, cur):
        c = self.sched
        snap = crossed(prev, cur, snapshot_interval(c, cur))
        return dict(status=crossed(prev, cur, c['status']), snapshot=snap, early=any(prev < e <= cur for e in c['early']),
                    guard=crossed(prev, cur, c['guard']), quick=crossed(prev, cur, c['quick']),
                    decision=crossed(prev, cur, c['decision']), state=crossed(prev, cur, c['state']))

    def _should_stop(self):
        a = self.args
        if self.stop_requested:
            return self.stop_requested
        if self.state['step'] >= a.total_steps:
            return 'total_steps'
        if a.max_hours is not None and (self._wall_base + time.time() - self._wall_start) / 3600 >= a.max_hours:
            return 'max_hours'
        if a.stop_at is not None and datetime.datetime.now(datetime.timezone.utc) >= datetime.datetime.fromisoformat(a.stop_at).astimezone(datetime.timezone.utc):
            return 'stop_at'
        return None

    def run(self):
        a, st = self.args, self.state
        bucket = losses.BucketStats(self.device)
        acc = dict(steps=0, train_sec=0.0, grad_sum=0.0, grad_max=0.0, nonfinite_entries=0, nonfinite_steps=0, nimg0=st['cur_nimg'])
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        self.log.print(f"Training from step {st['step']} to {a.total_steps} ({format_nimg(a.total_steps * self.batch_size)}) on {self.device}")
        while True:
            reason = self._should_stop()
            if reason:
                break
            prev = st['cur_nimg']
            timer = Timer()
            try:
                grad_norm, nonfinite, lr = self.train_step(prev, bucket)
            except torch.cuda.OutOfMemoryError:
                self._handle_oom()
                continue
            st['cur_nimg'] = prev + self.batch_size
            st['step'] += 1
            self.finish_step(st['cur_nimg'])
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            dt = timer()
            st['time']['train_sec'] += dt
            acc['steps'] += 1
            acc['train_sec'] += dt
            acc['grad_sum'] += grad_norm if np.isfinite(grad_norm) else 0.0
            acc['grad_max'] = max(acc['grad_max'], grad_norm if np.isfinite(grad_norm) else float('inf'))
            acc['nonfinite_entries'] += nonfinite
            acc['nonfinite_steps'] += int(nonfinite > 0)
            self._check_vram()

            due = self._due(prev, st['cur_nimg'])
            if due['status']:
                self._status(acc, bucket, lr)
                bucket.reset()
                acc = dict(steps=0, train_sec=0.0, grad_sum=0.0, grad_max=0.0, nonfinite_entries=0, nonfinite_steps=0, nimg0=st['cur_nimg'])
            evaluated = False
            if due['snapshot']:
                t = Timer()
                self.save_snapshots()
                st['time']['ckpt_sec'] += t()
            t_eval = Timer()
            if due['early']:
                self.early_samples()
                evaluated = True
            quick = None
            if due['quick'] and not a.no_fid:
                quick = self.quick_check()
                evaluated = True
            if due['decision'] and not a.no_fid:
                self.decision_check(quick)
                evaluated = True
            if due['guard'] and not a.no_guard:
                self.guard_check()
                evaluated = True
            if evaluated:
                st['time']['eval_sec'] += t_eval()
            if due['state']:
                t = Timer()
                self.save_state()
                st['time']['ckpt_sec'] += t()
            if due['status'] or evaluated:
                self.write_time()

        st['stop_reason'] = reason
        self.log.print(f"Stopping at step {st['step']} ({format_nimg(st['cur_nimg'])}): {reason}")
        st['events'].append(dict(kind='stop', nimg=st['cur_nimg'], reason=reason, time=now_iso()))
        t = Timer()
        self.save_snapshots()
        path = self.save_state()
        st['time']['ckpt_sec'] += t()
        self.write_time()
        self.log.print(f"Saved final snapshots and training state {path}")
        self.log.print(f"Training time: {st['time']['wall_sec'] / 3600:.2f} h wall-clock in total "
                       f"({st['time']['train_sec'] / 3600:.2f} h training, {st['time']['eval_sec'] / 3600:.2f} h evaluation, "
                       f"{st['time']['ckpt_sec'] / 3600:.2f} h checkpoints)")
        return reason

    def _status(self, acc, bucket, lr):
        st = self.state
        s = bucket.summary()
        imgs = st['cur_nimg'] - acc['nimg0']
        rec = dict(nimg=st['cur_nimg'], mimg=st['cur_nimg'] / 1e6, step=st['step'], epoch=(st['cur_nimg'] - st['data_offset_nimg']) / self.stream.n,
                   lr=lr, img_per_sec=imgs / max(acc['train_sec'], 1e-9), sec_per_step=acc['train_sec'] / max(acc['steps'], 1),
                   grad_norm_mean=acc['grad_sum'] / max(acc['steps'], 1), grad_norm_max=acc['grad_max'],
                   nonfinite_entries=acc['nonfinite_entries'], nonfinite_steps=acc['nonfinite_steps'],
                   micro_batch=st['micro_batch'], dropout=st['dropout'], wall_hours=(self._wall_base + time.time() - self._wall_start) / 3600,
                   train_hours=st['time']['train_sec'] / 3600, **s)
        if self.device.type == 'cuda':
            rec['max_memory_allocated'] = torch.cuda.max_memory_allocated(self.device)
            rec['max_memory_reserved'] = torch.cuda.max_memory_reserved(self.device)
            st['peak_memory_allocated'] = max(st.get('peak_memory_allocated', 0), rec['max_memory_allocated'])
        rec['disk_free_gb'] = self._check_disk() / 1e9
        self.log.record('stats', rec)
        mem = f" mem {rec['max_memory_allocated'] / 1e9:.2f}GB" if 'max_memory_allocated' in rec else ''
        self.log.print(f"step {st['step']:>7d} {st['cur_nimg'] / MI:9.3f}Mi loss {s['loss']:+.4f} wmse {s['wmse']:.4f} u {s['logvar']:+.3f} "
                       f"lr {lr:.2e} |g| {rec['grad_norm_mean']:.3f} {rec['img_per_sec']:.1f} img/s{mem} nonfinite {acc['nonfinite_entries']}")

    def _min_free_bytes(self):
        a = self.args
        auto = 2 * (self.state_bytes + self.snapshot_pair_bytes)
        return auto if getattr(a, 'min_free_gb', 'auto') in (None, 'auto') else float(a.min_free_gb) * 1e9

    def _disk_report(self):
        cur = self.state['cur_nimg']
        pairs = 1 + count_snapshots(self.sched, cur, self.args.total_steps * self.batch_size)
        need = pairs * self.snapshot_pair_bytes + max(1, int(self.args.keep_states)) * self.state_bytes
        free = shutil.disk_usage(self.args.run_dir).free
        self.log.print(f'Disk: {free / 1e9:.1f} GB free in {self.args.run_dir}; the rest of this run writes about {need / 1e9:.1f} GB '
                       f'({pairs} snapshot pairs of {self.snapshot_pair_bytes / 1e9:.2f} GB, {self.args.keep_states} states of '
                       f'{self.state_bytes / 1e9:.2f} GB kept); it stops with a saved state below {self._min_free_bytes() / 1e9:.1f} GB free')
        if need > free - self._min_free_bytes():
            self.log.print('  Warning: not enough free disk space for the whole run; free space (e.g. finished pilot arms) before it runs out')

    def _check_disk(self):
        free = shutil.disk_usage(self.args.run_dir).free
        floor = self._min_free_bytes()
        if free < floor:
            self.log.print(f'  Disk: {free / 1e9:.2f} GB free, below {floor / 1e9:.2f} GB; stopping with a saved state. '
                           f'Free some space and run the same command again.')
            self.stop_requested = self.stop_requested or 'disk_space'
        elif free < 4 * floor and not self.disk_warned:
            self.log.print(f'  Warning: {free / 1e9:.1f} GB of disk left; the run stops by itself below {floor / 1e9:.1f} GB')
            self.disk_warned = True
        return free

    def _check_vram(self):
        if self.device.type != 'cuda':
            return
        peak = torch.cuda.max_memory_allocated(self.device)
        if peak > self.vram_limit:
            self.log.print(f'  VRAM guard: max_memory_allocated {peak / 1e9:.2f} GB > {self.vram_limit / 1e9:.2f} GB')
            self._halve_micro_batch(reason='peak', peak=peak)

    def _handle_oom(self):
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        self.log.print(f"  out of memory at micro-batch {self.state['micro_batch']}")
        self._halve_micro_batch(reason='oom', peak=torch.cuda.max_memory_allocated(self.device))

    def _halve_micro_batch(self, reason, peak):
        mb = self.state['micro_batch']
        if mb <= 8:
            raise RuntimeError('Micro-batch already at 8 and memory still exceeds the limit')
        self.state['micro_batch'] = mb // 2
        self.state['events'].append(dict(kind='micro_batch', nimg=self.state['cur_nimg'], reason=reason, peak=int(peak), old=mb, new=mb // 2))
        self.log.print(f"  micro-batch {mb} -> {mb // 2} ({reason}); batch stays {self.batch_size}")
        torch.cuda.reset_peak_memory_stats(self.device)
