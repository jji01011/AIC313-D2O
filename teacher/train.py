import argparse

from . import presets
from .training import TeacherTraining

def build_parser():
    p = argparse.ArgumentParser(description='Train the FastGen teacher (TEACHER_DESIGN.md).', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group('run')
    g.add_argument('--prep-dir', required=True, help='Output of python -m teacher.prepare_data')
    g.add_argument('--run-dir', required=True, help='Where snapshots, states and logs go')
    g.add_argument('--preset', default='E112', choices=sorted(presets.NETWORK_PRESETS))
    g.add_argument('--subset', default='main', choices=['main', 'pilot'], help='main: all train images minus the canary; pilot: also minus the holdout H')
    g.add_argument('--resume', default='auto', help="'auto' (latest state in --run-dir), 'none', or a training-state file")
    g.add_argument('--allow-data-switch', action='store_true', help='Allow resuming a pilot state on the main set (design 10.4)')
    g.add_argument('--force', action='store_true', help='Let a fresh start or an explicit --resume replace what --run-dir holds after the '
                   'starting point (moved to <run-dir>/abandoned/, never deleted)')
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--limit-images', type=int, default=None, help='Train on a fixed random N images of the set only: the sanity check of design 10.1, not for real runs')
    g.add_argument('--device', default='cuda')

    g = p.add_argument_group('length and stopping')
    g.add_argument('--total-steps', type=int, default=600_000, help='Optimizer steps (batch 256) at which training stops')
    g.add_argument('--converge-min', default='32Mi', help='No convergence stop before this many images')
    g.add_argument('--converge-patience', type=int, default=2, help='Consecutive decision checks with small improvement')
    g.add_argument('--converge-sd', type=float, default=2.0, help='"Small" = improvement of the best FID below this many seed SDs')
    g.add_argument('--max-hours', type=float, default=None, help='Optional cap on total wall-clock hours (all segments)')
    g.add_argument('--stop-at', default=None, help='Optional ISO date-time to stop at, e.g. 2026-10-17T23:59+09:00')

    g = p.add_argument_group('recipe')
    g.add_argument('--ref-lr', type=float, default=None, help='alpha_ref; default from the preset (E112: 0.0045)')
    g.add_argument('--override-ref-lr', type=float, default=None, help='Change alpha_ref when resuming, e.g. to 0.5 x alpha_ref after '
                   'instability (design 12); recorded in the events and config.json')
    g.add_argument('--dropout', default='auto', help="'auto' (design 5.6 rule from planned images) or a rate")
    g.add_argument('--planned-images', default=None, help='Planned images for the auto dropout rule; default total-steps x 256')
    g.add_argument('--batch-size', type=int, default=None, help='Images per optimizer step; default 256 (the design). Smaller only for tests')
    g.add_argument('--micro-batch', type=int, default=64)
    g.add_argument('--micro-batch-override', type=int, default=None, help='Change the micro-batch when resuming')
    g.add_argument('--precision', default=None, choices=['fp16', 'bf16', 'fp32'], help='U-Net dtype on CUDA; default fp16 (EDM2), bf16 (DDPM++)')
    g.add_argument('--compile', action='store_true', help='torch.compile the training network')
    g.add_argument('--channels-last', action='store_true')
    g.add_argument('--attention', default='einsum', choices=['einsum', 'sdpa'], help="EDM2 attention: the official einsum or PyTorch's SDPA (speed option, design 5.8)")
    g.add_argument('--tf32', action='store_true', help='Allow TF32 for fp32 matmuls/convs (EDM2 disables it)')
    g.add_argument('--no-fused-adam', dest='fused_adam', action='store_false')
    g.add_argument('--vram-limit-bytes', type=float, default=20e9, help='Cap on torch.cuda.max_memory_allocated()')
    g.add_argument('--min-free-gb', default='auto', help="Stop with a saved state when the disk has less free space than this; "
                   "'auto' = twice a training state plus a snapshot pair (E112: ~4.6 GB)")

    g = p.add_argument_group('schedules (image counts: 256Ki, 4Mi, 1M, ...)')
    g.add_argument('--status-every', default='256Ki')
    g.add_argument('--snapshot-dense', default='512Ki', help='Snapshot spacing up to --snapshot-dense-until (early guides)')
    g.add_argument('--snapshot-dense-until', default='4Mi')
    g.add_argument('--snapshot-mid', default='2Mi', help='Spacing after the dense phase, up to --snapshot-mid-until (guides at T/16 to T/4)')
    g.add_argument('--snapshot-mid-until', default='48Mi', help='0 or anything below --snapshot-dense-until: no middle phase')
    g.add_argument('--snapshot-every', default='6Mi', help='Spacing after that; post-hoc EMA at the end stays accurate (README, Disk)')
    g.add_argument('--state-every', default='1Mi')
    g.add_argument('--keep-states', type=int, default=2)
    g.add_argument('--early-samples', default='256Ki,512Ki,1Mi,2Mi,4Mi')
    g.add_argument('--guard-every', default='2Mi')
    g.add_argument('--no-guard', action='store_true')
    g.add_argument('--guard-max-images', type=int, default=None, help='Use only the first N canary and probe images (tests)')
    g.add_argument('--no-guard-action', dest='guard_action', action='store_false', help='Log the guard but never raise dropout')
    g.add_argument('--quick-fid-every', default='4Mi')
    g.add_argument('--decision-fid-every', default='8Mi')
    g.add_argument('--no-fid', action='store_true')
    g.add_argument('--fid-per-class', type=int, default=20)
    g.add_argument('--fid-steps', type=int, default=32, help='Heun steps for in-run FID, the guard and sample grids')
    g.add_argument('--fid-batch', type=int, default=256)
    g.add_argument('--fid-workers', type=int, default=4)
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    TeacherTraining(args).run()

if __name__ == '__main__':
    main()
