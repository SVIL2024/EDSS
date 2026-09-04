import argparse

parser = argparse.ArgumentParser(description='EDSS')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=64, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=7, type=int)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='model/best_xd.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='model/checkpoint.pth')
parser.add_argument('--batch-size', default=96, type=int)
parser.add_argument('--train-list', default='list/xd_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/xd_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='list/gt_label.npy')

parser.add_argument('--lr', default=1e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[3, 6, 10])

# ------------------------------------------------------------------ run mgmt
parser.add_argument('--tag', default='base', type=str,
                    help='run name; used for the log file only (one ckpt per dataset)')
parser.add_argument('--log-dir', default='logs', type=str)
parser.add_argument('--log-mode', default='a', choices=['a', 'w'],
                    help='append to an existing log or start an isolated fresh log')
parser.add_argument('--best-registry', default='model/best_scores.json', type=str,
                    help='global best-per-dataset registry; a ckpt is only written to '
                         'disk when the run beats the recorded best')
parser.add_argument('--test-every', default=0, type=int,
                    help='evaluate every N steps in addition to the end of each epoch '
                         '(0 = epoch end only, the original protocol)')
parser.add_argument('--run-manifest', default='', type=str,
                    help='write an atomic completion manifest after the run-best '
                         'checkpoint has been dumped')

# ------------------------------------------------ original VadCLIP MIL losses
parser.add_argument('--clas2-weight', default=1.0, type=float,
                    help='initial weight of the binary top-k MIL loss')
parser.add_argument('--clasm-weight', default=1.0, type=float,
                    help='initial weight of the multiclass top-k MIL loss')
parser.add_argument('--clas2-final-weight', default=-1.0, type=float,
                    help='post-warmup/fade binary MIL weight; negative keeps the '
                         'initial weight for the whole run')
parser.add_argument('--clasm-final-weight', default=-1.0, type=float,
                    help='post-warmup/fade multiclass MIL weight; negative keeps '
                         'the initial weight for the whole run')
parser.add_argument('--mil-weight-warmup-epochs', default=0, type=int,
                    help='epochs that retain initial MIL weights before retirement')
parser.add_argument('--mil-weight-fade-epochs', default=0, type=int,
                    help='epochs used to linearly reach final MIL weights; zero '
                         'switches immediately after warmup')

# ---------------------------------------------- testing-by-betting aggregator
parser.add_argument('--bet-weight1', default=0.0, type=float,
                    help='weight of the betting MIL loss on the C-branch')
parser.add_argument('--bet-weight2', default=0.0, type=float,
                    help='weight of the betting MIL loss on the A-branch (XD metric)')
parser.add_argument('--bet-eta', default=1.0, type=float)
parser.add_argument('--bet-eta-max', default=4.0, type=float)
parser.add_argument('--bet-lambda', default=0.2, type=float)
parser.add_argument('--bet-lambda-max', default=0.95, type=float)
parser.add_argument('--bet-null-momentum', default=0.95, type=float)
parser.add_argument('--bet-tau', default=20.0, type=float)
parser.add_argument('--bet-bias', default=0.0, type=float)
parser.add_argument('--bet-lr', default=1e-3, type=float,
                    help='lr for the few aggregator scalars (they must calibrate fast)')
parser.add_argument('--bet-warmup-epochs', default=0, type=int,
                    help='train the plain VadCLIP objective for N epochs before the '
                         'sceptic starts betting (lets the trunk settle first)')
parser.add_argument('--grad-clip', default=0.0, type=float,
                    help='global grad-norm clip; 0 disables')
parser.add_argument('--bet-freeze-eta', action='store_true')
parser.add_argument('--bet-freeze-lambda', action='store_true')

# ------------------------- e-BH-inspired adaptive snippet pseudo-labelling
parser.add_argument('--ebh-weight1', default=0.0, type=float,
                    help='weight of the e-BH dense snippet loss on the C-branch')
parser.add_argument('--ebh-weight2', default=0.0, type=float,
                    help='weight of the e-BH dense snippet loss on the A-branch')
parser.add_argument('--ebh-alpha', default=0.5, type=float,
                    help='nominal e-BH level; deployed FDR control requires calibrated e-values')
parser.add_argument('--ebh-alpha2', default=-1.0, type=float,
                    help='nominal e-BH level for the A branch; <0 reuses --ebh-alpha. '
                         'The two branches need different levels because their score '
                         'variance differs (C-branch binary logits are far tighter '
                         'than A-branch multiclass margins), and alpha only bites '
                         'relative to that spread')
parser.add_argument('--ebh-min-reject', default=1, type=int,
                    help='floor on rejections per anomalous bag (MIL needs >=1)')
parser.add_argument('--ebh-normal-weight', default=1.0, type=float,
                    help='weight of the dense negative term on normal videos')
parser.add_argument('--ebh-max-frac', default=0.2, type=float,
                    help='cap the rejection set at this fraction of the video '
                         '(guards the self-training feedback loop)')
parser.add_argument('--ebh-neg-frac', default=0.0, type=float,
                    help='fraction of the lowest-evidence snippets of an ANOMALOUS '
                         'video to supervise as normal on the C branch; the '
                         'evaluated A-branch call keeps this term disabled')
parser.add_argument('--ebh-warmup-epochs', default=0, type=int)
parser.add_argument('--pseudo-selector', default='ebh',
                    choices=['ebh', 'fixed_k', 'fixed_frac', 'original_topk',
                             'soft', 'normal_only'],
                    help='positive pseudo-label selector; e-BH is the proposed rule')
parser.add_argument('--pseudo-fixed-k', default=17, type=int,
                    help='positive snippets per anomalous video for fixed_k ablations')
parser.add_argument('--pseudo-fixed-frac', default=0.10, type=float,
                    help='positive fraction per anomalous video for fixed_frac ablations')
parser.add_argument('--pseudo-soft-temperature', default=1.0, type=float,
                    help='evidence temperature for the detached continuous soft baseline')
parser.add_argument('--dump-run-best', default='', type=str,
                    help='path to dump the run-best weights at the end of '
                         'training (scratch, never touches the tracked best '
                         'checkpoint); used for checkpoint-level ablations')

# -------------------------------------------------- auxiliary temporal priors
parser.add_argument('--smooth-weight', default=0.0, type=float)
parser.add_argument('--sparse-weight', default=0.0, type=float)
parser.add_argument('--temporal-branch', default='A', type=str,
                    help='which branch the smooth/sparse temporal priors act on: '
                         'A = A-branch margin (default; the branch XD is evaluated '
                         'on), C = C-branch binary logit (mirrors the UCF recipe; '
                         'only an indirect influence on the XD metric)')

# -------------------------------------------------- early stopping (sweeps)
parser.add_argument('--early-stop-patience', default=0, type=int,
                    help='legacy integer epoch patience (0 disables); ignored '
                         'when --early-stop-patience-frac is positive')
parser.add_argument('--early-stop-frac', default=0.2, type=float,
                    help='protected fraction of complete epochs before the '
                         'no-improvement counter starts')
parser.add_argument('--early-stop-patience-frac', default=0.4, type=float,
                    help='stop after this fraction of the total epoch budget '
                         'passes without a new within-run best (default: 0.4; '
                         '0 uses the legacy integer patience)')
