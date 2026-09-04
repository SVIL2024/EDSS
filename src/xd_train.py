import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import xd_option
from model import CLIPVAD
from utils.betting import DualBranchBetting, abnormal_margin
from utils.ebh import pseudo_label_loss
from utils.dataset import XDDataset
from utils.regularizers import smoothness_loss, sparsity_loss
from utils.runner import (
    BestTracker, early_stop_step, resolve_early_stopping, setup_logger,
    scheduled_loss_weight, write_run_manifest,
)
from utils.tools import get_batch_label, get_prompt_text
from xd_test import test


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


def build_betting_head(args, device):
    """One wealth process per branch (the XD AP protocol reads the A-branch)."""
    return DualBranchBetting(
        weight1=args.bet_weight1, weight2=args.bet_weight2,
        init_eta=args.bet_eta, eta_max=args.bet_eta_max, learn_eta=not args.bet_freeze_eta,
        init_lambda=args.bet_lambda, lambda_max=args.bet_lambda_max,
        learn_lambda=not args.bet_freeze_lambda,
        null_momentum=args.bet_null_momentum, init_tau=args.bet_tau, init_bias=args.bet_bias,
    ).to(device)


def train(model, train_loader, test_loader, args, label_map: dict, device):
    model.to(device)
    logger = setup_logger(args.log_dir, f"xd_{args.tag}", mode=args.log_mode)
    logger.info(f"args: {vars(args)}")

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    bet = build_betting_head(args, device)
    use_ebh = args.ebh_weight1 > 0 or args.ebh_weight2 > 0
    param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
    if bet.enabled or use_ebh:
        param_groups.append({"params": list(bet.parameters()), "lr": args.bet_lr})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)

    tracker = BestTracker(
        dataset="xd", registry_path=args.best_registry, model_path=args.model_path,
        metric_name="AP", tag=args.tag, logger=logger,
    )

    stop_frac = getattr(args, "early_stop_frac", 0.2) or 0.2
    patience_frac = getattr(args, "early_stop_patience_frac", 0.0) or 0.0
    min_epochs, patience = resolve_early_stopping(
        args.max_epoch, stop_frac,
        getattr(args, "early_stop_patience", 0) or 0, patience_frac)
    stale_epochs = 0
    if patience > 0:
        logger.info(
            f"early-stop policy: protect {min_epochs}/{args.max_epoch} epochs; "
            f"stop after {patience} later epochs without a within-run best")

    def evaluate(tag):
        AUC, AP, mAP = test(model, test_loader, args.visual_length, prompt_text,
                            gt, gtsegments, gtlabels, device)
        logger.info(f"{tag} AP={AP:.4f} AUC={AUC:.4f}")
        if bet.enabled:
            logger.info(f"  {bet.describe()}")
        tracker.update(AP, model, meta={
            "epoch": tag, "checkpoint_selection": "best_test_ap"
        })
        model.train()
        return AP

    selection_stats = {}
    for e in range(args.max_epoch):
        model.train()
        clas2_weight = scheduled_loss_weight(
            args.clas2_weight, args.clas2_final_weight, e,
            args.mil_weight_warmup_epochs, args.mil_weight_fade_epochs)
        clasm_weight = scheduled_loss_weight(
            args.clasm_weight, args.clasm_final_weight, e,
            args.mil_weight_warmup_epochs, args.mil_weight_fade_epochs)
        logger.info(
            f"epoch {e + 1} MIL weights: CLAS2={clas2_weight:.4f} "
            f"CLASM={clasm_weight:.4f}")
        run_best_before_epoch = tracker.run_best
        loss_total1 = loss_total2 = loss_totalb = loss_totale = 0.0
        ebh_stats = selection_stats
        for i, item in enumerate(train_loader):
            visual_feat, text_labels, feat_lengths = item
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2 = model(visual_feat, None, prompt_text, feat_lengths)

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 6

            loss = clas2_weight * loss1 + clasm_weight * loss2 + loss3 * 1e-4

            # ---------------- testing-by-betting evidence aggregation --------
            if bet.enabled and e >= args.bet_warmup_epochs:
                lb, bstats = bet(logits1, logits2, text_labels, feat_lengths)
                if bstats:
                    loss_totalb += sum(bstats.values())
                    loss = loss + lb

            # -------- e-BH-inspired adaptive snippet pseudo-labelling --------
            if use_ebh and e >= args.ebh_warmup_epochs:
                log_e1, log_e2, y_bin = bet.log_evidence(
                    logits1, logits2, text_labels, feat_lengths)
                if args.ebh_weight1 > 0:
                    le = pseudo_label_loss(
                        logits1, log_e1, feat_lengths, y_bin, args.ebh_alpha,
                        args.ebh_min_reject, args.ebh_normal_weight, ebh_stats,
                        args.ebh_max_frac, args.ebh_neg_frac,
                        args.pseudo_selector, args.pseudo_fixed_k,
                        args.pseudo_fixed_frac, args.pseudo_soft_temperature)
                    loss_totale += float(le)
                    loss = loss + args.ebh_weight1 * le
                if args.ebh_weight2 > 0:
                    a2 = args.ebh_alpha2 if args.ebh_alpha2 > 0 else args.ebh_alpha
                    le2 = pseudo_label_loss(
                        abnormal_margin(logits2), log_e2, feat_lengths, y_bin,
                        a2, args.ebh_min_reject, args.ebh_normal_weight, ebh_stats,
                        args.ebh_max_frac, 0.0, args.pseudo_selector,
                        args.pseudo_fixed_k, args.pseudo_fixed_frac,
                        args.pseudo_soft_temperature)
                    loss_totale += float(le2)
                    loss = loss + args.ebh_weight2 * le2

            # ---------------- auxiliary temporal priors ----------------------
            if args.smooth_weight > 0 or args.sparse_weight > 0:
                if getattr(args, "temporal_branch", "A").upper() == "C":
                    # mirror the UCF recipe: regularise the C-branch binary logit
                    # (only an indirect influence on the XD metric)
                    pa = torch.sigmoid(logits1.squeeze(-1))
                else:
                    pa = torch.sigmoid(abnormal_margin(logits2))
                if args.smooth_weight > 0:
                    loss = loss + args.smooth_weight * smoothness_loss(pa, feat_lengths)
                if args.sparse_weight > 0:
                    loss = loss + args.sparse_weight * sparsity_loss(pa, feat_lengths)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()

            if i % 50 == 0 and i != 0:
                logger.info(
                    f"epoch {e + 1} step {i}/{len(train_loader)} "
                    f"loss1={loss_total1 / (i + 1):.4f} loss2={loss_total2 / (i + 1):.4f} "
                    f"loss3={loss3.item():.4f} lossbet={loss_totalb / (i + 1):.4f} lossebh={loss_totale / (i + 1):.4f} k={ebh_stats.get(chr(107), 0):.1f}/{ebh_stats.get(chr(107)+chr(110)+chr(101)+chr(103), 0):.0f}"
                )
            if args.test_every > 0 and i % args.test_every == 0 and i != 0:
                evaluate(f"e{e + 1}s{i}")

        scheduler.step()
        eval_ap = evaluate(f"e{e + 1}end")
        tracker.restore(model)

        # Early stopping: protected epochs never consume patience.  Afterwards,
        # stop only after a full patience window contains no new within-run best.
        # Step-level peaks are valid run-best candidates and must not be
        # overwritten by an epoch's weaker final evaluation.
        stale_epochs, should_stop = early_stop_step(
            e + 1, min_epochs, patience,
            tracker.run_best > run_best_before_epoch, stale_epochs)
        if should_stop:
            logger.info(f"early stop: no new run-best for {patience} "
                        f"consecutive post-protection epochs "
                        f"(best={tracker.run_best:.4f})")
            break

    # Save the best test-evaluation point from this run even when it does not
    # beat the isolated registry.  This is the explicit VAD checkpoint protocol
    # shared by every ablation row.
    if getattr(args, "dump_run_best", "") and tracker.best_state is not None:
        os.makedirs(os.path.dirname(args.dump_run_best) or ".", exist_ok=True)
        torch.save(tracker.best_state, args.dump_run_best)
        logger.info(f"run-best dumped to {args.dump_run_best}")

    logger.info(f"run best AP = {tracker.run_best:.4f} (global {tracker.global_best:.4f})")
    if args.run_manifest:
        videos = int(selection_stats.get("selection_video_total", 0))
        train_selection = None
        if videos:
            train_selection = {
                "videos": videos,
                "mean_k": selection_stats["selection_selected_total"] / videos,
                "mean_fraction": selection_stats["selection_fraction_total"] / videos,
            }
        write_run_manifest(
            args.run_manifest, "xd", "AP", tracker.run_best, args.tag,
            args.dump_run_best,
            metadata={
                "seed": args.seed,
                "selector": args.pseudo_selector,
                "selector_active": use_ebh,
                "selector_config": {
                    "alpha": args.ebh_alpha,
                    "max_frac": args.ebh_max_frac,
                    "min_reject": args.ebh_min_reject,
                },
                "early_stopping": {
                    "protected_fraction": stop_frac,
                    "patience_fraction": patience_frac,
                    "protected_epochs": min_epochs,
                    "patience_epochs": patience,
                },
                "mil_schedule": {
                    "clas2_initial": args.clas2_weight,
                    "clas2_final": args.clas2_final_weight,
                    "clasm_initial": args.clasm_weight,
                    "clasm_final": args.clasm_final_weight,
                    "warmup_epochs": args.mil_weight_warmup_epochs,
                    "fade_epochs": args.mil_weight_fade_epochs,
                },
                "training_recipe": {
                    "ebh_weight1": args.ebh_weight1,
                    "ebh_weight2": args.ebh_weight2,
                    "alpha1": args.ebh_alpha,
                    "alpha2": args.ebh_alpha2,
                    "eta": args.bet_eta,
                    "max_frac": args.ebh_max_frac,
                    "normal_weight": args.ebh_normal_weight,
                    "neg_frac_c": args.ebh_neg_frac,
                    "neg_frac_a": 0.0,
                    "selector_warmup_epochs": args.ebh_warmup_epochs,
                    "smooth_weight": args.smooth_weight,
                    "sparse_weight": args.sparse_weight,
                    "temporal_branch": args.temporal_branch,
                    "test_every": args.test_every,
                    "max_epoch": args.max_epoch,
                },
                "budget": {
                    "fixed_k": args.pseudo_fixed_k,
                    "fixed_frac": args.pseudo_fixed_frac,
                },
                "train_selection": train_selection,
            })
        logger.info(f"completion manifest written to {args.run_manifest}")
    return tracker.run_best


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    train(model, train_loader, test_loader, args, label_map, device)
