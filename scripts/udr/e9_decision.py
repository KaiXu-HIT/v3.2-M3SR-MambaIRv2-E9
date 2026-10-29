"""E9: auditable Continue / Pivot / Stop decisions; never train a model.

The primary decision uses DIV2K x4 E4-E7 evidence. Optional DF2K/multiscale
E8 results are kept separate. Missing/invalid evidence cannot become failure.
Only the proposal's qualitative/approximate terms use an explicit policy.
No torch, BasicSR, model import or CUDA kernel is needed by this script.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ('Set5', 'Set14', 'B100', 'Urban100', 'Manga109')
COUNTS = dict(zip(DATASETS, (5, 14, 100, 100, 109)))
MODES = dict(U1='route_concentration', U2='feature_variance', U3='learned_error')
TAGS = dict(a0=0., a1e5=1e-5, a1e4=1e-4, a5e4=5e-4)
CONDITIONS = tuple('D' + str(i) for i in range(8))
POLICY = dict(failure_gain_max=.015, shuffle_equivalence_margin=.01,
              clear_causality_min=.01)
ROUTES = dict(
    A=['完成 UDR-v2 定稿', 'DF2K 扩展', '多倍率验证', '论文规模 benchmark', '上述验证后再考虑 Text'],
    B=['优先研究 geometry representation，不继续增加 fusion block',
       '对比 D、∇D、∇²D、normal-like geometry 和 multi-scale geometry'],
    C=['停止当前 UDR 复杂化路线',
       '路线1 Depth source：比较 Depth Anything V2、Metric3D、Marigold、Depth Pro 及其他 pseudo-depth 模型',
       '路线2 Geometry representation：探索 dense depth 以外的几何表示',
       '路线3 Task reformulation：转为 geometry-guided reconstruction',
       '路线4 Text：仅考虑 region-aware、spatially grounded 或 geometry-conditioned text；不回到 global CLIP FiLM'])


class EvidenceError(ValueError):
    pass


def read(path):
    def invalid(value):
        raise EvidenceError('Nonfinite JSON number: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=invalid)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def record(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def number(value):
    if isinstance(value, bool):
        raise EvidenceError('A metric cannot be a Boolean.')
    value = float(value)
    if not math.isfinite(value):
        raise EvidenceError('Metrics must be finite.')
    return value


def close(actual, expected, label, tolerance=1e-6):
    if abs(number(actual) - number(expected)) > tolerance:
        raise EvidenceError('Inconsistent evidence: ' + label)


def vector(values, n):
    if not isinstance(values, list) or len(values) != n:
        raise EvidenceError('Missing or unequal matched-seed vectors.')
    return [number(value) for value in values]


def above(value, threshold):
    # PSNR differences can move an exact decimal threshold by floating-point
    # cancellation. A 1e-10 dB arithmetic tolerance is not a practical gain.
    return value - threshold > 1e-10


def at_least(value, threshold):
    return value >= threshold or math.isclose(value, threshold, rel_tol=0., abs_tol=1e-10)


def summary_check(item, values, label):
    close(item['mean'], st.mean(values), label + '/mean')
    close(item['std'], st.stdev(values), label + '/std')


def policy_values(value):
    if set(value) != set(POLICY):
        raise EvidenceError('Policy only configures the three qualitative/approximate terms.')
    result = {key: number(v) for key, v in value.items()}
    if not (0 <= result['failure_gain_max'] < .02 and
            0 <= result['shuffle_equivalence_margin'] <= result['clear_causality_min']):
        raise EvidenceError('Invalid approximate-term policy or overlapping weak/failure bands.')
    return result


def input_template():
    return dict(selection='experiments/E4_selection/E4_selection.json',
        e4_report='results/E4_udrv2/ablation/E4_ablation.json',
        e5_report='results/E5_depth_causality/E5_causality.json',
        e5_csv='results/E5_depth_causality/E5_per_image.csv',
        e6_report='results/E6_multiseed/E6_multiseed.json',
        e7_report='results/E7_image_statistics/E7_statistics.json',
        e7_csv='results/E7_image_statistics/E7_per_image.csv',
        e8_metrics=[])


def csv_index(path, seeds, conditions=False):
    index = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            ds, image = row['dataset'], row['image_id']
            seed = int(row['seed'])
            if ds not in DATASETS or not image or seed not in seeds:
                raise EvidenceError('Unexpected CSV dataset/image/seed.')
            key = ds, image, seed
            if conditions:
                if row['condition'] not in CONDITIONS:
                    raise EvidenceError('Unknown E5 Depth condition.')
                key += (row['condition'],)
                for metric in ('psnr', 'ssim'):
                    row[metric] = number(row[metric])
            else:
                for metric in ('psnr', 'ssim'):
                    for prefix in ('baseline', 'udrv2', 'delta'):
                        field = prefix + '_' + metric
                        row[field] = number(row[field])
                    close(row['delta_' + metric], row['udrv2_' + metric] - row['baseline_' + metric],
                          'CSV paired delta', 1e-8)
            if key in index:
                raise EvidenceError('Duplicate per-image evidence: ' + str(key))
            index[key] = row
    for ds in DATASETS:
        images = {key[1] for key in index if key[0] == ds}
        expected = COUNTS[ds] * len(seeds) * (8 if conditions else 1)
        if len(images) != COUNTS[ds] or sum(key[0] == ds for key in index) != expected:
            raise EvidenceError('Incomplete benchmark CSV: ' + ds)
        for image in images:
            for seed in seeds:
                keys = [(ds, image, seed, c) for c in CONDITIONS] if conditions else [(ds, image, seed)]
                if any(key not in index for key in keys):
                    raise EvidenceError('Missing exact paired image/seed/condition.')
    return index


def performance(e6, e7, index):
    seeds = e6['seeds']
    if (len(seeds) not in (3, 5) or len(set(seeds)) != len(seeds) or
            any(type(seed) is not int for seed in seeds) or e7['seeds'] != seeds):
        raise EvidenceError('E9 needs the matching formal 3- or 5-independent-training-seed experiment.')
    if set(e6['per_dataset']) != set(DATASETS) or set(e7['datasets']) != set(DATASETS):
        raise EvidenceError('E6/E7 must contain all original five datasets.')
    if e7['bootstrap_replicates'] != 10000:
        raise EvidenceError('E7 requires exactly 10,000 image bootstrap resamples.')
    # Recompute E7 using its original NumPy SeedSequence streams, without
    # importing E7's CUDA/torch-dependent parent modules.
    import numpy as np
    streams = np.random.SeedSequence(e7['bootstrap_seed']).spawn(len(DATASETS))
    datasets = {}
    for ds, stream in zip(DATASETS, streams):
        images = sorted({key[1] for key in index if key[0] == ds})
        by_seed = []
        for i, seed in enumerate(seeds):
            for label, prefix in (('rgb', 'baseline'), ('udrv2', 'udrv2')):
                for metric in ('psnr', 'ssim'):
                    values = vector(e6['per_dataset'][ds][label]['metrics'][metric]['by_seed'], len(seeds))
                    summary_check(e6['per_dataset'][ds][label]['metrics'][metric], values, ds + '/' + label + '/' + metric)
                    close(st.mean(index[(ds, image, seed)][prefix + '_' + metric] for image in images),
                          values[i], ds + '/E6-CSV/' + label + '/' + metric)
            gain = st.mean(index[(ds, image, seed)]['delta_psnr'] for image in images)
            close(gain, e6['per_dataset'][ds]['udrv2']['delta_psnr_vs_rgb']['by_seed'][i], ds + '/paired gain')
            by_seed.append(gain)
        summary_check(e6['per_dataset'][ds]['udrv2']['delta_psnr_vs_rgb'], by_seed, ds + '/gain')
        image_gains = np.array([st.mean(index[(ds, image, seed)]['delta_psnr'] for seed in seeds)
                                for image in images], dtype=np.float64)
        rng = np.random.default_rng(stream)
        samples = rng.integers(0, len(images), size=(10000, len(images)))
        ci = np.quantile(image_gains[samples].mean(axis=1), [.025, .975])
        item = e7['datasets'][ds]
        if item['images'] != len(images) or item['seeds'] != seeds or item['bootstrap_replicates'] != 10000:
            raise EvidenceError('E7 dataset/seed/bootstrap count mismatch.')
        for actual, expected in zip(ci, item['ci95_psnr']):
            close(actual, expected, ds + '/E7 CI', 1e-8)
        if len(item['ci95_psnr']) != 2:
            raise EvidenceError('E7 interval must contain two endpoints.')
        close(float(image_gains.mean()), item['mean_delta_psnr'], ds + '/E7 mean')
        low, high = map(float, ci)
        datasets[ds] = dict(mean_delta_psnr=st.mean(by_seed), std_delta_psnr=st.stdev(by_seed),
            by_seed=by_seed, ci95_psnr=[low, high], positive_seeds=sum(v > 0 for v in by_seed),
            stable_all_seeds=all(v > 0 for v in by_seed),
            image_evidence='positive_ci' if low > 0 else ('negative_ci' if high < 0 else 'crosses_zero'))
    averages = [st.mean(datasets[ds]['by_seed'][i] for ds in DATASETS) for i in range(len(seeds))]
    item = e6['five_set']['udrv2']['delta_psnr_vs_rgb']
    summary_check(item, averages, 'E6 five-set equal-weight gain')
    for actual, expected in zip(vector(item['by_seed'], len(seeds)), averages):
        close(actual, expected, 'E6 five-set seed pairing')
    # Distinct weight hashes are the report-level E6 independence evidence.
    for label in ('rgb', 'udrv2'):
        hashes = []
        for seed in seeds:
            sha = e6['checkpoints'][label][str(seed)]['sha256']
            if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                raise EvidenceError('Malformed E6 checkpoint SHA256.')
            if sha != e7['checkpoint_sha256'][label][str(seed)]:
                raise EvidenceError('E6/E7 tested different checkpoint hashes.')
            hashes.append(sha)
        if len(set(hashes)) != len(seeds):
            raise EvidenceError('One checkpoint was reused across purported independent training seeds.')
    return dict(seeds=seeds, datasets=datasets, five_set_mean=st.mean(averages),
                five_set_std=st.stdev(averages), five_set_by_seed=averages,
                five_set_stable=all(v > 0 for v in averages))


def causality(e5, index):
    seeds = e5['seeds']
    if e5['complete_five_set'] is not True or set(e5['datasets']) != set(DATASETS):
        raise EvidenceError('Partial E5 smoke tests cannot support an E9 direction decision.')
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise EvidenceError('E5 needs at least three matched inference seeds.')
    means = {condition: [] for condition in CONDITIONS}
    for condition in CONDITIONS:
        for i, seed in enumerate(seeds):
            per_dataset = []
            for ds in DATASETS:
                images = sorted({key[1] for key in index if key[0] == ds})
                item = e5['per_dataset'][ds][condition]
                if item['images'] != len(images):
                    raise EvidenceError('E5 report image count differs from CSV.')
                for metric in ('psnr', 'ssim'):
                    value = st.mean(index[(ds, image, seed, condition)][metric] for image in images)
                    stored = vector(item['metrics'][metric]['by_seed'], len(seeds))
                    summary_check(item['metrics'][metric], stored, 'E5/' + ds + '/' + metric)
                    close(value, stored[i], 'E5 report/CSV metric')
                    if metric == 'psnr':
                        per_dataset.append(value)
            means[condition].append(st.mean(per_dataset))
        close(st.mean(means[condition]), e5['five_set'][condition]['psnr_mean'], 'E5 five-set mean')
    zero = [a - b for a, b in zip(means['D0'], means['D1'])]
    shuffle = [a - b for a, b in zip(means['D0'], means['D3'])]
    for condition, values in (('D1', zero), ('D3', shuffle)):
        close(st.mean(values), e5['five_set'][condition]['correct_minus_condition_psnr'], 'E5 five-set contrast')
        for expected, actual in zip(vector(e5['five_set'][condition]['paired_delta_by_seed'], len(seeds)), values):
            close(expected, actual, 'E5 paired seed contrast')
    return dict(seeds=seeds, correct_minus_zero=st.mean(zero), correct_minus_shuffle=st.mean(shuffle),
                zero_by_seed=zero, shuffle_by_seed=shuffle,
                all_seeds_correct_gt_zero=all(v > 0 for v in zero),
                all_seeds_correct_gt_shuffle=all(v > 0 for v in shuffle),
                scope='E5 frozen selected E4 checkpoint with repeated inference seeds; '
                      'not a causal re-test of every E6/E8 checkpoint')


def mechanism(selection, e4, e5, e6):
    u, a = selection['selected_uncertainty'], selection['selected_alpha']
    if u['variant'] not in MODES or a['tag'] not in TAGS or u['mode'] != MODES[u['variant']]:
        raise EvidenceError('Unknown selected uncertainty/local-alpha mechanism.')
    if (e6['selected_uncertainty'] != u['variant'] or e6['selected_alpha'] != a['tag'] or
            e5['provenance']['e4_uncertainty_mode'] != u['mode']):
        raise EvidenceError('E4/E5/E6 describe different mechanism choices.')
    rows = {row['model']: row for row in e4['ablation']}
    if len(e4['ablation']) != 5 or set(rows) != {'RGB', 'UDR-v1', 'U only', 'A only', 'U + A'}:
        raise EvidenceError('The required five-row E4 ablation is incomplete.')
    if rows['U + A']['uncertainty'] != u['variant'] or rows['U + A']['local_alpha'] != a['tag']:
        raise EvidenceError('E4 ablation does not test the selected U+A.')
    gains = [number(e4['per_dataset']['U + A'][ds]['delta_psnr']) for ds in DATASETS]
    close(st.mean(gains), rows['U + A']['average_delta_psnr'], 'E4 five-set equal-weight gain')
    close(a['lambda_a'], TAGS[a['tag']], 'E3 selected alpha regularization')
    rho, old = number(u['spearman']), number(selection['e0_uncertainty_spearman'])
    close(u['gain_over_e0'], rho - old, 'Selected uncertainty improvement')
    repaired = (rho > .2 and rho - old >= .05 and
                number(a['local_alpha_std']) > 0 and number(a['local_alpha_p95_minus_p05']) > 0)
    return dict(uncertainty=u['variant'], local_alpha=a['tag'], repaired=repaired,
        spearman=rho, spearman_gain=rho-old, e4_five_set_mean=st.mean(gains),
        e4_urban_delta=number(e4['per_dataset']['U + A']['Urban100']['delta_psnr']),
        e5_alpha_attribution='Legacy E5 reports record uncertainty mode, not the alpha regularization tag; '
            'the supplied E4 selection must belong to the tested E5 experiment. '
            'This report does not independently re-prove that alpha-training attribution.')


def decide(perf, causal, mech, policy):
    """Failure triggers have precedence; exact +0.05 belongs to A if all A
    conditions hold, otherwise the inclusive B band is checked. Gaps are
    REVIEW, not forced into a more convenient success/failure category.
    """
    mean = perf['five_set_mean']
    priority = [perf['datasets'][ds] for ds in ('Urban100', 'Manga109')]
    near_shuffle = (not above(abs(causal['correct_minus_shuffle']), policy['shuffle_equivalence_margin']) and
                    all(not above(abs(v), policy['shuffle_equivalence_margin']) for v in causal['shuffle_by_seed']))
    clear = (above(causal['correct_minus_zero'], policy['clear_causality_min']) and
             above(causal['correct_minus_shuffle'], policy['clear_causality_min']) and
             causal['all_seeds_correct_gt_zero'] and causal['all_seeds_correct_gt_shuffle'])
    urban = perf['datasets']['Urban100']
    persistent_urban = (mech['e4_urban_delta'] < 0 and all(v < 0 for v in urban['by_seed']) and
                        urban['ci95_psnr'][1] < 0)
    unstable = [ds for ds, item in perf['datasets'].items()
                if not item['stable_all_seeds'] or item['ci95_psnr'][0] <= 0]
    checks = dict(mechanisms_repaired=mech['repaired'], five_set_at_least_005=at_least(mean, .05),
        urban_positive=priority[0]['mean_delta_psnr'] > 0,
        manga_positive=priority[1]['mean_delta_psnr'] > 0,
        matched_seeds_stable=perf['five_set_stable'] and all(item['stable_all_seeds'] for item in priority),
        depth_causality_clear=clear, weak_gain_band=at_least(mean, .02) and not above(mean, .05),
        some_dataset_unstable=bool(unstable), near_zero_gain=not above(mean, policy['failure_gain_max']),
        urban_persistently_negative=persistent_urban, correct_approximately_shuffle=near_shuffle,
        shuffle_not_worse_all_seeds=all(v <= 0 for v in causal['shuffle_by_seed']))
    failures = [name for name in ('near_zero_gain', 'urban_persistently_negative',
        'correct_approximately_shuffle', 'shuffle_not_worse_all_seeds') if checks[name]]
    if not mech['repaired']:
        case, action, reasons = 'REVIEW', 'review_evidence', ['Repair uncertainty/local-alpha evidence before judging the repaired UDR direction.']
    elif failures:
        case, action, reasons = 'C', 'stop_current_udr_complexification_and_pivot', failures
    elif all(checks[name] for name in ('five_set_at_least_005', 'urban_positive', 'manga_positive',
                                      'matched_seeds_stable', 'depth_causality_clear')):
        case, action, reasons = 'A', 'continue', ['All operational Case A conditions hold.']
    elif checks['weak_gain_band'] and clear and unstable:
        case, action, reasons = 'B', 'pivot_geometry_representation', ['Weak positive gain with clear causality and dataset instability.']
    else:
        case, action, reasons = 'REVIEW', 'review_evidence', ['Evidence does not fully match any stated A/B/C condition.']
    return dict(case=case, label_zh=dict(A='成功：继续', B='弱成功：转向几何表示',
                C='失败：停止当前复杂化路线并转向', REVIEW='待复核：未完全匹配附件条件')[case],
                action=action, reasons=reasons, criteria=checks,
                unstable_datasets=unstable, next_steps=ROUTES.get(case, [
                    '先检查未满足判据及已有逐图/seed 证据，再确定下一项实验。']),
                statistical_note='A positive mean with a crossing-zero image interval is only a trend. '
                    'Image bootstrap is conditional on the observed training seeds; no new significance claim is inferred.',
                automation='Recommendations only: no training, architecture changes, data downloads or external messages are launched.')


def optional_scaling(paths):
    result = []
    for path in paths:
        item = read(path)
        if item['training_data'] != 'DF2K' or item['scale'] not in (4, 2, 3):
            raise EvidenceError('Optional E8 evidence has an unexpected data/scale label.')
        seeds = item['seeds']
        if len(seeds) not in (3, 5) or len(set(seeds)) != len(seeds):
            raise EvidenceError('Invalid E8 independent seed count.')
        values = [vector(item['per_dataset'][ds]['delta_psnr']['by_seed'], len(seeds)) for ds in DATASETS]
        for ds, per_seed in zip(DATASETS, values):
            summary_check(item['per_dataset'][ds]['delta_psnr'], per_seed, 'E8/' + ds)
        gain = [st.mean(v[i] for v in values) for i in range(len(seeds))]
        summary_check(item['five_set_delta_psnr'], gain, 'E8 five-set delta')
        result.append(dict(source=record(path), scale=item['scale'], training_data='DF2K',
            five_set_mean=st.mean(gain), five_set_std=st.stdev(gain),
            note='Separate scaling evidence; does not replace DIV2K x4 primary evidence or establish this checkpoint\'s Depth causality.'))
    if len({row['scale'] for row in result}) != len(result):
        raise EvidenceError('Duplicate optional E8 scale; do not select a favorable duplicate.')
    return result


def render(output, report):
    output = Path(output)
    write(output/'E9_decision.json', report)
    lines = ['# E9 继续 / 停止 / 转向决策', '',
             '状态：' + report['status'], 'Case：' + report.get('decision', {}).get('case', 'UNDETERMINED'), '']
    if report['status'] != 'complete':
        lines += ['当前证据不足或无效，不能判断研究方向成功或失败。', '', *report.get('issues', [])]
    else:
        decision = report['decision']
        lines += [decision['label_zh'], '行动：' + decision['action'], '', '依据：' + '; '.join(decision['reasons']), '',
            f"DIV2K ×4 五集等权 ΔPSNR：{report['performance']['five_set_mean']:+.6f} ± "
            f"{report['performance']['five_set_std']:.6f} dB。", '',
            '| Dataset | ΔPSNR mean±std | Positive seeds | 95% image CI | Evidence |',
            '|---|---:|---:|---|---|']
        for ds, item in report['performance']['datasets'].items():
            lines.append(f'| {ds} | {item["mean_delta_psnr"]:+.6f}±{item["std_delta_psnr"]:.6f} | '
                f'{item["positive_seeds"]}/{len(report["performance"]["seeds"])} | '
                f'{item["ci95_psnr"]} | {item["image_evidence"]} |')
        lines += ['', 'Depth 因果对照：' + json.dumps(report['causality'], ensure_ascii=False), '',
                  '下一步建议（不自动执行）：', '', *['- ' + value for value in decision['next_steps']], '',
                  decision['statistical_note'], '', 'E8 扩展证据：' + json.dumps(report['scaling'], ensure_ascii=False)]
    lines += ['', '数值解释策略：' + json.dumps(report['policy'], ensure_ascii=False),
              '精确阈值：A≥+0.05；B +0.02～+0.05。约+0.01、因果明显和近似相等的量化策略单独披露。',
              'E9 不训练模型，不自动实现转向路线。JSON 保存输入路径/SHA256、逐项判据与全部证据。', '']
    (output/'E9_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    rows = report.get('decision', {}).get('criteria', {})
    with (output/'E9_criteria.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(('criterion', 'value'))
        writer.writerows(rows.items())


def run(args):
    output = Path(args.output).resolve()
    report = dict(schema='E9-v1', status='insufficient_evidence', policy=dict(POLICY), sources={}, issues=[])
    try:
        policy = policy_values(read(args.policy) if args.policy else POLICY)
        report['policy'] = policy
        manifest = read(args.inputs)
        report['sources']['inputs'] = record(args.inputs)
        required = set(input_template()) - {'e8_metrics'}
        paths = {key: Path(manifest[key]) if Path(manifest[key]).is_absolute() else ROOT/manifest[key]
                 for key in required}
        extras = [Path(p) if Path(p).is_absolute() else ROOT/p for p in manifest.get('e8_metrics', [])]
        missing = [str(path) for path in list(paths.values()) + extras if not path.is_file()]
        if missing:
            report['issues'] = ['Missing measured evidence: ' + path for path in sorted(missing)]
        else:
            report['sources'].update({key: record(path) for key, path in paths.items()})
            if args.policy:
                report['sources']['policy'] = record(args.policy)
            e4, e5, e6, e7 = [read(paths[key]) for key in ('e4_report', 'e5_report', 'e6_report', 'e7_report')]
            perf = performance(e6, e7, csv_index(paths['e7_csv'], e6['seeds']))
            causal = causality(e5, csv_index(paths['e5_csv'], e5['seeds'], conditions=True))
            mech = mechanism(read(paths['selection']), e4, e5, e6)
            audit = 'report_hash_linkage_only; weight files not rehashed'
            if args.verify_checkpoints:
                records = [e6['checkpoints'][label][str(seed)] for label in ('rgb', 'udrv2') for seed in e6['seeds']]
                records.append(dict(path=e5['provenance']['e4_checkpoint'], sha256=e5['provenance']['checkpoint_sha256']))
                for item in records:
                    if digest(item['path']) != item['sha256']:
                        raise EvidenceError('A reported checkpoint changed: ' + item['path'])
                audit = 'checkpoint files rehashed; matches E5/E6/E7 reported SHA256'
            report.update(status='complete', performance=perf, causality=causal, mechanism=mech,
                checkpoint_audit=audit, scaling=optional_scaling(extras), decision=decide(perf, causal, mech, policy))
    except FileNotFoundError as exc:
        report['status'] = 'insufficient_evidence'
        report['issues'] = [str(exc)]
    except (EvidenceError, ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
        report['status'] = 'invalid_evidence'
        report['issues'] = [type(exc).__name__ + ': ' + str(exc)]
    render(output, report)
    print('E9:', report['status'], report.get('decision', {}).get('case', 'UNDETERMINED'))
    print('Saved report:', output)
    return 0 if report['status'] == 'complete' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', default=str(ROOT/'options/E9_inputs.json'))
    parser.add_argument('--policy', help='Only overrides the disclosed approximate-term policy.')
    parser.add_argument('--verify-checkpoints', action='store_true')
    parser.add_argument('--output', default=str(ROOT/'results/E9_decision'))
    parser.add_argument('--write-input-template', metavar='JSON')
    args = parser.parse_args()
    if args.write_input_template:
        if Path(args.write_input_template).exists():
            parser.error('Input template already exists; refusing to overwrite it.')
        write(args.write_input_template, input_template())
        return 0
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
