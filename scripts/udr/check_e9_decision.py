"""E9 decision boundaries and complete synthetic report/CSV integration tests.

Temporary fixtures are explicitly synthetic and are never real SR results.
"""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.udr.e9_decision import (COUNTS, DATASETS, POLICY, EvidenceError,
    decide, input_template, policy_values, read, run, write)


def stats(values):
    return dict(mean=st.mean(values), std=st.stdev(values), by_seed=values)


def conditions(mean=.06):
    datasets = {ds: dict(mean_delta_psnr=mean, std_delta_psnr=.001,
        by_seed=[mean-.001, mean, mean+.001], ci95_psnr=[mean-.01, mean+.01],
        positive_seeds=3, stable_all_seeds=True, image_evidence='positive_ci') for ds in DATASETS}
    perf = dict(five_set_mean=mean, five_set_std=.001, five_set_by_seed=[mean]*3,
                five_set_stable=True, seeds=[10, 11, 12], datasets=datasets)
    causal = dict(correct_minus_zero=.05, correct_minus_shuffle=.03,
        shuffle_by_seed=[.029, .03, .031], zero_by_seed=[.049, .05, .051],
        all_seeds_correct_gt_zero=True, all_seeds_correct_gt_shuffle=True)
    return perf, causal, dict(repaired=True, e4_urban_delta=.01)


def decision_checks():
    for mean in (.05, .05 - 1e-14, .06):
        p, c, m = conditions(mean)
        assert decide(p, c, m, POLICY)['case'] == 'A'
    p, c, m = conditions(.03)
    p['datasets']['Set5']['ci95_psnr'] = [-.01, .07]
    assert decide(p, c, m, POLICY)['case'] == 'B'
    for mean in (.02, .02 - 1e-14, .05):
        p, c, m = conditions(mean)
        p['datasets']['Urban100']['stable_all_seeds'] = False
        assert decide(p, c, m, POLICY)['case'] == 'B'
    for mean in (-.01, 0., .01, .015):
        p, c, m = conditions(mean)
        assert decide(p, c, m, POLICY)['case'] == 'C'
    p, c, m = conditions()
    c.update(correct_minus_shuffle=.005, shuffle_by_seed=[.004, .005, .006])
    assert decide(p, c, m, POLICY)['case'] == 'C'
    # A zero mean caused by large opposing seed contrasts is not approximate equality.
    c.update(correct_minus_shuffle=0., shuffle_by_seed=[-.1, .2, -.1])
    assert decide(p, c, m, POLICY)['case'] == 'REVIEW'
    p, c, m = conditions()
    m['e4_urban_delta'] = -.01
    p['datasets']['Urban100'].update(by_seed=[-.01, -.02, -.03], ci95_psnr=[-.04, -.005])
    assert decide(p, c, m, POLICY)['case'] == 'C'
    for mean in (.018, .03, .06):
        p, c, m = conditions(mean)
        if mean == .06:
            p['five_set_stable'] = False
        assert decide(p, c, m, POLICY)['case'] == 'REVIEW'
    p, c, m = conditions(.01)
    m['repaired'] = False
    assert decide(p, c, m, POLICY)['case'] == 'REVIEW'
    try:
        policy_values(dict(POLICY, success_threshold=.01))
    except EvidenceError:
        pass
    else:
        raise AssertionError('Exact proposal thresholds became configurable.')
    print('PASS: Cases A/B/C, exact 0.02/0.05 boundaries, failure precedence and REVIEW gaps')


def fixtures(folder):
    seeds = [10, 11, 12]
    records = {}
    for label in ('rgb', 'udrv2'):
        records[label] = {}
        for seed in seeds:
            path = folder/f'{label}{seed}.pth'
            path.write_bytes(f'SYNTHETIC CHECKPOINT CONTENT {label} {seed}'.encode())
            records[label][str(seed)] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    e6 = dict(seeds=seeds, selected_uncertainty='U3', selected_alpha='a1e4',
              checkpoints=records, per_dataset={}, five_set={})
    e7 = dict(seeds=seeds, bootstrap_seed=2026, bootstrap_replicates=10000,
              datasets={}, checkpoint_sha256={label: {seed: item['sha256'] for seed, item in values.items()}
                                            for label, values in records.items()})
    e5 = dict(seeds=seeds, complete_five_set=True, datasets=list(DATASETS), per_dataset={}, five_set={},
        provenance=dict(e4_uncertainty_mode='learned_error', e4_checkpoint=records['udrv2']['10']['path'],
                        checkpoint_sha256=records['udrv2']['10']['sha256']))
    e7rows, e5rows = [], []
    streams = np.random.SeedSequence(2026).spawn(5)
    for ds, stream in zip(DATASETS, streams):
        n = COUNTS[ds]
        for image in range(n):
            for seed in seeds:
                b = 30 + seed / 1000 + image / 10000
                gain = .06 + image / 100000
                e7rows.append(dict(dataset=ds, image_id=f'i{image}', seed=seed,
                    baseline_psnr=b, udrv2_psnr=b+gain, delta_psnr=(b+gain)-b,
                    baseline_ssim=.9, udrv2_ssim=.901, delta_ssim=.001))
                for condition in range(8):
                    e5rows.append(dict(dataset=ds, image_id=f'i{image}', seed=seed,
                        condition=f'D{condition}', psnr=b + .1 - condition * .02, ssim=.9))
        per = {}
        for label, prefix in (('rgb', 'baseline'), ('udrv2', 'udrv2')):
            per[label] = dict(metrics={metric: stats([st.mean(row[prefix+'_'+metric] for row in e7rows
                if row['dataset'] == ds and row['seed'] == seed) for seed in seeds]) for metric in ('psnr', 'ssim')})
        gains = [st.mean(row['delta_psnr'] for row in e7rows if row['dataset'] == ds and row['seed'] == seed) for seed in seeds]
        per['udrv2']['delta_psnr_vs_rgb'] = stats(gains)
        e6['per_dataset'][ds] = per
        # Lexicographic image order must match the production E7 worker.
        images = sorted({row['image_id'] for row in e7rows if row['dataset'] == ds})
        values = np.array([st.mean(row['delta_psnr'] for row in e7rows if row['dataset'] == ds and row['image_id'] == image)
                           for image in images])
        rng = np.random.default_rng(stream)
        ci = np.quantile(values[rng.integers(0, n, (10000, n))].mean(axis=1), [.025, .975])
        e7['datasets'][ds] = dict(images=n, seeds=seeds, bootstrap_replicates=10000,
                                  mean_delta_psnr=float(values.mean()), ci95_psnr=ci.tolist())
        e5['per_dataset'][ds] = {}
        for condition in range(8):
            label = f'D{condition}'
            e5['per_dataset'][ds][label] = dict(images=n,
                metrics={metric: stats([st.mean(row[metric] for row in e5rows
                    if row['dataset'] == ds and row['seed'] == seed and row['condition'] == label) for seed in seeds])
                         for metric in ('psnr', 'ssim')})
    e6['five_set']['udrv2'] = dict(delta_psnr_vs_rgb=stats([
        st.mean(e6['per_dataset'][ds]['udrv2']['delta_psnr_vs_rgb']['by_seed'][i] for ds in DATASETS) for i in range(3)]))
    for condition in range(8):
        label = f'D{condition}'
        values = [st.mean(e5['per_dataset'][ds][label]['metrics']['psnr']['by_seed'][i] for ds in DATASETS) for i in range(3)]
        deltas = [st.mean(e5['per_dataset'][ds]['D0']['metrics']['psnr']['by_seed'][i] -
                         e5['per_dataset'][ds][label]['metrics']['psnr']['by_seed'][i] for ds in DATASETS) for i in range(3)]
        e5['five_set'][label] = dict(psnr_mean=st.mean(values), correct_minus_condition_psnr=st.mean(deltas), paired_delta_by_seed=deltas)
    selection = dict(e0_uncertainty_spearman=.1,
        selected_uncertainty=dict(variant='U3', mode='learned_error', spearman=.31, gain_over_e0=.21),
        selected_alpha=dict(tag='a1e4', lambda_a=1e-4, local_alpha_std=.003, local_alpha_p95_minus_p05=.006))
    e4 = dict(ablation=[dict(model=label, uncertainty='U3', local_alpha='a1e4', average_delta_psnr=.06)
                        for label in ('RGB', 'UDR-v1', 'U only', 'A only', 'U + A')],
              per_dataset={'U + A': {ds: dict(delta_psnr=.06) for ds in DATASETS}})
    reports = dict(selection=selection, e4_report=e4, e5_report=e5, e6_report=e6, e7_report=e7)
    manifest = {}
    for key, data in reports.items():
        path = folder/(key+'.json')
        write(path, data)
        manifest[key] = str(path)
    for key, rows in (('e5_csv', e5rows), ('e7_csv', e7rows)):
        path = folder/(key+'.csv')
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest[key] = str(path)
    manifest['e8_metrics'] = []
    write(folder/'inputs.json', manifest)
    return manifest


def integration_checks():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        manifest = fixtures(folder)
        args = SimpleNamespace(inputs=str(folder/'inputs.json'), policy=None, verify_checkpoints=True,
                               output=str(folder/'output'))
        assert run(args) == 0
        report = read(folder/'output/E9_decision.json')
        assert report['decision']['case'] == 'A'
        assert report['scaling'] == []  # E8 is not required before a stop/pivot decision.
        assert (folder/'output/E9_summary.md').is_file()
        assert (folder/'output/E9_criteria.csv').is_file()
        # Repeated runs must be reproducible, with no wall-clock-dependent fields.
        first = (folder/'output/E9_decision.json').read_bytes()
        assert run(args) == 0
        assert (folder/'output/E9_decision.json').read_bytes() == first
        e7 = read(manifest['e7_report'])
        e7['datasets']['Set5']['ci95_psnr'][0] += .001
        write(manifest['e7_report'], e7)
        assert run(args) == 2
        assert read(folder/'output/E9_decision.json')['status'] == 'invalid_evidence'
        manifest = fixtures(folder)
        cp = Path(read(manifest['e6_report'])['checkpoints']['udrv2']['10']['path'])
        cp.write_bytes(b'changed checkpoint')
        assert run(args) == 2
        manifest = fixtures(folder)
        Path(manifest['e5_csv']).unlink()
        assert run(args) == 2
        report = read(folder/'output/E9_decision.json')
        assert report['status'] == 'insufficient_evidence' and 'decision' not in report
        manifest = fixtures(folder)
        with Path(manifest['e7_csv']).open('a', encoding='utf-8') as handle:
            handle.write('Set5,i0,10,30,30.1,0.1,0.9,0.901,0.001\n')
        assert run(args) == 2
        manifest = fixtures(folder)
        e6 = read(manifest['e6_report'])
        e6['checkpoints']['rgb']['11'] = deepcopy(e6['checkpoints']['rgb']['10'])
        write(manifest['e6_report'], e6)
        e7 = read(manifest['e7_report'])
        e7['checkpoint_sha256']['rgb']['11'] = e7['checkpoint_sha256']['rgb']['10']
        write(manifest['e7_report'], e7)
        assert run(args) == 2
        manifest = fixtures(folder)
        e5 = read(manifest['e5_report'])
        e5['complete_five_set'] = False
        write(manifest['e5_report'], e5)
        assert run(args) == 2
        manifest = fixtures(folder)
        e8path = folder/'E8_metrics.json'
        write(e8path, dict(scale=4, training_data='DF2K', seeds=[10,11,12],
            per_dataset={ds: dict(delta_psnr=stats([.079,.08,.081])) for ds in DATASETS},
            five_set_delta_psnr=stats([.079,.08,.081])))
        manifest['e8_metrics'] = [str(e8path)]
        write(folder/'inputs.json', manifest)
        assert run(args) == 0
        result = read(folder/'output/E9_decision.json')
        assert result['decision']['case'] == 'A'
        assert result['scaling'][0]['training_data'] == 'DF2K'
        assert result['performance']['five_set_mean'] < .08
    print('PASS: complete full-size E5/E7 CSV reconciliation, bootstrap reproduction, deterministic outputs, '
          'checkpoint tampering, missing input and duplicate rejection')


if __name__ == '__main__':
    decision_checks()
    integration_checks()
