"""DATA 709: repeated N=400 audit of frozen readmission risk scores.

No model fitting, plotting, or dependency on the original project directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CAPACITIES = [5, 10, 15, 20, 25, 30, 40, 50]
VARIANTS = ['primary', 'S02_home_expanded', 'S01_hba1c_high', 'S05_surgery_broad']
COUNTS = ['N', 'positives', 'negatives', 'selected_N', 'TP', 'FP', 'TN', 'FN']
RATES = ['TPR', 'FNR', 'FPR', 'selection_rate']
COMPARISONS = [
    ('race', 'African American', 'Caucasian', 'African American minus Caucasian'),
    ('insurance', 'Medicare', 'Medicaid', 'Medicare minus Medicaid'),
    ('intersection', 'African American x Medicare', 'Caucasian x Medicare',
     'African American minus Caucasian within Medicare'),
    ('intersection', 'African American x Medicaid', 'Caucasian x Medicaid',
     'African American minus Caucasian within Medicaid'),
]


def groups(frame):
    yield 'overall', 'All', np.ones(len(frame), dtype=bool)
    for race in sorted(frame.race_audit.unique()):
        yield 'race', race, frame.race_audit.eq(race).to_numpy()
    for payer in ['Medicare', 'Medicaid']:
        yield 'insurance', payer, frame.insurance.eq(payer).to_numpy()
    for race in sorted(frame.race_audit.unique()):
        for payer in ['Medicare', 'Medicaid']:
            mask = frame.race_audit.eq(race) & frame.insurance.eq(payer)
            yield 'intersection', race + ' x ' + payer, mask.to_numpy()


def metrics(y, selected, weight):
    positive = float(weight[y].sum())
    negative = float(weight[~y].sum())
    tp = float(weight[y & selected].sum())
    fp = float(weight[~y & selected].sum())
    fn, tn = positive - tp, negative - fp
    n = positive + negative
    return dict(N=n, positives=positive, negatives=negative, selected_N=tp+fp,
                TP=tp, FP=fp, TN=tn, FN=fn,
                TPR=tp/positive if positive else np.nan,
                FNR=fn/positive if positive else np.nan,
                FPR=fp/negative if negative else np.nan,
                selection_rate=(tp+fp)/n if n else np.nan)


def summarize(data, keys, columns, is_gap=False):
    rows = []
    for key, block in data.groupby(keys, sort=True, dropna=False):
        for metric in columns:
            values = block[metric].to_numpy(dtype=float)
            valid = values[np.isfinite(values)]
            n, total = len(valid), len(values)
            row = dict(zip(keys, key))
            row.update(metric=metric, repetitions=total, valid_repetitions=n,
                       undefined_repetitions=total-n, valid_proportion=n/total,
                       mean=float(valid.mean()) if n else np.nan,
                       median=float(np.median(valid)) if n else np.nan,
                       sd=float(valid.std(ddof=1)) if n > 1 else np.nan,
                       p025=float(np.quantile(valid, .025)) if n else np.nan,
                       p975=float(np.quantile(valid, .975)) if n else np.nan)
            if is_gap:
                row.update(positive_proportion_all=float((valid > 0).sum()/total),
                           negative_proportion_all=float((valid < 0).sum()/total),
                           zero_proportion_all=float((valid == 0).sum()/total),
                           undefined_proportion_all=(total-n)/total,
                           positive_proportion_valid=float((valid > 0).mean()) if n else np.nan,
                           negative_proportion_valid=float((valid < 0).mean()) if n else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'results')
    parser.add_argument('--design', choices=['primary', 'near_proportional', 'both'],
                        default='primary', help='Each design uses the same 200 prespecified seeds.')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Output directory is not empty. Choose a new --output directory.')
    out.mkdir(parents=True, exist_ok=True)
    paths = [HERE/name for name in ['frozen_predictions.csv', 'allocations.csv', 'seeds_200.csv']]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    frame = pd.read_csv(paths[0], keep_default_na=False).sort_values('patient_nbr').reset_index(drop=True)
    allocations = pd.read_csv(paths[1])
    seeds = pd.read_csv(paths[2], dtype={'seed': 'uint64'}).sort_values('replicate')
    assert len(frame) == 21248 and frame.patient_nbr.is_unique
    assert frame.outcome_30d.isin([0, 1]).all() and frame.outcome_30d.sum() == 1968
    assert frame.race_audit.isin(['African American', 'Caucasian', 'Asian', 'Hispanic', 'Other']).all()
    assert frame.payer_code.isin(['MC', 'MD']).all() and frame.tie_priority.is_unique
    assert seeds.replicate.tolist() == list(range(1, 201)) and seeds.seed.is_unique
    frame['insurance'] = frame.payer_code.map({'MC': 'Medicare', 'MD': 'Medicaid'})
    for variant in VARIANTS:
        probability = frame[variant].to_numpy()
        assert np.isfinite(probability).all() and ((probability > 0) & (probability < 1)).all()
        # Restore the fixed 0.0001 logit grid to preserve ties after CSV export.
        units = np.rint(np.log(probability/(1-probability))*10000).astype(np.int64)
        assert np.max(abs(1/(1+np.exp(-units/10000))-probability)) < 1e-11
        frame['units_'+variant] = units
    allocations['inclusion_probability'] = allocations.sample_n/allocations.population_N
    allocations['sampling_fraction'] = allocations.inclusion_probability
    allocations['design_weight'] = 1/allocations.inclusion_probability
    designs = ['primary', 'near_proportional'] if args.design == 'both' else [args.design]
    members, result_rows, gap_rows = [], [], []
    for design in designs:
        allocation = allocations[allocations.design.eq(design)].sort_values(['race_audit', 'payer_code'])
        assert len(allocation) == 10 and allocation.sample_n.sum() == 400
        pools = []
        for row in allocation.itertuples(index=False):
            pool = np.flatnonzero((frame.race_audit.eq(row.race_audit) & frame.payer_code.eq(row.payer_code)).to_numpy())
            assert len(pool) == row.population_N and 0 < row.sample_n <= len(pool)
            pools.append((row, pool))
        for seed_row in seeds.itertuples(index=False):
            rng = np.random.Generator(np.random.PCG64(int(seed_row.seed)))
            indices, weights = [], []
            for row, pool in pools:
                draw = rng.choice(pool, size=int(row.sample_n), replace=False)
                indices.extend(draw)
                weights.extend([row.design_weight]*len(draw))
            assert len(indices) == len(set(indices)) == 400
            sample = frame.iloc[indices].reset_index(drop=True)
            weights = np.asarray(weights)
            assert np.isclose(weights.sum(), len(frame))
            y = sample.outcome_30d.to_numpy(dtype=bool)
            membership = sample[['patient_nbr', 'encounter_id', 'race_audit', 'payer_code', 'insurance', 'outcome_30d', 'tie_priority']].copy()
            membership['design'] = design
            membership['replicate'] = seed_row.replicate
            membership['seed'] = int(seed_row.seed)
            membership['design_weight'] = weights
            membership['inclusion_probability'] = 1/weights
            members.append(membership)
            subgroup_masks = list(groups(sample))
            # The four reconstructions use identical primary-design sample IDs.
            variants = VARIANTS if design == 'primary' else ['primary']
            for variant in variants:
                order = np.lexsort((sample.tie_priority.to_numpy(), -sample['units_'+variant].to_numpy()))
                for capacity in CAPACITIES:
                    selected = np.zeros(400, dtype=bool)
                    selected[order[:4*capacity]] = True
                    cache = {}
                    base = dict(design=design, replicate=seed_row.replicate,
                                variant=variant, capacity_pct=capacity, allocation_N=400)
                    for kind, name, mask in subgroup_masks:
                        for estimate, w in [('sample', np.ones(mask.sum())), ('design_weighted', weights[mask])]:
                            values = metrics(y[mask], selected[mask], w)
                            cache[estimate, kind, name] = values
                            result_rows.append(dict(base, group_type=kind, group=name, estimate=estimate,
                                                    observed_N=int(mask.sum()), observed_positives=int(y[mask].sum()),
                                                    observed_negatives=int((~y[mask]).sum()), **values))
                    for estimate in ['sample', 'design_weighted']:
                        for kind, group1, group0, label in COMPARISONS:
                            one, zero = cache[estimate, kind, group1], cache[estimate, kind, group0]
                            # Differences are within each repetition, before aggregation.
                            gap_rows.append(dict(base, estimate=estimate, comparison=label,
                                                 group1=group1, group0=group0,
                                                 **{rate+'_gap_pp': 100*(one[rate]-zero[rate]) for rate in RATES}))
            if seed_row.replicate % 50 == 0:
                print(f'{design}: {seed_row.replicate}/200', flush=True)
    results, gaps = pd.DataFrame(result_rows), pd.DataFrame(gap_rows)
    group_summary = summarize(results, ['design', 'variant', 'estimate', 'capacity_pct', 'group_type', 'group'], COUNTS+RATES)
    gap_summary = summarize(gaps, ['design', 'variant', 'estimate', 'capacity_pct', 'comparison'], [rate+'_gap_pp' for rate in RATES], True)
    # Compare each alternative with primary on the same repetition and capacity.
    primary = gaps[gaps.design.eq('primary') & gaps.variant.eq('primary')]
    paired = gaps[gaps.design.eq('primary') & gaps.variant.ne('primary')].merge(
        primary, on=['design', 'replicate', 'estimate', 'capacity_pct', 'comparison'],
        suffixes=('', '_primary'), validate='many_to_one')
    paired['FNR_gap_change_pp'] = paired.FNR_gap_pp - paired.FNR_gap_pp_primary
    paired['both_valid'] = paired.FNR_gap_pp.notna() & paired.FNR_gap_pp_primary.notna()
    paired['strict_sign_reversal'] = paired.both_valid & (paired.FNR_gap_pp*paired.FNR_gap_pp_primary < 0)
    for name, table in [('sample_membership', pd.concat(members, ignore_index=True)),
                        ('group_metrics', results), ('group_gaps', gaps),
                        ('group_stability', group_summary), ('gap_stability', gap_summary),
                        ('paired_sensitivity', paired),
                        ('sampling_allocation', allocations[allocations.design.isin(designs)])]:
        table.to_csv(out/(name+'.csv'), index=False, float_format='%.12g')
    assert results.loc[results.positives.eq(0), 'FNR'].isna().all()
    assert results[results.group_type.eq('overall') & results.estimate.eq('sample')].selected_N.eq(
        4*results[results.group_type.eq('overall') & results.estimate.eq('sample')].capacity_pct).all()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.name] for p in paths)
    manifest = dict(repetitions_per_design=200, sample_N=400, frame_N=len(frame),
                    designs=designs, capacities_pct=CAPACITIES, model_refitted=False,
                    interval='Empirical sampling-stability interval, not confidence interval',
                    input_sha256=hashes, python=platform.python_version(),
                    numpy=np.__version__, pandas=pd.__version__)
    (out/'run.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved results to {out}')


if __name__ == '__main__':
    main()
