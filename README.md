# DATA 709: Diabetes Readmission Fairness Audit

## Project question

How do follow-up thresholds in a reconstructed diabetes readmission model shape error burdens across race, insurance, and their intersections?

## Data and method

This package reproduces the 200-repetition audit from frozen patient predictions. The sampling frame contains 21,248 patients with recorded race, payer code MC or MD, and a valid outcome and score. MC and MD are interpreted as Medicare and Medicaid using the approved Fairlearn-informed mapping. All five recorded race groups remain in the frame.

Each repetition draws 400 distinct patients within race × payer strata using the original allocations and seeds. Patients can appear in different repetitions. There is no outcome enrichment. Patients are ranked together within each sample, and the top 5%, 10%, 15%, 20%, 25%, 30%, 40%, and 50% receive simulated follow-up.

The primary analysis uses the fixed, approximate Strack logistic reconstruction. The model is not fitted or recalibrated here. This package starts from frozen scores; it does not rebuild the cohort or model from the original UCI data.

## Run

Use Python 3.12 (the original run used 3.12.14). From this folder:

```bash
python -m pip install -r requirements.txt
python audit.py
```

Results are written to `results/`. Existing results are protected; choose a new folder for another run:

```bash
python audit.py --output results_repeat
```

To also reproduce the separate near-proportional allocation sensitivity:

```bash
python audit.py --design both --output results_both
```

Each design has 200 repetitions. The four model specifications share the same primary samples; they are not additional sampling repetitions.

## Files and results


Outputs include sampled patient IDs and weights, subgroup event counts and confusion matrices, rates, within-repetition gaps, stability summaries, and paired reconstruction differences. Individual rates are proportions; gaps are percentage points. Racial gaps are African American minus Caucasian, and insurance gaps are Medicare minus Medicaid. FNR is undefined when there are no positive outcomes. Undefined values remain blank in CSVs.

The percentile ranges describe sampling stability, not formal population confidence intervals. Weighted rows are supplementary descriptions under sample-level allocation policies. They do not represent excluded patients or recreate a population-wide ranking. Repeated samples overlap, and recorded readmission is an imperfect proxy for clinical need. Observed gaps do not establish discrimination or causal effects. No plots are generated.
