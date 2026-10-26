# Part 2 validation evidence (copied from output/emerging/, which is not versioned)

One folder per run. `blend_weights_<line>.json` holds the fold scores (time fold 2023->2024, grouped 5-fold on policy_id), the searched and guarded blend weights, re-balance factors, refit iteration counts and, from C044 on, the test-mix and new-business-only views.

| Folder | Run | CHANGELOG |
|---|---|---|
| emerging_only | emerging data only, five models | C036 |
| pooled_w0.3 / pooled_w1 | developed rows added at weight 0.3 / 1.0 (GAM excluded) | C038, C040 |
| pooled_w0.3_glmx | GLM with market interaction terms (contents; rejected) | C042 |
| pooled_w0.3_nb / pooled_w0.3_ie | new-business-weighted training / emerging-only early stopping (contents; rejected) | C045 |
| xrun, xrun3, xrun_c044_contents_2run, xrun_c045_motor_noglm | cross-run blend searches | C041, C044 |
| views_c044 | four validation views for the P2-S02/S03/S04/S05 recipes | C044 |

`cv_emerging_trees_pooled_w0.3.md`: C039 tree search (nothing adopted). `check_dev_target.log`: developed path byte-identical after C034-C040. `build_target.log`: emerging target reconciliation and reporting-curve comparison (C037).
Submission files: `submissions/part2_candidate_P2-S12_*.zip` (public 0.15474) and `submissions/part2_hedge_P2-S11_*.zip` (0.15434); all scores in `docs/SUBMISSIONS.md`.
