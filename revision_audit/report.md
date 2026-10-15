# Evidence audit before reframing

## Existing benchmark results (complete datasets only)

| noise_rate | classifier | method | datasets | gmean | auc | ap |
| --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | lr | adasyn | 5 | 0.897010 | 0.946338 | 0.861622 |
| 0.000000 | lr | borderline | 6 | 0.913582 | 0.953094 | 0.878143 |
| 0.000000 | lr | no_downweight | 6 | 0.892047 | 0.954393 | 0.888853 |
| 0.000000 | lr | none | 6 | 0.875982 | 0.955112 | 0.885057 |
| 0.000000 | lr | proposed | 6 | 0.895319 | 0.954384 | 0.890840 |
| 0.000000 | lr | ros | 6 | 0.915013 | 0.955782 | 0.887922 |
| 0.000000 | lr | smote | 6 | 0.916389 | 0.954174 | 0.888951 |
| 0.000000 | lr | wrnd_author | 3 | 0.928712 | 0.958260 | 0.842372 |
| 0.000000 | rf | adasyn | 5 | 0.917018 | 0.971164 | 0.899166 |
| 0.000000 | rf | borderline | 6 | 0.921531 | 0.978295 | 0.920987 |
| 0.000000 | rf | no_downweight | 6 | 0.915279 | 0.979418 | 0.929605 |
| 0.000000 | rf | none | 6 | 0.902885 | 0.979115 | 0.935004 |
| 0.000000 | rf | proposed | 6 | 0.922773 | 0.979468 | 0.926162 |
| 0.000000 | rf | ros | 6 | 0.927943 | 0.977365 | 0.920440 |
| 0.000000 | rf | smote | 6 | 0.930343 | 0.979041 | 0.923340 |
| 0.000000 | rf | wrnd_author | 3 | 0.871494 | 0.971125 | 0.874005 |
| 0.000000 | svm | adasyn | 5 | 0.929586 | 0.975243 | 0.904806 |
| 0.000000 | svm | borderline | 6 | 0.929246 | 0.977988 | 0.916787 |
| 0.000000 | svm | no_downweight | 6 | 0.927567 | 0.979312 | 0.923204 |
| 0.000000 | svm | none | 6 | 0.905253 | 0.976038 | 0.923070 |
| 0.000000 | svm | proposed | 6 | 0.925157 | 0.978733 | 0.922827 |
| 0.000000 | svm | ros | 6 | 0.939048 | 0.980033 | 0.926190 |
| 0.000000 | svm | smote | 6 | 0.934202 | 0.979902 | 0.927624 |
| 0.000000 | svm | wrnd_author | 3 | 0.903394 | 0.974973 | 0.872866 |
| 0.100000 | lr | adasyn | 6 | 0.884789 | 0.948060 | 0.866738 |
| 0.100000 | lr | borderline | 6 | 0.871501 | 0.943164 | 0.855637 |
| 0.100000 | lr | no_downweight | 6 | 0.909028 | 0.952648 | 0.886762 |
| 0.100000 | lr | none | 6 | 0.846680 | 0.946568 | 0.874048 |
| 0.100000 | lr | proposed | 6 | 0.908134 | 0.950895 | 0.882987 |
| 0.100000 | lr | ros | 6 | 0.898045 | 0.949935 | 0.871171 |
| 0.100000 | lr | smote | 6 | 0.901161 | 0.948747 | 0.877221 |
| 0.100000 | lr | wrnd_author | 3 | 0.918757 | 0.954652 | 0.826280 |
| 0.100000 | rf | adasyn | 6 | 0.909987 | 0.958670 | 0.889123 |
| 0.100000 | rf | borderline | 6 | 0.904047 | 0.957975 | 0.885517 |
| 0.100000 | rf | no_downweight | 6 | 0.916990 | 0.972373 | 0.920595 |
| 0.100000 | rf | none | 6 | 0.897067 | 0.966737 | 0.915533 |
| 0.100000 | rf | proposed | 6 | 0.923299 | 0.972879 | 0.919171 |
| 0.100000 | rf | ros | 6 | 0.904626 | 0.961820 | 0.898874 |
| 0.100000 | rf | smote | 6 | 0.921353 | 0.965281 | 0.907462 |
| 0.100000 | rf | wrnd_author | 3 | 0.886130 | 0.957687 | 0.859227 |
| 0.100000 | svm | adasyn | 6 | 0.919247 | 0.966704 | 0.892245 |
| 0.100000 | svm | borderline | 6 | 0.897455 | 0.953920 | 0.863187 |
| 0.100000 | svm | no_downweight | 6 | 0.927056 | 0.979395 | 0.921554 |
| 0.100000 | svm | none | 6 | 0.901025 | 0.965451 | 0.913485 |
| 0.100000 | svm | proposed | 6 | 0.925930 | 0.976692 | 0.914544 |
| 0.100000 | svm | ros | 6 | 0.931564 | 0.975951 | 0.910473 |
| 0.100000 | svm | smote | 6 | 0.931987 | 0.976640 | 0.913145 |
| 0.100000 | svm | wrnd_author | 3 | 0.907422 | 0.971076 | 0.861862 |

## Synthetic ranking metrics

| mechanism | method | gmean | auc | ap | synthetic |
| --- | --- | --- | --- | --- | --- |
| clean | adasyn | 0.815022 | 0.887019 | 0.605558 | 1589.685185 |
| clean | borderline | 0.816483 | 0.886735 | 0.605139 | 1594.000000 |
| clean | no_downweight | 0.789176 | 0.886595 | 0.605019 | 1594.000000 |
| clean | none | 0.569880 | 0.887193 | 0.606158 | 0.000000 |
| clean | proposed | 0.792104 | 0.886585 | 0.604961 | 1594.000000 |
| clean | smote | 0.820424 | 0.887091 | 0.605853 | 1594.000000 |
| dispersed | adasyn | 0.744348 | 0.884176 | 0.598824 | 1226.112346 |
| dispersed | borderline | 0.755150 | 0.882678 | 0.594745 | 1222.444444 |
| dispersed | no_downweight | 0.798220 | 0.884881 | 0.601559 | 1222.444444 |
| dispersed | none | 0.433578 | 0.885442 | 0.602886 | 0.000000 |
| dispersed | proposed | 0.799686 | 0.884814 | 0.601319 | 1222.444444 |
| dispersed | smote | 0.773960 | 0.884619 | 0.600814 | 1222.444444 |
| localized | adasyn | 0.792705 | 0.886925 | 0.605547 | 1220.691358 |
| localized | borderline | 0.796875 | 0.886608 | 0.604809 | 1222.444444 |
| localized | no_downweight | 0.797495 | 0.884646 | 0.602441 | 1222.444444 |
| localized | none | 0.433207 | 0.886539 | 0.604989 | 0.000000 |
| localized | proposed | 0.793682 | 0.884078 | 0.602024 | 1222.444444 |
| localized | smote | 0.772870 | 0.886291 | 0.604292 | 1222.444444 |

## Realized sample budgets

| study | pairs | equal | proposed_fewer | proposed_more | min_difference | max_difference |
| --- | --- | --- | --- | --- | --- | --- |
| grid | 1890 | 1890 | 0 | 0 | 0.000000 | 0.000000 |
| stress | 510 | 510 | 0 | 0 | 0.000000 | 0.000000 |
| benchmark | 540 | 507 | 33 | 0 | -125.000000 | 0.000000 |

## Seed audit

{
  "development_evaluation_seed_intersection": [],
  "grid_reused_role_seed_groups": 120,
  "note": "Disjoint RNG seeds do not make grid conditions independent. Noise levels share data deliberately; some distinct IR/separation cells also reuse RNG seeds. Different configurations transform those streams differently. No claim of preregistration follows from this audit."
}

## Inference limits

Six nonzero pairs give a minimum exact two-sided Wilcoxon p=2/2^6=0.03125 when all differences favor one method and there are no ties. This cannot pass the first Holm threshold in families of 6, 12 or 42 (0.00833, 0.00417, 0.00119). Minimum nonzero dataset counts permitting any rejection in those families under ideal conditions are 8, 9 and 11. These are feasibility bounds, not power or sample-size targets. Increasing datasets must broaden a predefined target population, not chase significance. Bayesian analysis does not create evidence and also needs assumptions and sensitivity analysis.

The new ROC-AUC/AP analysis is exploratory, with one conservative 84-comparison Holm family. Do not imply it was a prespecified primary family.

## What existing evidence does not answer

The separate Gaussian threshold control covers separation 2, IR about 10 and 10% corruption only; it cannot show that the 39/63 grid wins generally disappear after threshold tuning. Raw-count equality does not equal matched effective training weights. Corruption concentrated at class means is a structured stress test, not an established realistic noise model. Dropping downweighting now is a post-result method revision and requires fresh confirmation. Development/test historical access cannot be proven from RNG code alone.
