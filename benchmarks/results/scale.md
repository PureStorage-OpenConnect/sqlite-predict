# Scale study: distillation beyond 1500 rows

TabICL v2 as the teacher on the naturally large TabArena
datasets, at increasing training sizes. Questions: does student
retention hold, where does tuned xgboost catch the zero-shot
teacher, and does the in-context label leak grow with context
(the gap column; see permissive-teachers.md).

Device: mps with cpu fallback (per-cell `dev`).

| dataset | n_train | xgboost | TabICL | gbt<-TabICL | soft | soft-oof | leak gap | oof gap | teacher s | distill s | blob KB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| APSFailure | 1500 | 0.994 | 0.996 | 0.990 | 0.990 | 0.996 | -0.0040 | -0.0093 | 2.6 | 1.4 | 76.6 |
| Diabetes130US | 1500 | 0.906 | 0.916 | - | 0.916 | 0.916 | 0.0007 | 0.0007 | 2.2 | - | - |
| Diabetes130US | 9999 | 0.907 | 0.913 | - | 0.913 | 0.913 | -0.0000 | -0.0000 | 101.9 | - | - |
| GiveMeSomeCredit | 1500 | 0.922 | 0.936 | 0.932 | 0.926 | 0.930 | 0.0067 | -0.0113 | 1.2 | 0.3 | 97.1 |
| GiveMeSomeCredit | 9999 | 0.929 | 0.932 | 0.933 | 0.933 | 0.932 | 0.0210 | 0.0009 | 8.4 | 2.6 | 104.3 |
| SDSS17 | 1500 | 0.970 | 0.972 | 0.968 | 0.968 | 0.964 | 0.0167 | 0.0007 | 1.4 | 1.1 | 145.8 |
| SDSS17 | 9999 | 0.968 | 0.972 | 0.966 | 0.966 | 0.968 | 0.0205 | 0.0052 | 8.7 | 9.2 | 159.2 |
| SDSS17 | 49999 | 0.975 | 0.610 | - | 0.610 | - | -0.0000 | - | 118.9 | - | - |
| airline_satisfaction | 1500 | 0.910 | 0.928 | 0.908 | 0.904 | 0.906 | 0.0693 | -0.0073 | 1.6 | 0.5 | 103.4 |
| credit_card_default | 1500 | 0.810 | 0.836 | 0.798 | 0.810 | 0.838 | -0.0247 | -0.0153 | 1.7 | 0.9 | 93.5 |
| credit_card_default | 9999 | 0.805 | 0.771 | 0.776 | 0.776 | 0.776 | 0.0019 | 0.0054 | 11.7 | 7.4 | 96.2 |
| online_shoppers_intention | 1500 | 0.882 | 0.890 | 0.890 | 0.888 | 0.886 | 0.0507 | 0.0153 | 1.5 | 0.4 | 100.9 |
| taiwanese_bankruptcy | 1500 | 0.966 | 0.968 | 0.968 | 0.966 | 0.968 | 0.0040 | 0.0007 | 2.4 | 2.9 | 93.8 |
