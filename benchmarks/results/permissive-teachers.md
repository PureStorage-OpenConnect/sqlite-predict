# Permissive tabular teachers: TabICL and Mitra

Every teacher in this table may be distilled and the student
shipped commercially: TabICL checkpoints are BSD-3-Clause, Mitra
is Apache-2.0, TabPFN-2 is under the Prior Labs License
(distillation permitted with attribution). Same datasets, splits,
and caps as `tabarena-full.md`; xgboost/TabFM/TabPFN columns come
from the other campaigns.

Device: mps. First calls include one-time weight download.

| dataset | task | xgboost | TabPFN-2 | TabICL | Mitra | gbt<-TabICL | soft<-TabICL | soft-oof<-TabICL | gbt<-Mitra | s/call (TabICL) |
|---|---|---|---|---|---|---|---|---|---|---|
| blood-transfusion | cls | 0.711 | 0.765 | 0.775 | 0.775 | 0.765 | 0.759 | 0.759 | 0.765 | 1.6 |
| diabetes | cls | 0.740 | 0.750 | 0.760 | 0.750 | 0.766 | 0.740 | 0.760 | 0.729 | 1.03 |
| anneal | cls | 0.987 | 0.991 | 0.987 | 0.978 | 0.987 | 0.987 | 0.982 | 0.982 | 1.2 |
| QSAR_fish_toxicity | reg | 0.949 | 0.918 | 0.907 | 0.905 | 0.912 | - | - | 0.915 | 59.76 |
| credit-g | cls | 0.752 | 0.784 | 0.760 | 0.764 | 0.728 | 0.736 | 0.748 | 0.788 | 1.06 |
| maternal_health_risk | cls | 0.807 | 0.831 | 0.843 | 0.780 | 0.787 | 0.756 | 0.705 | 0.748 | 0.75 |
| concrete_compressive_strength | reg | 5.047 | 4.198 | 4.115 | 5.002 | 4.956 | - | - | 5.636 | 0.83 |
| qsar-biodeg | cls | 0.871 | 0.890 | 0.886 | 0.871 | 0.845 | 0.841 | 0.856 | 0.837 | 1.62 |
| healthcare_insurance_expenses | reg | 5217.371 | 4829.207 | 4458.266 | 4550.568 | 4458.082 | - | - | 4530.398 | 0.93 |
| website_phishing | cls | 0.885 | 0.909 | 0.920 | 0.888 | 0.894 | 0.888 | 0.879 | 0.876 | 1.01 |
| Fitness_Club | cls | 0.747 | 0.781 | 0.784 | 0.781 | 0.781 | 0.781 | 0.784 | 0.787 | 0.91 |
| airfoil_self_noise | reg | 1.613 | 1.092 | 1.027 | 1.653 | 2.229 | - | - | 2.318 | 0.99 |
| used-Fiat-500 | reg | 830.136 | 803.507 | 784.192 | 841.820 | 794.371 | - | - | 835.630 | 1.1 |
| MIC | cls | 0.848 | 0.867 | 0.861 | 0.856 | 0.843 | 0.851 | 0.859 | 0.853 | 1.89 |
| Is-this-a-good-customer | cls | 0.880 | 0.893 | 0.893 | 0.893 | 0.891 | 0.888 | 0.891 | 0.893 | 1.12 |
| Marketing_Campaign | cls | 0.869 | 0.891 | 0.893 | 0.893 | 0.864 | 0.891 | 0.885 | 0.880 | 1.26 |
| hazelnut-contaminant | cls | 0.901 | 0.957 | 0.965 | 0.928 | 0.891 | 0.883 | 0.899 | 0.885 | 1.7 |
| seismic-bumps | cls | 0.931 | 0.936 | 0.936 | 0.936 | - | 0.936 | 0.936 | - | 1.1 |
| splice | cls | 0.965 | 0.965 | 0.976 | 0.963 | 0.955 | 0.955 | 0.957 | 0.952 | 1.79 |
| Bioresponse | cls | 0.781 | 0.773 | 0.792 | 0.795 | 0.768 | 0.787 | 0.776 | 0.795 | 1.94 |
| hiva_agnostic | cls | 0.968 | 0.968 | 0.968 | 0.968 | - | 0.968 | 0.968 | - | 1.9 |
| students_dropout | cls | 0.728 | 0.765 | 0.757 | 0.752 | 0.749 | 0.747 | 0.749 | 0.760 | 1.7 |
| churn | cls | 0.933 | 0.944 | 0.952 | 0.933 | 0.944 | 0.944 | 0.936 | 0.928 | 1.2 |
| QSAR-TID-11 | reg | 1.509 | 1.413 | 1.393 | 1.526 | 1.445 | - | - | 1.527 | 1.94 |
| polish_bankruptcy | cls | 0.944 | 0.939 | 0.939 | 0.944 | 0.944 | 0.939 | 0.944 | 0.944 | 1.69 |
| wine_quality | reg | 0.783 | 0.654 | 0.634 | 0.669 | 0.666 | - | - | 0.675 | 1.1 |
| taiwanese_bankruptcy | cls | 0.968 | 0.976 | 0.979 | 0.979 | 0.971 | 0.971 | 0.971 | 0.973 | 1.85 |
| NATICUSdroid | cls | 0.880 | 0.899 | 0.904 | 0.885 | 0.896 | 0.893 | 0.880 | 0.893 | 1.91 |
| coil2000_insurance | cls | 0.944 | 0.952 | 0.952 | 0.952 | - | 0.952 | 0.952 | - | 1.89 |
| Bank_Customer_Churn | cls | 0.837 | 0.875 | 0.872 | 0.877 | 0.869 | 0.875 | 0.877 | 0.867 | 1.32 |
| heloc | cls | 0.757 | 0.776 | 0.773 | 0.765 | 0.771 | 0.752 | 0.757 | 0.760 | 1.28 |
| jm1 | cls | 0.776 | 0.805 | 0.805 | 0.797 | 0.795 | 0.800 | 0.795 | 0.797 | 1.4 |
| E-CommerceShipping | cls | 0.645 | 0.693 | 0.693 | 0.707 | 0.685 | 0.691 | 0.699 | 0.707 | 0.95 |
| online_shoppers_intention | cls | 0.883 | 0.901 | 0.901 | 0.891 | 0.893 | 0.901 | 0.808 | 0.893 | 1.07 |
| in_vehicle_coupon | cls | 0.632 | 0.643 | 0.669 | 0.683 | 0.629 | 0.643 | 0.648 | 0.667 | 1.28 |
| miami_housing | reg | 149102.718 | 83065.624 | 90764.039 | 96749.143 | 195469.481 | - | - | 105196.030 | 1.17 |
| HR_job_change | cls | 0.728 | 0.744 | 0.731 | 0.747 | 0.755 | 0.733 | 0.741 | 0.741 | 0.89 |
| houses | reg | 0.268 | 0.239 | 0.226 | 0.249 | 0.261 | - | - | 0.268 | 0.88 |
| superconductivity | reg | 14.182 | 11.978 | 11.852 | 15.355 | 15.483 | - | - | 16.155 | 1.92 |
| credit_card_default | cls | 0.800 | 0.827 | 0.835 | 0.829 | 0.832 | 0.832 | 0.829 | 0.824 | 1.59 |
| Amazon_employee_access | cls | 0.928 | 0.939 | 0.939 | 0.939 | - | 0.939 | 0.939 | - | 0.96 |
| bank-marketing | cls | 0.856 | 0.867 | 0.867 | 0.875 | 0.867 | 0.867 | 0.869 | 0.877 | 0.91 |
| Food_Delivery_Time | reg | 8.720 | 7.445 | 7.493 | 7.514 | 7.458 | - | - | 7.505 | 0.85 |
| physiochemical_protein | reg | 5.071 | 4.491 | 4.414 | 4.814 | 5.059 | - | - | 5.051 | 0.78 |
| kddcup09_appetency | cls | 0.981 | 0.981 | 0.981 | 0.981 | - | 0.981 | 0.981 | - | 1.89 |
| diamonds | reg | 878.255 | 707.699 | 669.062 | 897.465 | 928.764 | - | - | 957.403 | 1.03 |
| Diabetes130US | cls | 0.904 | 0.917 | 0.917 | 0.917 | 0.917 | 0.917 | 0.917 | - | 1.84 |
| APSFailure | cls | 0.995 | 0.995 | 0.992 | 0.992 | 0.997 | 1.000 | 0.997 | 0.992 | 2.54 |
| SDSS17 | cls | 0.952 | 0.960 | 0.957 | 0.965 | 0.960 | 0.963 | 0.968 | 0.960 | 1.74 |
| airline_satisfaction | cls | 0.909 | 0.920 | 0.909 | 0.912 | 0.885 | 0.891 | 0.883 | 0.875 | 1.74 |
| GiveMeSomeCredit | cls | 0.931 | 0.939 | 0.936 | 0.936 | 0.933 | 0.939 | 0.931 | 0.936 | 1.23 |

## Win counts

- TabICL beats xgboost: 45/51
- TabICL beats TabPFN-2: 25/51
- Mitra beats xgboost: 37/51
- our gbt<-TabICL beats xgboost: 30/46
- out-of-fold soft labels beat in-context soft labels: 17/38

The soft-oof column uses stratified out-of-fold teacher labels:
an in-context teacher scoring rows already in its own context
leaks labels and collapses the soft targets toward one-hot
(Tanna et al., "Pocket Foundation Models", arXiv:2605.18654;
technique adopted from their paper with thanks). Measured
honestly: on this suite the fix does not lift accuracy (win
count above; median delta 0.000). The likely reasons: these
datasets cap at 1500 rows, so each fold-fit teacher loses
context it can ill afford; the paper's gains are measured in
AUC, where soft-label structure matters, while this table is
accuracy, where only the argmax does; and their pipeline adds
temperature scaling ours does not. The two results bracket
where the technique earns its keep: larger data, probability
metrics.
