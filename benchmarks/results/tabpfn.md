# TabPFN on the TabArena subset (and distilling it)

TabPFN-2's weights are under the Prior Labs License (distillation
permitted, commercial use included, with attribution when the
student is distributed). TabPFN-3, the current default, is
non-commercial: its numbers are the evaluation ceiling. Same
datasets, splits, and row/feature caps as `tabarena-full.md`;
TabFM/xgboost/knn5 columns are that campaign's numbers.

Device: mps. TabPFN package 8.2.0.

| dataset | task | xgboost | TabFM | TabPFN-2 | TabPFN-3 | gbt<-2 | soft<-2 | gbt<-3 | s/call (3) |
|---|---|---|---|---|---|---|---|---|---|
| blood-transfusion | cls | 0.711 | 0.759 | 0.765 | 0.765 | 0.765 | 0.765 | 0.754 | 108.1 |
| diabetes | cls | 0.740 | 0.776 | 0.750 | 0.755 | 0.771 | 0.750 | 0.740 | 1.78 |
| anneal | cls | 0.987 | 0.991 | 0.991 | 0.991 | 0.987 | 0.987 | 0.987 | 1.91 |
| QSAR_fish_toxicity | reg | 0.949 | 0.910 | 0.918 | 0.922 | 0.921 | - | 0.925 | 64.84 |
| credit-g | cls | 0.752 | 0.768 | 0.784 | 0.780 | 0.772 | 0.788 | 0.732 | 1.45 |
| maternal_health_risk | cls | 0.807 | 0.850 | 0.831 | 0.835 | 0.776 | 0.768 | 0.787 | 1.18 |
| concrete_compressive_strength | reg | 5.047 | 4.082 | 4.198 | 4.147 | 5.075 | - | 5.031 | 1.27 |
| qsar-biodeg | cls | 0.871 | 0.890 | 0.890 | 0.890 | 0.856 | 0.871 | 0.845 | 1.67 |
| healthcare_insurance_expenses | reg | 5217.371 | 4455.508 | 4829.207 | 4532.083 | 4504.716 | - | 4511.130 | 1.49 |
| website_phishing | cls | 0.885 | 0.917 | 0.909 | 0.914 | 0.885 | 0.882 | 0.885 | 1.46 |
| Fitness_Club | cls | 0.747 | 0.787 | 0.781 | 0.784 | 0.784 | 0.779 | 0.787 | 1.35 |
| airfoil_self_noise | reg | 1.613 | 1.081 | 1.092 | 1.022 | 2.259 | - | 2.294 | 1.29 |
| used-Fiat-500 | reg | 830.136 | 779.429 | 803.507 | 786.372 | 813.531 | - | 794.604 | 1.29 |
| MIC | cls | 0.848 | 0.861 | 0.867 | 0.859 | 0.851 | 0.861 | 0.851 | 1.81 |
| Marketing_Campaign | cls | 0.869 | 0.891 | 0.891 | 0.885 | 0.880 | 0.893 | 0.885 | 1.82 |
| hazelnut-contaminant | cls | 0.901 | 0.971 | 0.957 | 0.968 | 0.891 | 0.885 | 0.891 | 1.52 |
| splice | cls | 0.965 | 0.973 | 0.965 | 0.976 | 0.952 | 0.955 | 0.955 | 1.47 |
| Bioresponse | cls | 0.781 | 0.808 | 0.773 | 0.787 | 0.776 | 0.776 | 0.765 | 1.48 |
| students_dropout | cls | 0.728 | 0.781 | 0.765 | 0.757 | 0.755 | 0.749 | 0.733 | 1.84 |
| churn | cls | 0.933 | 0.968 | 0.944 | 0.955 | 0.933 | 0.936 | 0.952 | 1.66 |
| QSAR-TID-11 | reg | 1.509 | 1.352 | 1.413 | 1.390 | 1.442 | - | 1.441 | 1.93 |
| polish_bankruptcy | cls | 0.944 | 0.957 | 0.939 | 0.949 | 0.947 | 0.939 | 0.949 | 1.46 |
| wine_quality | reg | 0.783 | 0.634 | 0.654 | 0.634 | 0.666 | - | 0.685 | 1.52 |
| taiwanese_bankruptcy | cls | 0.968 | 0.976 | 0.976 | 0.979 | 0.973 | 0.981 | 0.973 | 1.46 |
| NATICUSdroid | cls | 0.880 | 0.920 | 0.899 | 0.909 | 0.893 | 0.888 | 0.899 | 1.44 |
| Bank_Customer_Churn | cls | 0.837 | 0.880 | 0.875 | 0.875 | 0.867 | 0.869 | 0.867 | 1.32 |
| heloc | cls | 0.757 | 0.773 | 0.776 | 0.771 | 0.765 | 0.773 | 0.763 | 1.62 |
| jm1 | cls | 0.776 | 0.803 | 0.805 | 0.803 | 0.800 | 0.800 | 0.781 | 1.56 |
| E-CommerceShipping | cls | 0.645 | 0.701 | 0.693 | 0.696 | 0.693 | 0.688 | 0.696 | 1.13 |
| online_shoppers_intention | cls | 0.883 | 0.904 | 0.901 | 0.904 | 0.896 | 0.893 | 0.883 | 1.37 |
| in_vehicle_coupon | cls | 0.632 | 0.656 | 0.643 | 0.643 | 0.632 | 0.645 | 0.664 | 1.3 |
| miami_housing | reg | 149102.718 | 85055.456 | 83065.624 | 118263.713 | 120661.346 | - | 118344.679 | 1.55 |
| HR_job_change | cls | 0.728 | 0.739 | 0.744 | 0.739 | 0.749 | 0.747 | 0.741 | 1.14 |
| houses | reg | 0.268 | 0.202 | 0.239 | 0.213 | 0.270 | - | 0.258 | 1.28 |
| superconductivity | reg | 14.182 | 12.276 | 11.978 | 11.851 | 14.601 | - | 14.781 | 1.7 |
| credit_card_default | cls | 0.800 | 0.829 | 0.827 | 0.835 | 0.835 | 0.824 | 0.835 | 1.33 |
| bank-marketing | cls | 0.856 | 0.864 | 0.867 | 0.872 | 0.864 | 0.877 | 0.872 | 1.13 |
| Food_Delivery_Time | reg | 8.720 | 7.128 | 7.445 | 7.222 | 7.434 | - | 7.359 | 1.31 |
| physiochemical_protein | reg | 5.071 | 4.343 | 4.491 | 4.321 | 5.009 | - | 4.992 | 1.13 |
| diamonds | reg | 878.255 | 763.139 | 707.699 | 677.011 | 926.225 | - | 949.039 | 1.11 |
| APSFailure | cls | 0.995 | 0.995 | 0.995 | 0.997 | 0.995 | 0.995 | 0.995 | 1.9 |
| SDSS17 | cls | 0.952 | 0.960 | 0.960 | 0.963 | 0.960 | 0.963 | 0.957 | 1.33 |
| airline_satisfaction | cls | 0.909 | 0.923 | 0.920 | 0.925 | 0.891 | 0.883 | 0.896 | 1.23 |
| GiveMeSomeCredit | cls | 0.931 | 0.939 | 0.939 | 0.939 | 0.931 | 0.936 | 0.933 | 1.08 |
| Is-this-a-good-customer | cls | 0.880 | 0.891 | 0.893 | 0.893 | - | 0.893 | 0.893 | 1.37 |
| seismic-bumps | cls | 0.931 | 0.936 | 0.936 | 0.936 | - | 0.936 | 0.936 | 1.16 |
| hiva_agnostic | cls | 0.968 | nan | 0.968 | 0.968 | - | 0.968 | - | 1.48 |
| coil2000_insurance | cls | 0.944 | nan | 0.952 | 0.949 | - | 0.952 | 0.952 | 1.43 |
| Amazon_employee_access | cls | 0.928 | 0.939 | 0.939 | 0.939 | - | 0.939 | 0.941 | 1.1 |
| kddcup09_appetency | cls | 0.981 | nan | 0.981 | 0.981 | - | 0.981 | 0.981 | 1.41 |
| Diabetes130US | cls | 0.904 | 0.917 | 0.917 | 0.917 | - | 0.917 | 0.907 | 1.64 |

## Win counts

- TabPFN-2 beats xgboost: 45/51
- TabPFN-3 beats xgboost: 49/51
- TabPFN-3 beats TabFM: 15/51
- soft gbt<-TabPFN-2 beats xgboost: 26/38

Built with PriorLabs-TabPFN (evaluation; students were
not distributed).
