# TabArena-v0.1 datasets

The 51 datasets the full benchmark (`benchmarks/results/tabarena-full.md`)
runs on. Canonical sources for what each one tests:

- **TabArena leaderboard + docs:** <https://tabarena.ai>
- **Paper (dataset appendix):** [TabArena: A Living Benchmark for Machine
  Learning on Tabular Data](https://arxiv.org/abs/2506.16791) (Erickson et
  al., 2025)
- **OpenML suite (master list, each dataset links to its full description and
  target feature):** <https://www.openml.org/s/457>

Every row below links to its OpenML page, which carries the full prose
description, the feature list, and the exact target column. `n` is the full
dataset size; our benchmark subsamples each to ≤1500 rows / ≤40 features so
the in-context TabFM teacher stays tractable on CPU.

## Classification

| dataset | classes | n | predicts | OpenML |
| --- | --- | --- | --- | --- |
| blood-transfusion | 2 | 748 | whether a donor gives blood in a target window | [46913](https://www.openml.org/d/46913) |
| diabetes | 2 | 768 | diabetes diagnosis (Pima) | [46921](https://www.openml.org/d/46921) |
| anneal | 5 | 898 | steel annealing class | [46906](https://www.openml.org/d/46906) |
| credit-g | 2 | 1000 | good vs bad credit risk (German credit) | [46918](https://www.openml.org/d/46918) |
| maternal_health_risk | 3 | 1014 | maternal health risk level | [46941](https://www.openml.org/d/46941) |
| qsar-biodeg | 2 | 1054 | whether a chemical is ready-biodegradable | [46952](https://www.openml.org/d/46952) |
| website_phishing | 3 | 1353 | phishing / suspicious / legitimate site | [46963](https://www.openml.org/d/46963) |
| Fitness_Club | 2 | 1500 | gym-class attendance | [46927](https://www.openml.org/d/46927) |
| MIC | 8 | 1699 | multi-class molecular / activity label | [46980](https://www.openml.org/d/46980) |
| Is-this-a-good-customer | 2 | 1723 | customer creditworthiness | [46938](https://www.openml.org/d/46938) |
| Marketing_Campaign | 2 | 2240 | response to a marketing campaign | [46940](https://www.openml.org/d/46940) |
| hazelnut-contaminant | 2 | 2400 | contaminant detection in hazelnut spread | [46930](https://www.openml.org/d/46930) |
| seismic-bumps | 2 | 2584 | hazardous seismic activity in a mine | [46956](https://www.openml.org/d/46956) |
| splice | 3 | 3190 | DNA splice-junction type (donor/acceptor/neither) | [46958](https://www.openml.org/d/46958) |
| Bioresponse | 2 | 3751 | whether a molecule elicits a biological response | [46912](https://www.openml.org/d/46912) |
| hiva_agnostic | 3 | 3845 | compound activity against HIV | [46933](https://www.openml.org/d/46933) |
| students_dropout | 3 | 4424 | student dropout / enrolled / graduate | [46960](https://www.openml.org/d/46960) |
| churn | 2 | 5000 | telecom customer churn | [46915](https://www.openml.org/d/46915) |
| polish_bankruptcy | 2 | 5910 | company bankruptcy (Polish firms) | [46950](https://www.openml.org/d/46950) |
| taiwanese_bankruptcy | 2 | 6819 | company bankruptcy (Taiwanese firms) | [46962](https://www.openml.org/d/46962) |
| NATICUSdroid | 2 | 7491 | Android app benign vs malware | [46969](https://www.openml.org/d/46969) |
| coil2000_insurance | 2 | 9822 | caravan insurance purchase (highly imbalanced) | [46916](https://www.openml.org/d/46916) |
| Bank_Customer_Churn | 2 | 10000 | bank customer churn | [46911](https://www.openml.org/d/46911) |
| heloc | 2 | 10459 | home-equity line repayment risk | [46932](https://www.openml.org/d/46932) |
| jm1 | 2 | 10885 | software module defect-proneness (NASA) | [46979](https://www.openml.org/d/46979) |
| E-CommerceShipping | 2 | 10999 | on-time delivery | [46924](https://www.openml.org/d/46924) |
| online_shoppers_intention | 2 | 12330 | whether a session ends in a purchase | [46947](https://www.openml.org/d/46947) |
| in_vehicle_coupon | 2 | 12684 | whether a driver accepts an in-vehicle coupon | [46937](https://www.openml.org/d/46937) |
| HR_job_change | 2 | 19158 | whether a data-science trainee seeks a job change | [46935](https://www.openml.org/d/46935) |
| credit_card_default | 2 | 30000 | credit-card payment default (Taiwan) | [46919](https://www.openml.org/d/46919) |
| Amazon_employee_access | 2 | 32769 | whether an access request is approved | [46905](https://www.openml.org/d/46905) |
| bank-marketing | 2 | 45211 | term-deposit subscription | [46910](https://www.openml.org/d/46910) |
| kddcup09_appetency | 2 | 50000 | customer appetency (KDD Cup 2009) | [46939](https://www.openml.org/d/46939) |
| Diabetes130US | 2 | 71518 | hospital readmission of diabetic patients | [46922](https://www.openml.org/d/46922) |
| APSFailure | 2 | 76000 | Scania truck air-pressure-system failure | [46908](https://www.openml.org/d/46908) |
| SDSS17 | 3 | 78053 | star / galaxy / quasar (Sloan sky survey) | [46955](https://www.openml.org/d/46955) |
| airline_satisfaction | 2 | 129880 | airline passenger satisfaction | [46920](https://www.openml.org/d/46920) |
| GiveMeSomeCredit | 2 | 150000 | serious financial delinquency in two years | [46929](https://www.openml.org/d/46929) |

## Regression

| dataset | n | predicts | OpenML |
| --- | --- | --- | --- |
| QSAR_fish_toxicity | 907 | fish acute toxicity (LC50) of a chemical | [46954](https://www.openml.org/d/46954) |
| concrete_compressive_strength | 1030 | concrete compressive strength | [46917](https://www.openml.org/d/46917) |
| healthcare_insurance_expenses | 1338 | individual medical insurance charges | [46931](https://www.openml.org/d/46931) |
| airfoil_self_noise | 1503 | airfoil self-noise (sound pressure level) | [46904](https://www.openml.org/d/46904) |
| used-Fiat-500 | 1538 | used Fiat 500 resale price | [46907](https://www.openml.org/d/46907) |
| QSAR-TID-11 | 5742 | compound bioactivity against a target | [46953](https://www.openml.org/d/46953) |
| wine_quality | 6497 | wine sensory quality score | [46964](https://www.openml.org/d/46964) |
| miami_housing | 13776 | Miami house sale price | [46942](https://www.openml.org/d/46942) |
| houses | 20640 | California district median house value | [46934](https://www.openml.org/d/46934) |
| superconductivity | 21263 | superconductor critical temperature | [46961](https://www.openml.org/d/46961) |
| Food_Delivery_Time | 45451 | food delivery time | [46928](https://www.openml.org/d/46928) |
| physiochemical_protein | 45730 | protein-structure residue RMSD | [46949](https://www.openml.org/d/46949) |
| diamonds | 53940 | diamond price | [46923](https://www.openml.org/d/46923) |

Glosses are short summaries; the OpenML page is authoritative for each
dataset's exact target and features. A few (MIC, hiva_agnostic, QSAR-TID-11)
are specialist cheminformatics/materials sets where the OpenML description is
the reliable reference.
