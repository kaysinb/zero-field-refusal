## outcome

| field | value |
|---|---|
| variant | standard |
| stage | done |
| candidate | 1 |
| outcome | confirmed |
| reason | n_reached |
| readings | {"A-F": "confirmatory", "P-F": "confirmatory"} |
| active | {"arms": ["A", "F", "P"], "comparisons": ["A-F", "P-F"]} |

| reasons |
|---|

| candidate | stage | event | arms | why |
|---|---|---|---|---|
| 1 | numerical | admitted | F | amendment 10: adam_frozen-1.3e-05: failing; adam_frozen-1.83e-05: failing |
| 1 | numerical | admitted | P | amendment 10: adamw_prewarmed-1.85e-05: failing |
| 1 | choice | chosen | A, F, P | the first candidate whose A and F passed their qualification and validation |

| closure | state | path | run_id | arm | reason | decision |
|---|---|---|---|---|---|---|

| dead end | candidate | stage | arms | reason | batch |
|---|---|---|---|---|---|


## dry_run

| field | value |
|---|---|
| record | study3/decisions/engineering.json |
| sha256 | 4f4948094d6055f774b211f51e1091fdeb273646e4094c093a2ed8bb3daa44b7 |
| state | passed |

| check | path | sha256 | root | passed |
|---|---|---|---|---|
| bridge | docs/reports/study3-engineering-evidence/verdict.json | 16558de4c3b77e204e51b265ca00af3da5443c3150dbbba10494e0781c752a36 | repo | true |
| dry_run | docs/reports/study3-engineering.md | 12f92b580f9c7f04a1c9d9c8fa246fec0b3cd0d3d1cb10499c892a44f29b518c | repo | true |
| stage0 | docs/reports/study3-engineering-evidence/stage0-passed.out | 2c419020d07e41d57a23780b1c969f77feaaae6be66afc85707b1ee13f91e5bb | repo | true |
| theta0_utility | docs/reports/study3-engineering-evidence/theta0-utility.out | 7318d076f53610acf159dd61618203a0beb6216068b65153fd21ae3d03933fdb | repo | true |

| registration | commit | sha256 | amendments |
|---|---|---|---|
| docs/study3-preregistration.md | b13e297317fe2452fcf116ba645d9f020b71f2c8 | 2c15c2ae8b14ff9202f154f1a4760843b2e22c5113355d9e1debc8ecd32d8372 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |


## engineering

| field | value |
|---|---|
| verdict | qualification/c1/engineering/verdict.json |
| sha256 | 16558de4c3b77e204e51b265ca00af3da5443c3150dbbba10494e0781c752a36 |
| e_eng | 2.11371e-10 |
| repeats | 24 |

| path maximum | value |
|---|---|
| repeat | 2.02372e-10 |
| reference | 0.00398288 |
| shift | 0.0557809 |

| margin | value |
|---|---|
| launcher~baseline | 3.98747e+07 |
| launcher~repeat | 3.98747e+07 |
| baseline~repeat | 4.73101e+07 |
| launcher~reference | 2.02117 |
| baseline~reference | 2.02117 |
| launcher~shift | 0.601888 |
| baseline~shift | 0.601888 |


## simulation

| field | value |
|---|---|
| record | study3/decisions/simulation.json |
| sha256 | 22ba64302e02a80f73d0355d2507c9449e0244e8fd94f63ee170fd64558c28b1 |
| state | passed |
| L | 0.985 |
| validation_blocks | 28 |
| results | study3/simulation/results.json |
| results_sha256 | 053e0ea6ced3fc6fa9c6d79a09da6409ee489488b356942cc06725b4d986dacf |

| gate | value | requirement | passed | power_short | n_min |
|---|---|---|---|---|---|
| detection_power_comparison | 0.963055 | null | null | null | null |
| detection_power_joint | 0.989774 | null | null | null | null |
| empirical_pass_rate | 0.9465 | null | null | null | null |
| power_af | 0.8642 | 0.8 | true | false | null |
| power_af_1pct_missing | 0.2999 | null | null | null | null |
| power_pf | 0.8596 | null | null | null | null |
| safe_pass_rate | 0.9834 | 0.95 | true | null | null |

| level | tests | J | M | X comparison | c comparison | X joint | c joint | passed |
|---|---|---|---|---|---|---|---|---|
| 0.975 | 357 | 2856 | 10000 | 334 | 317 | 678 | 593 | false |
| 0.98 | 357 | 2856 | 10000 | 288 | 317 | 589 | 593 | true |
| 0.985 | 357 | 2856 | 10000 | 237 | 317 | 539 | 593 | true |
| 0.99 | 357 | 2856 | 10000 | 157 | 317 | 498 | 593 | true |

| field | value |
|---|---|
| verification calibration_stream | 20261101 |
| verification level | 0.985 |
| verification passed | true |
| verification tests | 1428 |
| verification verification_stream | 20261102 |

| test | level | due | studies | X | c | passed | non-coverage | 95 % interval | reason | floor |
|---|---|---|---|---|---|---|---|---|---|---|
| calibration/C01/36/0.975/A-F | 0.975 | true | 10000 | 269 | 317 | true | 0.0269 | 0.0238173, 0.0302622 | null | null |
| calibration/C01/36/0.975/P-F | 0.975 | true | 10000 | 246 | 317 | true | 0.0246 | 0.021653, 0.0278281 | null | null |
| calibration/C01/36/0.975/joint | 0.975 | true | 10000 | 503 | 593 | true | 0.0503 | 0.0460977, 0.0547665 | null | null |
| calibration/C01/36/0.980/A-F | 0.98 | true | 10000 | 212 | 317 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| calibration/C01/36/0.980/P-F | 0.98 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| calibration/C01/36/0.980/joint | 0.98 | true | 10000 | 394 | 593 | true | 0.0394 | 0.0356717, 0.0433995 | null | null |
| calibration/C01/36/0.985/A-F | 0.985 | true | 10000 | 143 | 317 | true | 0.0143 | 0.0120653, 0.0168237 | null | null |
| calibration/C01/36/0.985/P-F | 0.985 | true | 10000 | 148 | 317 | true | 0.0148 | 0.0125254, 0.0173632 | null | null |
| calibration/C01/36/0.985/joint | 0.985 | true | 10000 | 285 | 593 | true | 0.0285 | 0.0253266, 0.0319518 | null | null |
| calibration/C01/36/0.990/A-F | 0.99 | true | 10000 | 98 | 317 | true | 0.0098 | 0.00796307, 0.0119302 | null | null |
| calibration/C01/36/0.990/P-F | 0.99 | true | 10000 | 108 | 317 | true | 0.0108 | 0.00886763, 0.0130246 | null | null |
| calibration/C01/36/0.990/joint | 0.99 | true | 10000 | 204 | 593 | true | 0.0204 | 0.0177197, 0.0233645 | null | null |
| calibration/C01/45/0.975/A-F | 0.975 | true | 10000 | 260 | 317 | true | 0.026 | 0.0229697, 0.0293105 | null | null |
| calibration/C01/45/0.975/P-F | 0.975 | true | 10000 | 237 | 317 | true | 0.0237 | 0.020808, 0.0268738 | null | null |
| calibration/C01/45/0.975/joint | 0.975 | true | 10000 | 492 | 593 | true | 0.0492 | 0.0450427, 0.0536223 | null | null |
| calibration/C01/45/0.980/A-F | 0.98 | true | 10000 | 203 | 317 | true | 0.0203 | 0.0176264, 0.0232579 | null | null |
| calibration/C01/45/0.980/P-F | 0.98 | true | 10000 | 198 | 317 | true | 0.0198 | 0.0171601, 0.0227245 | null | null |
| calibration/C01/45/0.980/joint | 0.98 | true | 10000 | 398 | 593 | true | 0.0398 | 0.0360531, 0.0438179 | null | null |
| calibration/C01/45/0.985/A-F | 0.985 | true | 10000 | 145 | 317 | true | 0.0145 | 0.0122493, 0.0170396 | null | null |
| calibration/C01/45/0.985/P-F | 0.985 | true | 10000 | 149 | 317 | true | 0.0149 | 0.0126175, 0.017471 | null | null |
| calibration/C01/45/0.985/joint | 0.985 | true | 10000 | 292 | 593 | true | 0.0292 | 0.0259878, 0.0326901 | null | null |
| calibration/C01/45/0.990/A-F | 0.99 | true | 10000 | 100 | 317 | true | 0.01 | 0.0081436, 0.0121495 | null | null |
| calibration/C01/45/0.990/P-F | 0.99 | true | 10000 | 101 | 317 | true | 0.0101 | 0.00823393, 0.0122591 | null | null |
| calibration/C01/45/0.990/joint | 0.99 | true | 10000 | 200 | 593 | true | 0.02 | 0.0173465, 0.0229379 | null | null |
| calibration/C01/60/0.975/A-F | 0.975 | true | 10000 | 239 | 317 | true | 0.0239 | 0.0209957, 0.0270859 | null | null |
| calibration/C01/60/0.975/P-F | 0.975 | true | 10000 | 245 | 317 | true | 0.0245 | 0.0215591, 0.0277221 | null | null |
| calibration/C01/60/0.975/joint | 0.975 | true | 10000 | 478 | 593 | true | 0.0478 | 0.0437008, 0.0521651 | null | null |
| calibration/C01/60/0.980/A-F | 0.98 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| calibration/C01/60/0.980/P-F | 0.98 | true | 10000 | 193 | 317 | true | 0.0193 | 0.0166942, 0.0221908 | null | null |
| calibration/C01/60/0.980/joint | 0.98 | true | 10000 | 380 | 593 | true | 0.038 | 0.0343377, 0.0419344 | null | null |
| calibration/C01/60/0.985/A-F | 0.985 | true | 10000 | 137 | 317 | true | 0.0137 | 0.0115142, 0.0161754 | null | null |
| calibration/C01/60/0.985/P-F | 0.985 | true | 10000 | 144 | 317 | true | 0.0144 | 0.0121573, 0.0169316 | null | null |
| calibration/C01/60/0.985/joint | 0.985 | true | 10000 | 279 | 593 | true | 0.0279 | 0.0247603, 0.0313185 | null | null |
| calibration/C01/60/0.990/A-F | 0.99 | true | 10000 | 95 | 317 | true | 0.0095 | 0.00769267, 0.011601 | null | null |
| calibration/C01/60/0.990/P-F | 0.99 | true | 10000 | 97 | 317 | true | 0.0097 | 0.00787289, 0.0118205 | null | null |
| calibration/C01/60/0.990/joint | 0.99 | true | 10000 | 191 | 593 | true | 0.0191 | 0.016508, 0.0219771 | null | null |
| calibration/C01/75/0.975/A-F | 0.975 | true | 10000 | 240 | 317 | true | 0.024 | 0.0210896, 0.027192 | null | null |
| calibration/C01/75/0.975/P-F | 0.975 | true | 10000 | 259 | 317 | true | 0.0259 | 0.0228755, 0.0292047 | null | null |
| calibration/C01/75/0.975/joint | 0.975 | true | 10000 | 490 | 593 | true | 0.049 | 0.0448509, 0.0534142 | null | null |
| calibration/C01/75/0.980/A-F | 0.98 | true | 10000 | 189 | 317 | true | 0.0189 | 0.0163219, 0.0217634 | null | null |
| calibration/C01/75/0.980/P-F | 0.98 | true | 10000 | 207 | 317 | true | 0.0207 | 0.0179997, 0.0236842 | null | null |
| calibration/C01/75/0.980/joint | 0.98 | true | 10000 | 391 | 593 | true | 0.0391 | 0.0353857, 0.0430857 | null | null |
| calibration/C01/75/0.985/A-F | 0.985 | true | 10000 | 142 | 317 | true | 0.0142 | 0.0119734, 0.0167157 | null | null |
| calibration/C01/75/0.985/P-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| calibration/C01/75/0.985/joint | 0.985 | true | 10000 | 300 | 593 | true | 0.03 | 0.026744, 0.0335334 | null | null |
| calibration/C01/75/0.990/A-F | 0.99 | true | 10000 | 104 | 317 | true | 0.0104 | 0.00850524, 0.0125874 | null | null |
| calibration/C01/75/0.990/P-F | 0.99 | true | 10000 | 117 | 317 | true | 0.0117 | 0.00968556, 0.0140058 | null | null |
| calibration/C01/75/0.990/joint | 0.99 | true | 10000 | 219 | 593 | true | 0.0219 | 0.0191214, 0.0249617 | null | null |
| calibration/C01/90/0.975/A-F | 0.975 | true | 10000 | 256 | 317 | true | 0.0256 | 0.0225932, 0.0288872 | null | null |
| calibration/C01/90/0.975/P-F | 0.975 | true | 10000 | 239 | 317 | true | 0.0239 | 0.0209957, 0.0270859 | null | null |
| calibration/C01/90/0.975/joint | 0.975 | true | 10000 | 492 | 593 | true | 0.0492 | 0.0450427, 0.0536223 | null | null |
| calibration/C01/90/0.980/A-F | 0.98 | true | 10000 | 212 | 317 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| calibration/C01/90/0.980/P-F | 0.98 | true | 10000 | 192 | 317 | true | 0.0192 | 0.0166011, 0.0220839 | null | null |
| calibration/C01/90/0.980/joint | 0.98 | true | 10000 | 401 | 593 | true | 0.0401 | 0.0363392, 0.0441316 | null | null |
| calibration/C01/90/0.985/A-F | 0.985 | true | 10000 | 159 | 317 | true | 0.0159 | 0.0135401, 0.0185476 | null | null |
| calibration/C01/90/0.985/P-F | 0.985 | true | 10000 | 142 | 317 | true | 0.0142 | 0.0119734, 0.0167157 | null | null |
| calibration/C01/90/0.985/joint | 0.985 | true | 10000 | 300 | 593 | true | 0.03 | 0.026744, 0.0335334 | null | null |
| calibration/C01/90/0.990/A-F | 0.99 | true | 10000 | 102 | 317 | true | 0.0102 | 0.00832432, 0.0123686 | null | null |
| calibration/C01/90/0.990/P-F | 0.99 | true | 10000 | 98 | 317 | true | 0.0098 | 0.00796307, 0.0119302 | null | null |
| calibration/C01/90/0.990/joint | 0.99 | true | 10000 | 200 | 593 | true | 0.02 | 0.0173465, 0.0229379 | null | null |
| calibration/C01/99/0.975/A-F | 0.975 | true | 10000 | 255 | 317 | true | 0.0255 | 0.0224992, 0.0287813 | null | null |
| calibration/C01/99/0.975/P-F | 0.975 | true | 10000 | 244 | 317 | true | 0.0244 | 0.0214652, 0.0276161 | null | null |
| calibration/C01/99/0.975/joint | 0.975 | true | 10000 | 492 | 593 | true | 0.0492 | 0.0450427, 0.0536223 | null | null |
| calibration/C01/99/0.980/A-F | 0.98 | true | 10000 | 205 | 317 | true | 0.0205 | 0.017813, 0.0234711 | null | null |
| calibration/C01/99/0.980/P-F | 0.98 | true | 10000 | 194 | 317 | true | 0.0194 | 0.0167874, 0.0222975 | null | null |
| calibration/C01/99/0.980/joint | 0.98 | true | 10000 | 392 | 593 | true | 0.0392 | 0.035481, 0.0431903 | null | null |
| calibration/C01/99/0.985/A-F | 0.985 | true | 10000 | 154 | 317 | true | 0.0154 | 0.0130785, 0.0180096 | null | null |
| calibration/C01/99/0.985/P-F | 0.985 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | null |
| calibration/C01/99/0.985/joint | 0.985 | true | 10000 | 300 | 593 | true | 0.03 | 0.026744, 0.0335334 | null | null |
| calibration/C01/99/0.990/A-F | 0.99 | true | 10000 | 102 | 317 | true | 0.0102 | 0.00832432, 0.0123686 | null | null |
| calibration/C01/99/0.990/P-F | 0.99 | true | 10000 | 98 | 317 | true | 0.0098 | 0.00796307, 0.0119302 | null | null |
| calibration/C01/99/0.990/joint | 0.99 | true | 10000 | 199 | 593 | true | 0.0199 | 0.0172533, 0.0228312 | null | null |
| calibration/C01/cap/0.975/A-F | 0.975 | true | 10000 | 232 | 317 | true | 0.0232 | 0.020339, 0.0263431 | null | 60 |
| calibration/C01/cap/0.975/P-F | 0.975 | true | 10000 | 233 | 317 | true | 0.0233 | 0.0204328, 0.0264493 | null | 60 |
| calibration/C01/cap/0.975/joint | 0.975 | true | 10000 | 457 | 593 | true | 0.0457 | 0.0416898, 0.0499774 | null | 60 |
| calibration/C01/cap/0.980/A-F | 0.98 | true | 10000 | 183 | 317 | true | 0.0183 | 0.015764, 0.0211218 | null | 60 |
| calibration/C01/cap/0.980/P-F | 0.98 | true | 10000 | 188 | 317 | true | 0.0188 | 0.0162289, 0.0216565 | null | 60 |
| calibration/C01/cap/0.980/joint | 0.98 | true | 10000 | 365 | 593 | true | 0.0365 | 0.03291, 0.0403631 | null | 60 |
| calibration/C01/cap/0.985/A-F | 0.985 | true | 10000 | 142 | 317 | true | 0.0142 | 0.0119734, 0.0167157 | null | 60 |
| calibration/C01/cap/0.985/P-F | 0.985 | true | 10000 | 145 | 317 | true | 0.0145 | 0.0122493, 0.0170396 | null | 60 |
| calibration/C01/cap/0.985/joint | 0.985 | true | 10000 | 288 | 593 | true | 0.0288 | 0.0256099, 0.0322683 | null | 60 |
| calibration/C01/cap/0.990/A-F | 0.99 | true | 10000 | 86 | 317 | true | 0.0086 | 0.00688448, 0.0106102 | null | 60 |
| calibration/C01/cap/0.990/P-F | 0.99 | true | 10000 | 95 | 317 | true | 0.0095 | 0.00769267, 0.011601 | null | 60 |
| calibration/C01/cap/0.990/joint | 0.99 | true | 10000 | 181 | 593 | true | 0.0181 | 0.0155782, 0.0209077 | null | 60 |
| calibration/C02/36/0.975/A-F | 0.975 | true | 10000 | 252 | 317 | true | 0.0252 | 0.022217, 0.0284637 | null | null |
| calibration/C02/36/0.975/P-F | 0.975 | true | 10000 | 250 | 317 | true | 0.025 | 0.022029, 0.0282519 | null | null |
| calibration/C02/36/0.975/joint | 0.975 | true | 10000 | 499 | 593 | true | 0.0499 | 0.045714, 0.0543505 | null | null |
| calibration/C02/36/0.980/A-F | 0.98 | true | 10000 | 200 | 317 | true | 0.02 | 0.0173465, 0.0229379 | null | null |
| calibration/C02/36/0.980/P-F | 0.98 | true | 10000 | 213 | 317 | true | 0.0213 | 0.0185602, 0.0243233 | null | null |
| calibration/C02/36/0.980/joint | 0.98 | true | 10000 | 411 | 593 | true | 0.0411 | 0.0372932, 0.0451769 | null | null |
| calibration/C02/36/0.985/A-F | 0.985 | true | 10000 | 156 | 317 | true | 0.0156 | 0.013263, 0.0182249 | null | null |
| calibration/C02/36/0.985/P-F | 0.985 | true | 10000 | 166 | 317 | true | 0.0166 | 0.0141874, 0.0192998 | null | null |
| calibration/C02/36/0.985/joint | 0.985 | true | 10000 | 320 | 593 | true | 0.032 | 0.0286373, 0.0356387 | null | null |
| calibration/C02/36/0.990/A-F | 0.99 | true | 10000 | 106 | 317 | true | 0.0106 | 0.00868634, 0.0128061 | null | null |
| calibration/C02/36/0.990/P-F | 0.99 | true | 10000 | 112 | 317 | true | 0.0112 | 0.00923074, 0.0134611 | null | null |
| calibration/C02/36/0.990/joint | 0.99 | true | 10000 | 216 | 593 | true | 0.0216 | 0.0188407, 0.0246425 | null | null |
| calibration/C02/45/0.975/A-F | 0.975 | true | 10000 | 252 | 317 | true | 0.0252 | 0.022217, 0.0284637 | null | null |
| calibration/C02/45/0.975/P-F | 0.975 | true | 10000 | 250 | 317 | true | 0.025 | 0.022029, 0.0282519 | null | null |
| calibration/C02/45/0.975/joint | 0.975 | true | 10000 | 496 | 593 | true | 0.0496 | 0.0454263, 0.0540384 | null | null |
| calibration/C02/45/0.980/A-F | 0.98 | true | 10000 | 193 | 317 | true | 0.0193 | 0.0166942, 0.0221908 | null | null |
| calibration/C02/45/0.980/P-F | 0.98 | true | 10000 | 207 | 317 | true | 0.0207 | 0.0179997, 0.0236842 | null | null |
| calibration/C02/45/0.980/joint | 0.98 | true | 10000 | 396 | 593 | true | 0.0396 | 0.0358624, 0.0436087 | null | null |
| calibration/C02/45/0.985/A-F | 0.985 | true | 10000 | 153 | 317 | true | 0.0153 | 0.0129862, 0.0179019 | null | null |
| calibration/C02/45/0.985/P-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| calibration/C02/45/0.985/joint | 0.985 | true | 10000 | 303 | 593 | true | 0.0303 | 0.0270278, 0.0338494 | null | null |
| calibration/C02/45/0.990/A-F | 0.99 | true | 10000 | 100 | 317 | true | 0.01 | 0.0081436, 0.0121495 | null | null |
| calibration/C02/45/0.990/P-F | 0.99 | true | 10000 | 98 | 317 | true | 0.0098 | 0.00796307, 0.0119302 | null | null |
| calibration/C02/45/0.990/joint | 0.99 | true | 10000 | 198 | 593 | true | 0.0198 | 0.0171601, 0.0227245 | null | null |
| calibration/C02/60/0.975/A-F | 0.975 | true | 10000 | 279 | 317 | true | 0.0279 | 0.0247603, 0.0313185 | null | null |
| calibration/C02/60/0.975/P-F | 0.975 | true | 10000 | 244 | 317 | true | 0.0244 | 0.0214652, 0.0276161 | null | null |
| calibration/C02/60/0.975/joint | 0.975 | true | 10000 | 518 | 593 | true | 0.0518 | 0.0475373, 0.056326 | null | null |
| calibration/C02/60/0.980/A-F | 0.98 | true | 10000 | 215 | 317 | true | 0.0215 | 0.0187472, 0.0245361 | null | null |
| calibration/C02/60/0.980/P-F | 0.98 | true | 10000 | 186 | 317 | true | 0.0186 | 0.0160429, 0.0214427 | null | null |
| calibration/C02/60/0.980/joint | 0.98 | true | 10000 | 400 | 593 | true | 0.04 | 0.0362438, 0.044027 | null | null |
| calibration/C02/60/0.985/A-F | 0.985 | true | 10000 | 156 | 317 | true | 0.0156 | 0.013263, 0.0182249 | null | null |
| calibration/C02/60/0.985/P-F | 0.985 | true | 10000 | 136 | 317 | true | 0.0136 | 0.0114224, 0.0160672 | null | null |
| calibration/C02/60/0.985/joint | 0.985 | true | 10000 | 291 | 593 | true | 0.0291 | 0.0258933, 0.0325847 | null | null |
| calibration/C02/60/0.990/A-F | 0.99 | true | 10000 | 101 | 317 | true | 0.0101 | 0.00823393, 0.0122591 | null | null |
| calibration/C02/60/0.990/P-F | 0.99 | true | 10000 | 91 | 317 | true | 0.0091 | 0.0073329, 0.0111612 | null | null |
| calibration/C02/60/0.990/joint | 0.99 | true | 10000 | 190 | 593 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| calibration/C02/75/0.975/A-F | 0.975 | true | 10000 | 291 | 317 | true | 0.0291 | 0.0258933, 0.0325847 | null | null |
| calibration/C02/75/0.975/P-F | 0.975 | true | 10000 | 223 | 317 | true | 0.0223 | 0.0194957, 0.025387 | null | null |
| calibration/C02/75/0.975/joint | 0.975 | true | 10000 | 510 | 593 | true | 0.051 | 0.0467694, 0.0554944 | null | null |
| calibration/C02/75/0.980/A-F | 0.98 | true | 10000 | 227 | 317 | true | 0.0227 | 0.0198704, 0.0258121 | null | null |
| calibration/C02/75/0.980/P-F | 0.98 | true | 10000 | 178 | 317 | true | 0.0178 | 0.0152996, 0.0205865 | null | null |
| calibration/C02/75/0.980/joint | 0.98 | true | 10000 | 401 | 593 | true | 0.0401 | 0.0363392, 0.0441316 | null | null |
| calibration/C02/75/0.985/A-F | 0.985 | true | 10000 | 170 | 317 | true | 0.017 | 0.0145578, 0.019729 | null | null |
| calibration/C02/75/0.985/P-F | 0.985 | true | 10000 | 143 | 317 | true | 0.0143 | 0.0120653, 0.0168237 | null | null |
| calibration/C02/75/0.985/joint | 0.985 | true | 10000 | 310 | 593 | true | 0.031 | 0.0276902, 0.0345865 | null | null |
| calibration/C02/75/0.990/A-F | 0.99 | true | 10000 | 113 | 317 | true | 0.0113 | 0.00932162, 0.0135702 | null | null |
| calibration/C02/75/0.990/P-F | 0.99 | true | 10000 | 84 | 317 | true | 0.0084 | 0.00670554, 0.0103894 | null | null |
| calibration/C02/75/0.990/joint | 0.99 | true | 10000 | 196 | 593 | true | 0.0196 | 0.0169737, 0.0225111 | null | null |
| calibration/C02/90/0.975/A-F | 0.975 | true | 10000 | 275 | 317 | true | 0.0275 | 0.024383, 0.0308961 | null | null |
| calibration/C02/90/0.975/P-F | 0.975 | true | 10000 | 251 | 317 | true | 0.0251 | 0.022123, 0.0283578 | null | null |
| calibration/C02/90/0.975/joint | 0.975 | true | 10000 | 521 | 593 | true | 0.0521 | 0.0478254, 0.0566378 | null | null |
| calibration/C02/90/0.980/A-F | 0.98 | true | 10000 | 228 | 317 | true | 0.0228 | 0.0199641, 0.0259183 | null | null |
| calibration/C02/90/0.980/P-F | 0.98 | true | 10000 | 194 | 317 | true | 0.0194 | 0.0167874, 0.0222975 | null | null |
| calibration/C02/90/0.980/joint | 0.98 | true | 10000 | 419 | 593 | true | 0.0419 | 0.038057, 0.0460126 | null | null |
| calibration/C02/90/0.985/A-F | 0.985 | true | 10000 | 169 | 317 | true | 0.0169 | 0.0144651, 0.0196217 | null | null |
| calibration/C02/90/0.985/P-F | 0.985 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | null |
| calibration/C02/90/0.985/joint | 0.985 | true | 10000 | 313 | 593 | true | 0.0313 | 0.0279742, 0.0349023 | null | null |
| calibration/C02/90/0.990/A-F | 0.99 | true | 10000 | 116 | 317 | true | 0.0116 | 0.00959451, 0.013897 | null | null |
| calibration/C02/90/0.990/P-F | 0.99 | true | 10000 | 101 | 317 | true | 0.0101 | 0.00823393, 0.0122591 | null | null |
| calibration/C02/90/0.990/joint | 0.99 | true | 10000 | 217 | 593 | true | 0.0217 | 0.0189343, 0.0247489 | null | null |
| calibration/C02/99/0.975/A-F | 0.975 | true | 10000 | 281 | 317 | true | 0.0281 | 0.024949, 0.0315297 | null | null |
| calibration/C02/99/0.975/P-F | 0.975 | true | 10000 | 258 | 317 | true | 0.0258 | 0.0227814, 0.0290988 | null | null |
| calibration/C02/99/0.975/joint | 0.975 | true | 10000 | 530 | 593 | true | 0.053 | 0.0486897, 0.0575729 | null | null |
| calibration/C02/99/0.980/A-F | 0.98 | true | 10000 | 227 | 317 | true | 0.0227 | 0.0198704, 0.0258121 | null | null |
| calibration/C02/99/0.980/P-F | 0.98 | true | 10000 | 222 | 317 | true | 0.0222 | 0.0194021, 0.0252807 | null | null |
| calibration/C02/99/0.980/joint | 0.98 | true | 10000 | 444 | 593 | true | 0.0444 | 0.040446, 0.048622 | null | null |
| calibration/C02/99/0.985/A-F | 0.985 | true | 10000 | 173 | 317 | true | 0.0173 | 0.0148358, 0.0200508 | null | null |
| calibration/C02/99/0.985/P-F | 0.985 | true | 10000 | 166 | 317 | true | 0.0166 | 0.0141874, 0.0192998 | null | null |
| calibration/C02/99/0.985/joint | 0.985 | true | 10000 | 336 | 593 | true | 0.0336 | 0.0301546, 0.0373204 | null | null |
| calibration/C02/99/0.990/A-F | 0.99 | true | 10000 | 117 | 317 | true | 0.0117 | 0.00968556, 0.0140058 | null | null |
| calibration/C02/99/0.990/P-F | 0.99 | true | 10000 | 107 | 317 | true | 0.0107 | 0.00877696, 0.0129154 | null | null |
| calibration/C02/99/0.990/joint | 0.99 | true | 10000 | 224 | 593 | true | 0.0224 | 0.0195894, 0.0254933 | null | null |
| calibration/C02/cap/0.975/A-F | 0.975 | true | 10000 | 276 | 317 | true | 0.0276 | 0.0244773, 0.0310017 | null | 60 |
| calibration/C02/cap/0.975/P-F | 0.975 | true | 10000 | 240 | 317 | true | 0.024 | 0.0210896, 0.027192 | null | 60 |
| calibration/C02/cap/0.975/joint | 0.975 | true | 10000 | 508 | 593 | true | 0.0508 | 0.0465775, 0.0552865 | null | 60 |
| calibration/C02/cap/0.980/A-F | 0.98 | true | 10000 | 233 | 317 | true | 0.0233 | 0.0204328, 0.0264493 | null | 60 |
| calibration/C02/cap/0.980/P-F | 0.98 | true | 10000 | 195 | 317 | true | 0.0195 | 0.0168805, 0.0224043 | null | 60 |
| calibration/C02/cap/0.980/joint | 0.98 | true | 10000 | 425 | 593 | true | 0.0425 | 0.03863, 0.0466392 | null | 60 |
| calibration/C02/cap/0.985/A-F | 0.985 | true | 10000 | 168 | 317 | true | 0.0168 | 0.0143725, 0.0195144 | null | 60 |
| calibration/C02/cap/0.985/P-F | 0.985 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | 60 |
| calibration/C02/cap/0.985/joint | 0.985 | true | 10000 | 311 | 593 | true | 0.0311 | 0.0277849, 0.0346918 | null | 60 |
| calibration/C02/cap/0.990/A-F | 0.99 | true | 10000 | 110 | 317 | true | 0.011 | 0.0090491, 0.013243 | null | 60 |
| calibration/C02/cap/0.990/P-F | 0.99 | true | 10000 | 102 | 317 | true | 0.0102 | 0.00832432, 0.0123686 | null | 60 |
| calibration/C02/cap/0.990/joint | 0.99 | true | 10000 | 210 | 593 | true | 0.021 | 0.0182799, 0.0240038 | null | 60 |
| calibration/C03/36/0.975/A-F | 0.975 | true | 10000 | 345 | 317 | false | 0.0345 | 0.031009, 0.0382654 | null | null |
| calibration/C03/36/0.975/P-F | 0.975 | true | 10000 | 304 | 317 | true | 0.0304 | 0.0271224, 0.0339547 | null | null |
| calibration/C03/36/0.975/joint | 0.975 | true | 10000 | 636 | 593 | false | 0.0636 | 0.0588946, 0.0685615 | null | null |
| calibration/C03/36/0.980/A-F | 0.98 | true | 10000 | 292 | 317 | true | 0.0292 | 0.0259878, 0.0326901 | null | null |
| calibration/C03/36/0.980/P-F | 0.98 | true | 10000 | 243 | 317 | true | 0.0243 | 0.0213712, 0.0275101 | null | null |
| calibration/C03/36/0.980/joint | 0.98 | true | 10000 | 526 | 593 | true | 0.0526 | 0.0483055, 0.0571573 | null | null |
| calibration/C03/36/0.985/A-F | 0.985 | true | 10000 | 212 | 317 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| calibration/C03/36/0.985/P-F | 0.985 | true | 10000 | 185 | 317 | true | 0.0185 | 0.0159499, 0.0213357 | null | null |
| calibration/C03/36/0.985/joint | 0.985 | true | 10000 | 391 | 593 | true | 0.0391 | 0.0353857, 0.0430857 | null | null |
| calibration/C03/36/0.990/A-F | 0.99 | true | 10000 | 145 | 317 | true | 0.0145 | 0.0122493, 0.0170396 | null | null |
| calibration/C03/36/0.990/P-F | 0.99 | true | 10000 | 134 | 317 | true | 0.0134 | 0.011239, 0.0158508 | null | null |
| calibration/C03/36/0.990/joint | 0.99 | true | 10000 | 276 | 593 | true | 0.0276 | 0.0244773, 0.0310017 | null | null |
| calibration/C03/45/0.975/A-F | 0.975 | true | 10000 | 312 | 317 | true | 0.0312 | 0.0278796, 0.034797 | null | null |
| calibration/C03/45/0.975/P-F | 0.975 | true | 10000 | 322 | 317 | false | 0.0322 | 0.0288269, 0.035849 | null | null |
| calibration/C03/45/0.975/joint | 0.975 | true | 10000 | 620 | 593 | false | 0.062 | 0.0573515, 0.0669055 | null | null |
| calibration/C03/45/0.980/A-F | 0.98 | true | 10000 | 260 | 317 | true | 0.026 | 0.0229697, 0.0293105 | null | null |
| calibration/C03/45/0.980/P-F | 0.98 | true | 10000 | 262 | 317 | true | 0.0262 | 0.023158, 0.029522 | null | null |
| calibration/C03/45/0.980/joint | 0.98 | true | 10000 | 513 | 593 | true | 0.0513 | 0.0470573, 0.0558063 | null | null |
| calibration/C03/45/0.985/A-F | 0.985 | true | 10000 | 209 | 317 | true | 0.0209 | 0.0181865, 0.0238973 | null | null |
| calibration/C03/45/0.985/P-F | 0.985 | true | 10000 | 201 | 317 | true | 0.0201 | 0.0174398, 0.0230446 | null | null |
| calibration/C03/45/0.985/joint | 0.985 | true | 10000 | 404 | 593 | true | 0.0404 | 0.0366253, 0.0444452 | null | null |
| calibration/C03/45/0.990/A-F | 0.99 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | null |
| calibration/C03/45/0.990/P-F | 0.99 | true | 10000 | 138 | 317 | true | 0.0138 | 0.011606, 0.0162835 | null | null |
| calibration/C03/45/0.990/joint | 0.99 | true | 10000 | 292 | 593 | true | 0.0292 | 0.0259878, 0.0326901 | null | null |
| calibration/C03/60/0.975/A-F | 0.975 | true | 10000 | 314 | 317 | true | 0.0314 | 0.0280689, 0.0350075 | null | null |
| calibration/C03/60/0.975/P-F | 0.975 | true | 10000 | 289 | 317 | true | 0.0289 | 0.0257044, 0.0323738 | null | null |
| calibration/C03/60/0.975/joint | 0.975 | true | 10000 | 589 | 593 | true | 0.0589 | 0.0543645, 0.0636944 | null | null |
| calibration/C03/60/0.980/A-F | 0.98 | true | 10000 | 255 | 317 | true | 0.0255 | 0.0224992, 0.0287813 | null | null |
| calibration/C03/60/0.980/P-F | 0.98 | true | 10000 | 228 | 317 | true | 0.0228 | 0.0199641, 0.0259183 | null | null |
| calibration/C03/60/0.980/joint | 0.98 | true | 10000 | 474 | 593 | true | 0.0474 | 0.0433176, 0.0517485 | null | null |
| calibration/C03/60/0.985/A-F | 0.985 | true | 10000 | 205 | 317 | true | 0.0205 | 0.017813, 0.0234711 | null | null |
| calibration/C03/60/0.985/P-F | 0.985 | true | 10000 | 177 | 317 | true | 0.0177 | 0.0152068, 0.0204794 | null | null |
| calibration/C03/60/0.985/joint | 0.985 | true | 10000 | 379 | 593 | true | 0.0379 | 0.0342425, 0.0418297 | null | null |
| calibration/C03/60/0.990/A-F | 0.99 | true | 10000 | 141 | 317 | true | 0.0141 | 0.0118815, 0.0166077 | null | null |
| calibration/C03/60/0.990/P-F | 0.99 | true | 10000 | 122 | 317 | true | 0.0122 | 0.0101414, 0.0145495 | null | null |
| calibration/C03/60/0.990/joint | 0.99 | true | 10000 | 262 | 593 | true | 0.0262 | 0.023158, 0.029522 | null | null |
| calibration/C03/75/0.975/A-F | 0.975 | true | 10000 | 309 | 317 | true | 0.0309 | 0.0275955, 0.0344812 | null | null |
| calibration/C03/75/0.975/P-F | 0.975 | true | 10000 | 322 | 317 | false | 0.0322 | 0.0288269, 0.035849 | null | null |
| calibration/C03/75/0.975/joint | 0.975 | true | 10000 | 622 | 593 | false | 0.0622 | 0.0575444, 0.0671126 | null | null |
| calibration/C03/75/0.980/A-F | 0.98 | true | 10000 | 249 | 317 | true | 0.0249 | 0.021935, 0.0281459 | null | null |
| calibration/C03/75/0.980/P-F | 0.98 | true | 10000 | 267 | 317 | true | 0.0267 | 0.0236289, 0.0300508 | null | null |
| calibration/C03/75/0.980/joint | 0.98 | true | 10000 | 509 | 593 | true | 0.0509 | 0.0466734, 0.0553905 | null | null |
| calibration/C03/75/0.985/A-F | 0.985 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| calibration/C03/75/0.985/P-F | 0.985 | true | 10000 | 198 | 317 | true | 0.0198 | 0.0171601, 0.0227245 | null | null |
| calibration/C03/75/0.985/joint | 0.985 | true | 10000 | 397 | 593 | true | 0.0397 | 0.0359577, 0.0437133 | null | null |
| calibration/C03/75/0.990/A-F | 0.99 | true | 10000 | 148 | 317 | true | 0.0148 | 0.0125254, 0.0173632 | null | null |
| calibration/C03/75/0.990/P-F | 0.99 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | null |
| calibration/C03/75/0.990/joint | 0.99 | true | 10000 | 293 | 593 | true | 0.0293 | 0.0260823, 0.0327956 | null | null |
| calibration/C03/90/0.975/A-F | 0.975 | true | 10000 | 328 | 317 | false | 0.0328 | 0.0293957, 0.0364798 | null | null |
| calibration/C03/90/0.975/P-F | 0.975 | true | 10000 | 299 | 317 | true | 0.0299 | 0.0266494, 0.033428 | null | null |
| calibration/C03/90/0.975/joint | 0.975 | true | 10000 | 620 | 593 | false | 0.062 | 0.0573515, 0.0669055 | null | null |
| calibration/C03/90/0.980/A-F | 0.98 | true | 10000 | 265 | 317 | true | 0.0265 | 0.0234405, 0.0298393 | null | null |
| calibration/C03/90/0.980/P-F | 0.98 | true | 10000 | 245 | 317 | true | 0.0245 | 0.0215591, 0.0277221 | null | null |
| calibration/C03/90/0.980/joint | 0.98 | true | 10000 | 506 | 593 | true | 0.0506 | 0.0463856, 0.0550785 | null | null |
| calibration/C03/90/0.985/A-F | 0.985 | true | 10000 | 209 | 317 | true | 0.0209 | 0.0181865, 0.0238973 | null | null |
| calibration/C03/90/0.985/P-F | 0.985 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| calibration/C03/90/0.985/joint | 0.985 | true | 10000 | 395 | 593 | true | 0.0395 | 0.035767, 0.0435041 | null | null |
| calibration/C03/90/0.990/A-F | 0.99 | true | 10000 | 153 | 317 | true | 0.0153 | 0.0129862, 0.0179019 | null | null |
| calibration/C03/90/0.990/P-F | 0.99 | true | 10000 | 136 | 317 | true | 0.0136 | 0.0114224, 0.0160672 | null | null |
| calibration/C03/90/0.990/joint | 0.99 | true | 10000 | 288 | 593 | true | 0.0288 | 0.0256099, 0.0322683 | null | null |
| calibration/C03/99/0.975/A-F | 0.975 | true | 10000 | 294 | 317 | true | 0.0294 | 0.0261768, 0.032901 | null | null |
| calibration/C03/99/0.975/P-F | 0.975 | true | 10000 | 295 | 317 | true | 0.0295 | 0.0262713, 0.0330064 | null | null |
| calibration/C03/99/0.975/joint | 0.975 | true | 10000 | 576 | 593 | true | 0.0576 | 0.0531129, 0.0623468 | null | null |
| calibration/C03/99/0.980/A-F | 0.98 | true | 10000 | 244 | 317 | true | 0.0244 | 0.0214652, 0.0276161 | null | null |
| calibration/C03/99/0.980/P-F | 0.98 | true | 10000 | 249 | 317 | true | 0.0249 | 0.021935, 0.0281459 | null | null |
| calibration/C03/99/0.980/joint | 0.98 | true | 10000 | 484 | 593 | true | 0.0484 | 0.0442758, 0.0527897 | null | null |
| calibration/C03/99/0.985/A-F | 0.985 | true | 10000 | 185 | 317 | true | 0.0185 | 0.0159499, 0.0213357 | null | null |
| calibration/C03/99/0.985/P-F | 0.985 | true | 10000 | 185 | 317 | true | 0.0185 | 0.0159499, 0.0213357 | null | null |
| calibration/C03/99/0.985/joint | 0.985 | true | 10000 | 364 | 593 | true | 0.0364 | 0.0328149, 0.0402583 | null | null |
| calibration/C03/99/0.990/A-F | 0.99 | true | 10000 | 130 | 317 | true | 0.013 | 0.0108726, 0.0154176 | null | null |
| calibration/C03/99/0.990/P-F | 0.99 | true | 10000 | 137 | 317 | true | 0.0137 | 0.0115142, 0.0161754 | null | null |
| calibration/C03/99/0.990/joint | 0.99 | true | 10000 | 265 | 593 | true | 0.0265 | 0.0234405, 0.0298393 | null | null |
| calibration/C03/cap/0.975/A-F | 0.975 | true | 10000 | 309 | 317 | true | 0.0309 | 0.0275955, 0.0344812 | null | 60 |
| calibration/C03/cap/0.975/P-F | 0.975 | true | 10000 | 299 | 317 | true | 0.0299 | 0.0266494, 0.033428 | null | 60 |
| calibration/C03/cap/0.975/joint | 0.975 | true | 10000 | 600 | 593 | false | 0.06 | 0.055424, 0.0648343 | null | 60 |
| calibration/C03/cap/0.980/A-F | 0.98 | true | 10000 | 254 | 317 | true | 0.0254 | 0.0224051, 0.0286754 | null | 60 |
| calibration/C03/cap/0.980/P-F | 0.98 | true | 10000 | 236 | 317 | true | 0.0236 | 0.0207142, 0.0267676 | null | 60 |
| calibration/C03/cap/0.980/joint | 0.98 | true | 10000 | 488 | 593 | true | 0.0488 | 0.0446592, 0.053206 | null | 60 |
| calibration/C03/cap/0.985/A-F | 0.985 | true | 10000 | 210 | 317 | true | 0.021 | 0.0182799, 0.0240038 | null | 60 |
| calibration/C03/cap/0.985/P-F | 0.985 | true | 10000 | 186 | 317 | true | 0.0186 | 0.0160429, 0.0214427 | null | 60 |
| calibration/C03/cap/0.985/joint | 0.985 | true | 10000 | 396 | 593 | true | 0.0396 | 0.0358624, 0.0436087 | null | 60 |
| calibration/C03/cap/0.990/A-F | 0.99 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | 60 |
| calibration/C03/cap/0.990/P-F | 0.99 | true | 10000 | 124 | 317 | true | 0.0124 | 0.010324, 0.0147668 | null | 60 |
| calibration/C03/cap/0.990/joint | 0.99 | true | 10000 | 280 | 593 | true | 0.028 | 0.0248547, 0.0314241 | null | 60 |
| calibration/C04/36/0.975/A-F | 0.975 | true | 10000 | 354 | 317 | false | 0.0354 | 0.0318641, 0.0392097 | null | null |
| calibration/C04/36/0.975/P-F | 0.975 | true | 10000 | 336 | 317 | false | 0.0336 | 0.0301546, 0.0373204 | null | null |
| calibration/C04/36/0.975/joint | 0.975 | true | 10000 | 675 | 593 | false | 0.0675 | 0.0626592, 0.0725945 | null | null |
| calibration/C04/36/0.980/A-F | 0.98 | true | 10000 | 294 | 317 | true | 0.0294 | 0.0261768, 0.032901 | null | null |
| calibration/C04/36/0.980/P-F | 0.98 | true | 10000 | 285 | 317 | true | 0.0285 | 0.0253266, 0.0319518 | null | null |
| calibration/C04/36/0.980/joint | 0.98 | true | 10000 | 570 | 593 | true | 0.057 | 0.0525355, 0.0617246 | null | null |
| calibration/C04/36/0.985/A-F | 0.985 | true | 10000 | 214 | 317 | true | 0.0214 | 0.0186537, 0.0244297 | null | null |
| calibration/C04/36/0.985/P-F | 0.985 | true | 10000 | 238 | 317 | true | 0.0238 | 0.0209018, 0.0269798 | null | null |
| calibration/C04/36/0.985/joint | 0.985 | true | 10000 | 445 | 593 | true | 0.0445 | 0.0405417, 0.0487262 | null | null |
| calibration/C04/36/0.990/A-F | 0.99 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| calibration/C04/36/0.990/P-F | 0.99 | true | 10000 | 171 | 317 | true | 0.0171 | 0.0146504, 0.0198363 | null | null |
| calibration/C04/36/0.990/joint | 0.99 | true | 10000 | 317 | 593 | true | 0.0317 | 0.0283531, 0.0353231 | null | null |
| calibration/C04/45/0.975/A-F | 0.975 | true | 10000 | 357 | 317 | false | 0.0357 | 0.0321492, 0.0395244 | null | null |
| calibration/C04/45/0.975/P-F | 0.975 | true | 10000 | 333 | 317 | false | 0.0333 | 0.0298699, 0.0370052 | null | null |
| calibration/C04/45/0.975/joint | 0.975 | true | 10000 | 678 | 593 | false | 0.0678 | 0.062949, 0.0729045 | null | null |
| calibration/C04/45/0.980/A-F | 0.98 | true | 10000 | 296 | 317 | true | 0.0296 | 0.0263658, 0.0331118 | null | null |
| calibration/C04/45/0.980/P-F | 0.98 | true | 10000 | 282 | 317 | true | 0.0282 | 0.0250434, 0.0316352 | null | null |
| calibration/C04/45/0.980/joint | 0.98 | true | 10000 | 568 | 593 | true | 0.0568 | 0.0523431, 0.0615172 | null | null |
| calibration/C04/45/0.985/A-F | 0.985 | true | 10000 | 235 | 317 | true | 0.0235 | 0.0206204, 0.0266615 | null | null |
| calibration/C04/45/0.985/P-F | 0.985 | true | 10000 | 211 | 317 | true | 0.0211 | 0.0183733, 0.0241103 | null | null |
| calibration/C04/45/0.985/joint | 0.985 | true | 10000 | 443 | 593 | true | 0.0443 | 0.0403504, 0.0485176 | null | null |
| calibration/C04/45/0.990/A-F | 0.99 | true | 10000 | 171 | 317 | true | 0.0171 | 0.0146504, 0.0198363 | null | null |
| calibration/C04/45/0.990/P-F | 0.99 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| calibration/C04/45/0.990/joint | 0.99 | true | 10000 | 321 | 593 | true | 0.0321 | 0.0287321, 0.0357439 | null | null |
| calibration/C04/60/0.975/A-F | 0.975 | true | 10000 | 334 | 317 | false | 0.0334 | 0.0299648, 0.0371103 | null | null |
| calibration/C04/60/0.975/P-F | 0.975 | true | 10000 | 305 | 317 | true | 0.0305 | 0.027217, 0.0340601 | null | null |
| calibration/C04/60/0.975/joint | 0.975 | true | 10000 | 623 | 593 | false | 0.0623 | 0.0576408, 0.0672161 | null | null |
| calibration/C04/60/0.980/A-F | 0.98 | true | 10000 | 288 | 317 | true | 0.0288 | 0.0256099, 0.0322683 | null | null |
| calibration/C04/60/0.980/P-F | 0.98 | true | 10000 | 264 | 317 | true | 0.0264 | 0.0233463, 0.0297336 | null | null |
| calibration/C04/60/0.980/joint | 0.98 | true | 10000 | 543 | 593 | true | 0.0543 | 0.0499389, 0.0589229 | null | null |
| calibration/C04/60/0.985/A-F | 0.985 | true | 10000 | 237 | 317 | true | 0.0237 | 0.020808, 0.0268738 | null | null |
| calibration/C04/60/0.985/P-F | 0.985 | true | 10000 | 203 | 317 | true | 0.0203 | 0.0176264, 0.0232579 | null | null |
| calibration/C04/60/0.985/joint | 0.985 | true | 10000 | 436 | 593 | true | 0.0436 | 0.0396811, 0.0477874 | null | null |
| calibration/C04/60/0.990/A-F | 0.99 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | null |
| calibration/C04/60/0.990/P-F | 0.99 | true | 10000 | 141 | 317 | true | 0.0141 | 0.0118815, 0.0166077 | null | null |
| calibration/C04/60/0.990/joint | 0.99 | true | 10000 | 296 | 593 | true | 0.0296 | 0.0263658, 0.0331118 | null | null |
| calibration/C04/75/0.975/A-F | 0.975 | true | 10000 | 325 | 317 | false | 0.0325 | 0.0291112, 0.0361645 | null | null |
| calibration/C04/75/0.975/P-F | 0.975 | true | 10000 | 297 | 317 | true | 0.0297 | 0.0264603, 0.0332172 | null | null |
| calibration/C04/75/0.975/joint | 0.975 | true | 10000 | 612 | 593 | false | 0.0612 | 0.0565804, 0.0660772 | null | null |
| calibration/C04/75/0.980/A-F | 0.98 | true | 10000 | 277 | 317 | true | 0.0277 | 0.0245716, 0.0311073 | null | null |
| calibration/C04/75/0.980/P-F | 0.98 | true | 10000 | 241 | 317 | true | 0.0241 | 0.0211834, 0.027298 | null | null |
| calibration/C04/75/0.980/joint | 0.98 | true | 10000 | 510 | 593 | true | 0.051 | 0.0467694, 0.0554944 | null | null |
| calibration/C04/75/0.985/A-F | 0.985 | true | 10000 | 213 | 317 | true | 0.0213 | 0.0185602, 0.0243233 | null | null |
| calibration/C04/75/0.985/P-F | 0.985 | true | 10000 | 193 | 317 | true | 0.0193 | 0.0166942, 0.0221908 | null | null |
| calibration/C04/75/0.985/joint | 0.985 | true | 10000 | 401 | 593 | true | 0.0401 | 0.0363392, 0.0441316 | null | null |
| calibration/C04/75/0.990/A-F | 0.99 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | null |
| calibration/C04/75/0.990/P-F | 0.99 | true | 10000 | 143 | 317 | true | 0.0143 | 0.0120653, 0.0168237 | null | null |
| calibration/C04/75/0.990/joint | 0.99 | true | 10000 | 297 | 593 | true | 0.0297 | 0.0264603, 0.0332172 | null | null |
| calibration/C04/90/0.975/A-F | 0.975 | true | 10000 | 301 | 317 | true | 0.0301 | 0.0268386, 0.0336387 | null | null |
| calibration/C04/90/0.975/P-F | 0.975 | true | 10000 | 294 | 317 | true | 0.0294 | 0.0261768, 0.032901 | null | null |
| calibration/C04/90/0.975/joint | 0.975 | true | 10000 | 587 | 593 | true | 0.0587 | 0.0541719, 0.0634872 | null | null |
| calibration/C04/90/0.980/A-F | 0.98 | true | 10000 | 240 | 317 | true | 0.024 | 0.0210896, 0.027192 | null | null |
| calibration/C04/90/0.980/P-F | 0.98 | true | 10000 | 243 | 317 | true | 0.0243 | 0.0213712, 0.0275101 | null | null |
| calibration/C04/90/0.980/joint | 0.98 | true | 10000 | 477 | 593 | true | 0.0477 | 0.043605, 0.0520609 | null | null |
| calibration/C04/90/0.985/A-F | 0.985 | true | 10000 | 182 | 317 | true | 0.0182 | 0.0156711, 0.0210148 | null | null |
| calibration/C04/90/0.985/P-F | 0.985 | true | 10000 | 200 | 317 | true | 0.02 | 0.0173465, 0.0229379 | null | null |
| calibration/C04/90/0.985/joint | 0.985 | true | 10000 | 378 | 593 | true | 0.0378 | 0.0341473, 0.041725 | null | null |
| calibration/C04/90/0.990/A-F | 0.99 | true | 10000 | 128 | 317 | true | 0.0128 | 0.0106896, 0.0152008 | null | null |
| calibration/C04/90/0.990/P-F | 0.99 | true | 10000 | 145 | 317 | true | 0.0145 | 0.0122493, 0.0170396 | null | null |
| calibration/C04/90/0.990/joint | 0.99 | true | 10000 | 271 | 593 | true | 0.0271 | 0.0240058, 0.0304735 | null | null |
| calibration/C04/99/0.975/A-F | 0.975 | true | 10000 | 291 | 317 | true | 0.0291 | 0.0258933, 0.0325847 | null | null |
| calibration/C04/99/0.975/P-F | 0.975 | true | 10000 | 292 | 317 | true | 0.0292 | 0.0259878, 0.0326901 | null | null |
| calibration/C04/99/0.975/joint | 0.975 | true | 10000 | 575 | 593 | true | 0.0575 | 0.0530167, 0.0622431 | null | null |
| calibration/C04/99/0.980/A-F | 0.98 | true | 10000 | 227 | 317 | true | 0.0227 | 0.0198704, 0.0258121 | null | null |
| calibration/C04/99/0.980/P-F | 0.98 | true | 10000 | 240 | 317 | true | 0.024 | 0.0210896, 0.027192 | null | null |
| calibration/C04/99/0.980/joint | 0.98 | true | 10000 | 462 | 593 | true | 0.0462 | 0.0421684, 0.0504985 | null | null |
| calibration/C04/99/0.985/A-F | 0.985 | true | 10000 | 178 | 317 | true | 0.0178 | 0.0152996, 0.0205865 | null | null |
| calibration/C04/99/0.985/P-F | 0.985 | true | 10000 | 196 | 317 | true | 0.0196 | 0.0169737, 0.0225111 | null | null |
| calibration/C04/99/0.985/joint | 0.985 | true | 10000 | 372 | 593 | true | 0.0372 | 0.0335761, 0.0410965 | null | null |
| calibration/C04/99/0.990/A-F | 0.99 | true | 10000 | 124 | 317 | true | 0.0124 | 0.010324, 0.0147668 | null | null |
| calibration/C04/99/0.990/P-F | 0.99 | true | 10000 | 141 | 317 | true | 0.0141 | 0.0118815, 0.0166077 | null | null |
| calibration/C04/99/0.990/joint | 0.99 | true | 10000 | 265 | 593 | true | 0.0265 | 0.0234405, 0.0298393 | null | null |
| calibration/C04/cap/0.975/A-F | 0.975 | true | 10000 | 312 | 317 | true | 0.0312 | 0.0278796, 0.034797 | null | 60 |
| calibration/C04/cap/0.975/P-F | 0.975 | true | 10000 | 304 | 317 | true | 0.0304 | 0.0271224, 0.0339547 | null | 60 |
| calibration/C04/cap/0.975/joint | 0.975 | true | 10000 | 609 | 593 | false | 0.0609 | 0.0562912, 0.0657665 | null | 60 |
| calibration/C04/cap/0.980/A-F | 0.98 | true | 10000 | 252 | 317 | true | 0.0252 | 0.022217, 0.0284637 | null | 60 |
| calibration/C04/cap/0.980/P-F | 0.98 | true | 10000 | 254 | 317 | true | 0.0254 | 0.0224051, 0.0286754 | null | 60 |
| calibration/C04/cap/0.980/joint | 0.98 | true | 10000 | 505 | 593 | true | 0.0505 | 0.0462896, 0.0549745 | null | 60 |
| calibration/C04/cap/0.985/A-F | 0.985 | true | 10000 | 189 | 317 | true | 0.0189 | 0.0163219, 0.0217634 | null | 60 |
| calibration/C04/cap/0.985/P-F | 0.985 | true | 10000 | 192 | 317 | true | 0.0192 | 0.0166011, 0.0220839 | null | 60 |
| calibration/C04/cap/0.985/joint | 0.985 | true | 10000 | 379 | 593 | true | 0.0379 | 0.0342425, 0.0418297 | null | 60 |
| calibration/C04/cap/0.990/A-F | 0.99 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | 60 |
| calibration/C04/cap/0.990/P-F | 0.99 | true | 10000 | 124 | 317 | true | 0.0124 | 0.010324, 0.0147668 | null | 60 |
| calibration/C04/cap/0.990/joint | 0.99 | true | 10000 | 273 | 593 | true | 0.0273 | 0.0241944, 0.0306848 | null | 60 |
| calibration/C05/36/0.975/A-F | 0.975 | true | 10000 | 311 | 317 | true | 0.0311 | 0.0277849, 0.0346918 | null | null |
| calibration/C05/36/0.975/P-F | 0.975 | true | 10000 | 273 | 317 | true | 0.0273 | 0.0241944, 0.0306848 | null | null |
| calibration/C05/36/0.975/joint | 0.975 | true | 10000 | 575 | 593 | true | 0.0575 | 0.0530167, 0.0622431 | null | null |
| calibration/C05/36/0.980/A-F | 0.98 | true | 10000 | 255 | 317 | true | 0.0255 | 0.0224992, 0.0287813 | null | null |
| calibration/C05/36/0.980/P-F | 0.98 | true | 10000 | 219 | 317 | true | 0.0219 | 0.0191214, 0.0249617 | null | null |
| calibration/C05/36/0.980/joint | 0.98 | true | 10000 | 466 | 593 | true | 0.0466 | 0.0425514, 0.0509153 | null | null |
| calibration/C05/36/0.985/A-F | 0.985 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| calibration/C05/36/0.985/P-F | 0.985 | true | 10000 | 165 | 317 | true | 0.0165 | 0.0140948, 0.0191924 | null | null |
| calibration/C05/36/0.985/joint | 0.985 | true | 10000 | 351 | 593 | true | 0.0351 | 0.031579, 0.038895 | null | null |
| calibration/C05/36/0.990/A-F | 0.99 | true | 10000 | 141 | 317 | true | 0.0141 | 0.0118815, 0.0166077 | null | null |
| calibration/C05/36/0.990/P-F | 0.99 | true | 10000 | 115 | 317 | true | 0.0115 | 0.00950351, 0.0137881 | null | null |
| calibration/C05/36/0.990/joint | 0.99 | true | 10000 | 254 | 593 | true | 0.0254 | 0.0224051, 0.0286754 | null | null |
| calibration/C05/45/0.975/A-F | 0.975 | true | 10000 | 283 | 317 | true | 0.0283 | 0.0251378, 0.0317407 | null | null |
| calibration/C05/45/0.975/P-F | 0.975 | true | 10000 | 282 | 317 | true | 0.0282 | 0.0250434, 0.0316352 | null | null |
| calibration/C05/45/0.975/joint | 0.975 | true | 10000 | 551 | 593 | true | 0.0551 | 0.0507079, 0.0597533 | null | null |
| calibration/C05/45/0.980/A-F | 0.98 | true | 10000 | 234 | 317 | true | 0.0234 | 0.0205266, 0.0265554 | null | null |
| calibration/C05/45/0.980/P-F | 0.98 | true | 10000 | 223 | 317 | true | 0.0223 | 0.0194957, 0.025387 | null | null |
| calibration/C05/45/0.980/joint | 0.98 | true | 10000 | 448 | 593 | true | 0.0448 | 0.0408286, 0.0490391 | null | null |
| calibration/C05/45/0.985/A-F | 0.985 | true | 10000 | 184 | 317 | true | 0.0184 | 0.0158569, 0.0212288 | null | null |
| calibration/C05/45/0.985/P-F | 0.985 | true | 10000 | 164 | 317 | true | 0.0164 | 0.0140023, 0.019085 | null | null |
| calibration/C05/45/0.985/joint | 0.985 | true | 10000 | 344 | 593 | true | 0.0344 | 0.030914, 0.0381604 | null | null |
| calibration/C05/45/0.990/A-F | 0.99 | true | 10000 | 131 | 317 | true | 0.0131 | 0.0109641, 0.0155259 | null | null |
| calibration/C05/45/0.990/P-F | 0.99 | true | 10000 | 107 | 317 | true | 0.0107 | 0.00877696, 0.0129154 | null | null |
| calibration/C05/45/0.990/joint | 0.99 | true | 10000 | 236 | 593 | true | 0.0236 | 0.0207142, 0.0267676 | null | null |
| calibration/C05/60/0.975/A-F | 0.975 | true | 10000 | 276 | 317 | true | 0.0276 | 0.0244773, 0.0310017 | null | null |
| calibration/C05/60/0.975/P-F | 0.975 | true | 10000 | 297 | 317 | true | 0.0297 | 0.0264603, 0.0332172 | null | null |
| calibration/C05/60/0.975/joint | 0.975 | true | 10000 | 561 | 593 | true | 0.0561 | 0.0516696, 0.060791 | null | null |
| calibration/C05/60/0.980/A-F | 0.98 | true | 10000 | 223 | 317 | true | 0.0223 | 0.0194957, 0.025387 | null | null |
| calibration/C05/60/0.980/P-F | 0.98 | true | 10000 | 242 | 317 | true | 0.0242 | 0.0212773, 0.0274041 | null | null |
| calibration/C05/60/0.980/joint | 0.98 | true | 10000 | 460 | 593 | true | 0.046 | 0.0419769, 0.0502901 | null | null |
| calibration/C05/60/0.985/A-F | 0.985 | true | 10000 | 175 | 317 | true | 0.0175 | 0.0150213, 0.0202651 | null | null |
| calibration/C05/60/0.985/P-F | 0.985 | true | 10000 | 181 | 317 | true | 0.0181 | 0.0155782, 0.0209077 | null | null |
| calibration/C05/60/0.985/joint | 0.985 | true | 10000 | 355 | 593 | true | 0.0355 | 0.0319591, 0.0393146 | null | null |
| calibration/C05/60/0.990/A-F | 0.99 | true | 10000 | 115 | 317 | true | 0.0115 | 0.00950351, 0.0137881 | null | null |
| calibration/C05/60/0.990/P-F | 0.99 | true | 10000 | 118 | 317 | true | 0.0118 | 0.00977665, 0.0141146 | null | null |
| calibration/C05/60/0.990/joint | 0.99 | true | 10000 | 233 | 593 | true | 0.0233 | 0.0204328, 0.0264493 | null | null |
| calibration/C05/75/0.975/A-F | 0.975 | true | 10000 | 276 | 317 | true | 0.0276 | 0.0244773, 0.0310017 | null | null |
| calibration/C05/75/0.975/P-F | 0.975 | true | 10000 | 298 | 317 | true | 0.0298 | 0.0265549, 0.0333226 | null | null |
| calibration/C05/75/0.975/joint | 0.975 | true | 10000 | 569 | 593 | true | 0.0569 | 0.0524393, 0.0616209 | null | null |
| calibration/C05/75/0.980/A-F | 0.98 | true | 10000 | 226 | 317 | true | 0.0226 | 0.0197767, 0.0257058 | null | null |
| calibration/C05/75/0.980/P-F | 0.98 | true | 10000 | 235 | 317 | true | 0.0235 | 0.0206204, 0.0266615 | null | null |
| calibration/C05/75/0.980/joint | 0.98 | true | 10000 | 459 | 593 | true | 0.0459 | 0.0418812, 0.0501858 | null | null |
| calibration/C05/75/0.985/A-F | 0.985 | true | 10000 | 176 | 317 | true | 0.0176 | 0.015114, 0.0203723 | null | null |
| calibration/C05/75/0.985/P-F | 0.985 | true | 10000 | 171 | 317 | true | 0.0171 | 0.0146504, 0.0198363 | null | null |
| calibration/C05/75/0.985/joint | 0.985 | true | 10000 | 346 | 593 | true | 0.0346 | 0.031104, 0.0383703 | null | null |
| calibration/C05/75/0.990/A-F | 0.99 | true | 10000 | 124 | 317 | true | 0.0124 | 0.010324, 0.0147668 | null | null |
| calibration/C05/75/0.990/P-F | 0.99 | true | 10000 | 117 | 317 | true | 0.0117 | 0.00968556, 0.0140058 | null | null |
| calibration/C05/75/0.990/joint | 0.99 | true | 10000 | 240 | 593 | true | 0.024 | 0.0210896, 0.027192 | null | null |
| calibration/C05/90/0.975/A-F | 0.975 | true | 10000 | 255 | 317 | true | 0.0255 | 0.0224992, 0.0287813 | null | null |
| calibration/C05/90/0.975/P-F | 0.975 | true | 10000 | 282 | 317 | true | 0.0282 | 0.0250434, 0.0316352 | null | null |
| calibration/C05/90/0.975/joint | 0.975 | true | 10000 | 527 | 593 | true | 0.0527 | 0.0484016, 0.0572612 | null | null |
| calibration/C05/90/0.980/A-F | 0.98 | true | 10000 | 215 | 317 | true | 0.0215 | 0.0187472, 0.0245361 | null | null |
| calibration/C05/90/0.980/P-F | 0.98 | true | 10000 | 222 | 317 | true | 0.0222 | 0.0194021, 0.0252807 | null | null |
| calibration/C05/90/0.980/joint | 0.98 | true | 10000 | 430 | 593 | true | 0.043 | 0.0391077, 0.0471612 | null | null |
| calibration/C05/90/0.985/A-F | 0.985 | true | 10000 | 159 | 317 | true | 0.0159 | 0.0135401, 0.0185476 | null | null |
| calibration/C05/90/0.985/P-F | 0.985 | true | 10000 | 168 | 317 | true | 0.0168 | 0.0143725, 0.0195144 | null | null |
| calibration/C05/90/0.985/joint | 0.985 | true | 10000 | 323 | 593 | true | 0.0323 | 0.0289216, 0.0359542 | null | null |
| calibration/C05/90/0.990/A-F | 0.99 | true | 10000 | 118 | 317 | true | 0.0118 | 0.00977665, 0.0141146 | null | null |
| calibration/C05/90/0.990/P-F | 0.99 | true | 10000 | 110 | 317 | true | 0.011 | 0.0090491, 0.013243 | null | null |
| calibration/C05/90/0.990/joint | 0.99 | true | 10000 | 227 | 593 | true | 0.0227 | 0.0198704, 0.0258121 | null | null |
| calibration/C05/99/0.975/A-F | 0.975 | true | 10000 | 257 | 317 | true | 0.0257 | 0.0226873, 0.028993 | null | null |
| calibration/C05/99/0.975/P-F | 0.975 | true | 10000 | 273 | 317 | true | 0.0273 | 0.0241944, 0.0306848 | null | null |
| calibration/C05/99/0.975/joint | 0.975 | true | 10000 | 519 | 593 | true | 0.0519 | 0.0476333, 0.0564299 | null | null |
| calibration/C05/99/0.980/A-F | 0.98 | true | 10000 | 204 | 317 | true | 0.0204 | 0.0177197, 0.0233645 | null | null |
| calibration/C05/99/0.980/P-F | 0.98 | true | 10000 | 221 | 317 | true | 0.0221 | 0.0193085, 0.0251744 | null | null |
| calibration/C05/99/0.980/joint | 0.98 | true | 10000 | 420 | 593 | true | 0.042 | 0.0381524, 0.0461171 | null | null |
| calibration/C05/99/0.985/A-F | 0.985 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | null |
| calibration/C05/99/0.985/P-F | 0.985 | true | 10000 | 166 | 317 | true | 0.0166 | 0.0141874, 0.0192998 | null | null |
| calibration/C05/99/0.985/joint | 0.985 | true | 10000 | 311 | 593 | true | 0.0311 | 0.0277849, 0.0346918 | null | null |
| calibration/C05/99/0.990/A-F | 0.99 | true | 10000 | 108 | 317 | true | 0.0108 | 0.00886763, 0.0130246 | null | null |
| calibration/C05/99/0.990/P-F | 0.99 | true | 10000 | 117 | 317 | true | 0.0117 | 0.00968556, 0.0140058 | null | null |
| calibration/C05/99/0.990/joint | 0.99 | true | 10000 | 222 | 593 | true | 0.0222 | 0.0194021, 0.0252807 | null | null |
| calibration/C05/cap/0.975/A-F | 0.975 | true | 10000 | 265 | 317 | true | 0.0265 | 0.0234405, 0.0298393 | null | 60 |
| calibration/C05/cap/0.975/P-F | 0.975 | true | 10000 | 278 | 317 | true | 0.0278 | 0.0246659, 0.0312129 | null | 60 |
| calibration/C05/cap/0.975/joint | 0.975 | true | 10000 | 529 | 593 | true | 0.0529 | 0.0485937, 0.057469 | null | 60 |
| calibration/C05/cap/0.980/A-F | 0.98 | true | 10000 | 219 | 317 | true | 0.0219 | 0.0191214, 0.0249617 | null | 60 |
| calibration/C05/cap/0.980/P-F | 0.98 | true | 10000 | 236 | 317 | true | 0.0236 | 0.0207142, 0.0267676 | null | 60 |
| calibration/C05/cap/0.980/joint | 0.98 | true | 10000 | 446 | 593 | true | 0.0446 | 0.0406373, 0.0488305 | null | 60 |
| calibration/C05/cap/0.985/A-F | 0.985 | true | 10000 | 166 | 317 | true | 0.0166 | 0.0141874, 0.0192998 | null | 60 |
| calibration/C05/cap/0.985/P-F | 0.985 | true | 10000 | 191 | 317 | true | 0.0191 | 0.016508, 0.0219771 | null | 60 |
| calibration/C05/cap/0.985/joint | 0.985 | true | 10000 | 354 | 593 | true | 0.0354 | 0.0318641, 0.0392097 | null | 60 |
| calibration/C05/cap/0.990/A-F | 0.99 | true | 10000 | 127 | 317 | true | 0.0127 | 0.0105981, 0.0150923 | null | 60 |
| calibration/C05/cap/0.990/P-F | 0.99 | true | 10000 | 134 | 317 | true | 0.0134 | 0.011239, 0.0158508 | null | 60 |
| calibration/C05/cap/0.990/joint | 0.99 | true | 10000 | 261 | 593 | true | 0.0261 | 0.0230638, 0.0294162 | null | 60 |
| calibration/C06/36/0.975/A-F | 0.975 | true | 10000 | 302 | 317 | true | 0.0302 | 0.0269332, 0.0337441 | null | null |
| calibration/C06/36/0.975/P-F | 0.975 | true | 10000 | 247 | 317 | true | 0.0247 | 0.021747, 0.027934 | null | null |
| calibration/C06/36/0.975/joint | 0.975 | true | 10000 | 545 | 593 | true | 0.0545 | 0.0501311, 0.0591305 | null | null |
| calibration/C06/36/0.980/A-F | 0.98 | true | 10000 | 248 | 317 | true | 0.0248 | 0.021841, 0.02804 | null | null |
| calibration/C06/36/0.980/P-F | 0.98 | true | 10000 | 199 | 317 | true | 0.0199 | 0.0172533, 0.0228312 | null | null |
| calibration/C06/36/0.980/joint | 0.98 | true | 10000 | 446 | 593 | true | 0.0446 | 0.0406373, 0.0488305 | null | null |
| calibration/C06/36/0.985/A-F | 0.985 | true | 10000 | 188 | 317 | true | 0.0188 | 0.0162289, 0.0216565 | null | null |
| calibration/C06/36/0.985/P-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| calibration/C06/36/0.985/joint | 0.985 | true | 10000 | 339 | 593 | true | 0.0339 | 0.0304393, 0.0376355 | null | null |
| calibration/C06/36/0.990/A-F | 0.99 | true | 10000 | 120 | 317 | true | 0.012 | 0.00995893, 0.0143322 | null | null |
| calibration/C06/36/0.990/P-F | 0.99 | true | 10000 | 97 | 317 | true | 0.0097 | 0.00787289, 0.0118205 | null | null |
| calibration/C06/36/0.990/joint | 0.99 | true | 10000 | 216 | 593 | true | 0.0216 | 0.0188407, 0.0246425 | null | null |
| calibration/C06/45/0.975/A-F | 0.975 | true | 10000 | 286 | 317 | true | 0.0286 | 0.025421, 0.0320573 | null | null |
| calibration/C06/45/0.975/P-F | 0.975 | true | 10000 | 276 | 317 | true | 0.0276 | 0.0244773, 0.0310017 | null | null |
| calibration/C06/45/0.975/joint | 0.975 | true | 10000 | 551 | 593 | true | 0.0551 | 0.0507079, 0.0597533 | null | null |
| calibration/C06/45/0.980/A-F | 0.98 | true | 10000 | 234 | 317 | true | 0.0234 | 0.0205266, 0.0265554 | null | null |
| calibration/C06/45/0.980/P-F | 0.98 | true | 10000 | 229 | 317 | true | 0.0229 | 0.0200578, 0.0260245 | null | null |
| calibration/C06/45/0.980/joint | 0.98 | true | 10000 | 455 | 593 | true | 0.0455 | 0.0414984, 0.0497689 | null | null |
| calibration/C06/45/0.985/A-F | 0.985 | true | 10000 | 188 | 317 | true | 0.0188 | 0.0162289, 0.0216565 | null | null |
| calibration/C06/45/0.985/P-F | 0.985 | true | 10000 | 175 | 317 | true | 0.0175 | 0.0150213, 0.0202651 | null | null |
| calibration/C06/45/0.985/joint | 0.985 | true | 10000 | 358 | 593 | true | 0.0358 | 0.0322443, 0.0396292 | null | null |
| calibration/C06/45/0.990/A-F | 0.99 | true | 10000 | 119 | 317 | true | 0.0119 | 0.00986777, 0.0142234 | null | null |
| calibration/C06/45/0.990/P-F | 0.99 | true | 10000 | 124 | 317 | true | 0.0124 | 0.010324, 0.0147668 | null | null |
| calibration/C06/45/0.990/joint | 0.99 | true | 10000 | 240 | 593 | true | 0.024 | 0.0210896, 0.027192 | null | null |
| calibration/C06/60/0.975/A-F | 0.975 | true | 10000 | 276 | 317 | true | 0.0276 | 0.0244773, 0.0310017 | null | null |
| calibration/C06/60/0.975/P-F | 0.975 | true | 10000 | 255 | 317 | true | 0.0255 | 0.0224992, 0.0287813 | null | null |
| calibration/C06/60/0.975/joint | 0.975 | true | 10000 | 516 | 593 | true | 0.0516 | 0.0473453, 0.0561181 | null | null |
| calibration/C06/60/0.980/A-F | 0.98 | true | 10000 | 218 | 317 | true | 0.0218 | 0.0190278, 0.0248553 | null | null |
| calibration/C06/60/0.980/P-F | 0.98 | true | 10000 | 214 | 317 | true | 0.0214 | 0.0186537, 0.0244297 | null | null |
| calibration/C06/60/0.980/joint | 0.98 | true | 10000 | 425 | 593 | true | 0.0425 | 0.03863, 0.0466392 | null | null |
| calibration/C06/60/0.985/A-F | 0.985 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | null |
| calibration/C06/60/0.985/P-F | 0.985 | true | 10000 | 163 | 317 | true | 0.0163 | 0.0139098, 0.0189775 | null | null |
| calibration/C06/60/0.985/joint | 0.985 | true | 10000 | 317 | 593 | true | 0.0317 | 0.0283531, 0.0353231 | null | null |
| calibration/C06/60/0.990/A-F | 0.99 | true | 10000 | 112 | 317 | true | 0.0112 | 0.00923074, 0.0134611 | null | null |
| calibration/C06/60/0.990/P-F | 0.99 | true | 10000 | 114 | 317 | true | 0.0114 | 0.00941254, 0.0136791 | null | null |
| calibration/C06/60/0.990/joint | 0.99 | true | 10000 | 225 | 593 | true | 0.0225 | 0.019683, 0.0255996 | null | null |
| calibration/C06/75/0.975/A-F | 0.975 | true | 10000 | 267 | 317 | true | 0.0267 | 0.0236289, 0.0300508 | null | null |
| calibration/C06/75/0.975/P-F | 0.975 | true | 10000 | 229 | 317 | true | 0.0229 | 0.0200578, 0.0260245 | null | null |
| calibration/C06/75/0.975/joint | 0.975 | true | 10000 | 493 | 593 | true | 0.0493 | 0.0451386, 0.0537263 | null | null |
| calibration/C06/75/0.980/A-F | 0.98 | true | 10000 | 212 | 317 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| calibration/C06/75/0.980/P-F | 0.98 | true | 10000 | 187 | 317 | true | 0.0187 | 0.0161359, 0.0215496 | null | null |
| calibration/C06/75/0.980/joint | 0.98 | true | 10000 | 397 | 593 | true | 0.0397 | 0.0359577, 0.0437133 | null | null |
| calibration/C06/75/0.985/A-F | 0.985 | true | 10000 | 161 | 317 | true | 0.0161 | 0.0137249, 0.0187626 | null | null |
| calibration/C06/75/0.985/P-F | 0.985 | true | 10000 | 148 | 317 | true | 0.0148 | 0.0125254, 0.0173632 | null | null |
| calibration/C06/75/0.985/joint | 0.985 | true | 10000 | 308 | 593 | true | 0.0308 | 0.0275009, 0.034376 | null | null |
| calibration/C06/75/0.990/A-F | 0.99 | true | 10000 | 104 | 317 | true | 0.0104 | 0.00850524, 0.0125874 | null | null |
| calibration/C06/75/0.990/P-F | 0.99 | true | 10000 | 98 | 317 | true | 0.0098 | 0.00796307, 0.0119302 | null | null |
| calibration/C06/75/0.990/joint | 0.99 | true | 10000 | 202 | 593 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| calibration/C06/90/0.975/A-F | 0.975 | true | 10000 | 274 | 317 | true | 0.0274 | 0.0242887, 0.0307905 | null | null |
| calibration/C06/90/0.975/P-F | 0.975 | true | 10000 | 265 | 317 | true | 0.0265 | 0.0234405, 0.0298393 | null | null |
| calibration/C06/90/0.975/joint | 0.975 | true | 10000 | 535 | 593 | true | 0.0535 | 0.0491701, 0.0580922 | null | null |
| calibration/C06/90/0.980/A-F | 0.98 | true | 10000 | 225 | 317 | true | 0.0225 | 0.019683, 0.0255996 | null | null |
| calibration/C06/90/0.980/P-F | 0.98 | true | 10000 | 218 | 317 | true | 0.0218 | 0.0190278, 0.0248553 | null | null |
| calibration/C06/90/0.980/joint | 0.98 | true | 10000 | 440 | 593 | true | 0.044 | 0.0400635, 0.0482047 | null | null |
| calibration/C06/90/0.985/A-F | 0.985 | true | 10000 | 166 | 317 | true | 0.0166 | 0.0141874, 0.0192998 | null | null |
| calibration/C06/90/0.985/P-F | 0.985 | true | 10000 | 175 | 317 | true | 0.0175 | 0.0150213, 0.0202651 | null | null |
| calibration/C06/90/0.985/joint | 0.985 | true | 10000 | 340 | 593 | true | 0.034 | 0.0305342, 0.0377405 | null | null |
| calibration/C06/90/0.990/A-F | 0.99 | true | 10000 | 113 | 317 | true | 0.0113 | 0.00932162, 0.0135702 | null | null |
| calibration/C06/90/0.990/P-F | 0.99 | true | 10000 | 112 | 317 | true | 0.0112 | 0.00923074, 0.0134611 | null | null |
| calibration/C06/90/0.990/joint | 0.99 | true | 10000 | 224 | 593 | true | 0.0224 | 0.0195894, 0.0254933 | null | null |
| calibration/C06/99/0.975/A-F | 0.975 | true | 10000 | 251 | 317 | true | 0.0251 | 0.022123, 0.0283578 | null | null |
| calibration/C06/99/0.975/P-F | 0.975 | true | 10000 | 259 | 317 | true | 0.0259 | 0.0228755, 0.0292047 | null | null |
| calibration/C06/99/0.975/joint | 0.975 | true | 10000 | 505 | 593 | true | 0.0505 | 0.0462896, 0.0549745 | null | null |
| calibration/C06/99/0.980/A-F | 0.98 | true | 10000 | 211 | 317 | true | 0.0211 | 0.0183733, 0.0241103 | null | null |
| calibration/C06/99/0.980/P-F | 0.98 | true | 10000 | 213 | 317 | true | 0.0213 | 0.0185602, 0.0243233 | null | null |
| calibration/C06/99/0.980/joint | 0.98 | true | 10000 | 420 | 593 | true | 0.042 | 0.0381524, 0.0461171 | null | null |
| calibration/C06/99/0.985/A-F | 0.985 | true | 10000 | 159 | 317 | true | 0.0159 | 0.0135401, 0.0185476 | null | null |
| calibration/C06/99/0.985/P-F | 0.985 | true | 10000 | 162 | 317 | true | 0.0162 | 0.0138173, 0.0188701 | null | null |
| calibration/C06/99/0.985/joint | 0.985 | true | 10000 | 317 | 593 | true | 0.0317 | 0.0283531, 0.0353231 | null | null |
| calibration/C06/99/0.990/A-F | 0.99 | true | 10000 | 113 | 317 | true | 0.0113 | 0.00932162, 0.0135702 | null | null |
| calibration/C06/99/0.990/P-F | 0.99 | true | 10000 | 103 | 317 | true | 0.0103 | 0.00841475, 0.012478 | null | null |
| calibration/C06/99/0.990/joint | 0.99 | true | 10000 | 214 | 593 | true | 0.0214 | 0.0186537, 0.0244297 | null | null |
| calibration/C06/cap/0.975/A-F | 0.975 | true | 10000 | 260 | 317 | true | 0.026 | 0.0229697, 0.0293105 | null | 60 |
| calibration/C06/cap/0.975/P-F | 0.975 | true | 10000 | 245 | 317 | true | 0.0245 | 0.0215591, 0.0277221 | null | 60 |
| calibration/C06/cap/0.975/joint | 0.975 | true | 10000 | 501 | 593 | true | 0.0501 | 0.0459059, 0.0545585 | null | 60 |
| calibration/C06/cap/0.980/A-F | 0.98 | true | 10000 | 224 | 317 | true | 0.0224 | 0.0195894, 0.0254933 | null | 60 |
| calibration/C06/cap/0.980/P-F | 0.98 | true | 10000 | 204 | 317 | true | 0.0204 | 0.0177197, 0.0233645 | null | 60 |
| calibration/C06/cap/0.980/joint | 0.98 | true | 10000 | 423 | 593 | true | 0.0423 | 0.038439, 0.0464304 | null | 60 |
| calibration/C06/cap/0.985/A-F | 0.985 | true | 10000 | 168 | 317 | true | 0.0168 | 0.0143725, 0.0195144 | null | 60 |
| calibration/C06/cap/0.985/P-F | 0.985 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | 60 |
| calibration/C06/cap/0.985/joint | 0.985 | true | 10000 | 324 | 593 | true | 0.0324 | 0.0290164, 0.0360593 | null | 60 |
| calibration/C06/cap/0.990/A-F | 0.99 | true | 10000 | 110 | 317 | true | 0.011 | 0.0090491, 0.013243 | null | 60 |
| calibration/C06/cap/0.990/P-F | 0.99 | true | 10000 | 110 | 317 | true | 0.011 | 0.0090491, 0.013243 | null | 60 |
| calibration/C06/cap/0.990/joint | 0.99 | true | 10000 | 219 | 593 | true | 0.0219 | 0.0191214, 0.0249617 | null | 60 |
| calibration/C07/36/0.975/A-F | 0.975 | true | 10000 | 355 | 317 | false | 0.0355 | 0.0319591, 0.0393146 | null | null |
| calibration/C07/36/0.975/P-F | 0.975 | true | 10000 | 333 | 317 | false | 0.0333 | 0.0298699, 0.0370052 | null | null |
| calibration/C07/36/0.975/joint | 0.975 | true | 10000 | 670 | 593 | false | 0.067 | 0.0621763, 0.0720777 | null | null |
| calibration/C07/36/0.980/A-F | 0.98 | true | 10000 | 305 | 317 | true | 0.0305 | 0.027217, 0.0340601 | null | null |
| calibration/C07/36/0.980/P-F | 0.98 | true | 10000 | 300 | 317 | true | 0.03 | 0.026744, 0.0335334 | null | null |
| calibration/C07/36/0.980/joint | 0.98 | true | 10000 | 589 | 593 | true | 0.0589 | 0.0543645, 0.0636944 | null | null |
| calibration/C07/36/0.985/A-F | 0.985 | true | 10000 | 265 | 317 | true | 0.0265 | 0.0234405, 0.0298393 | null | null |
| calibration/C07/36/0.985/P-F | 0.985 | true | 10000 | 267 | 317 | true | 0.0267 | 0.0236289, 0.0300508 | null | null |
| calibration/C07/36/0.985/joint | 0.985 | true | 10000 | 521 | 593 | true | 0.0521 | 0.0478254, 0.0566378 | null | null |
| calibration/C07/36/0.990/A-F | 0.99 | true | 10000 | 221 | 317 | true | 0.0221 | 0.0193085, 0.0251744 | null | null |
| calibration/C07/36/0.990/P-F | 0.99 | true | 10000 | 228 | 317 | true | 0.0228 | 0.0199641, 0.0259183 | null | null |
| calibration/C07/36/0.990/joint | 0.99 | true | 10000 | 441 | 593 | true | 0.0441 | 0.0401592, 0.048309 | null | null |
| calibration/C07/45/0.975/A-F | 0.975 | true | 10000 | 304 | 317 | true | 0.0304 | 0.0271224, 0.0339547 | null | null |
| calibration/C07/45/0.975/P-F | 0.975 | true | 10000 | 298 | 317 | true | 0.0298 | 0.0265549, 0.0333226 | null | null |
| calibration/C07/45/0.975/joint | 0.975 | true | 10000 | 593 | 593 | true | 0.0593 | 0.0547497, 0.064109 | null | null |
| calibration/C07/45/0.980/A-F | 0.98 | true | 10000 | 263 | 317 | true | 0.0263 | 0.0232521, 0.0296278 | null | null |
| calibration/C07/45/0.980/P-F | 0.98 | true | 10000 | 252 | 317 | true | 0.0252 | 0.022217, 0.0284637 | null | null |
| calibration/C07/45/0.980/joint | 0.98 | true | 10000 | 510 | 593 | true | 0.051 | 0.0467694, 0.0554944 | null | null |
| calibration/C07/45/0.985/A-F | 0.985 | true | 10000 | 207 | 317 | true | 0.0207 | 0.0179997, 0.0236842 | null | null |
| calibration/C07/45/0.985/P-F | 0.985 | true | 10000 | 214 | 317 | true | 0.0214 | 0.0186537, 0.0244297 | null | null |
| calibration/C07/45/0.985/joint | 0.985 | true | 10000 | 417 | 593 | true | 0.0417 | 0.037866, 0.0458037 | null | null |
| calibration/C07/45/0.990/A-F | 0.99 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | null |
| calibration/C07/45/0.990/P-F | 0.99 | true | 10000 | 168 | 317 | true | 0.0168 | 0.0143725, 0.0195144 | null | null |
| calibration/C07/45/0.990/joint | 0.99 | true | 10000 | 316 | 593 | true | 0.0316 | 0.0282584, 0.0352179 | null | null |
| calibration/C07/60/0.975/A-F | 0.975 | true | 10000 | 268 | 317 | true | 0.0268 | 0.0237231, 0.0301565 | null | null |
| calibration/C07/60/0.975/P-F | 0.975 | true | 10000 | 253 | 317 | true | 0.0253 | 0.022311, 0.0285696 | null | null |
| calibration/C07/60/0.975/joint | 0.975 | true | 10000 | 518 | 593 | true | 0.0518 | 0.0475373, 0.056326 | null | null |
| calibration/C07/60/0.980/A-F | 0.98 | true | 10000 | 220 | 317 | true | 0.022 | 0.0192149, 0.025068 | null | null |
| calibration/C07/60/0.980/P-F | 0.98 | true | 10000 | 211 | 317 | true | 0.0211 | 0.0183733, 0.0241103 | null | null |
| calibration/C07/60/0.980/joint | 0.98 | true | 10000 | 431 | 593 | true | 0.0431 | 0.0392033, 0.0472656 | null | null |
| calibration/C07/60/0.985/A-F | 0.985 | true | 10000 | 176 | 317 | true | 0.0176 | 0.015114, 0.0203723 | null | null |
| calibration/C07/60/0.985/P-F | 0.985 | true | 10000 | 169 | 317 | true | 0.0169 | 0.0144651, 0.0196217 | null | null |
| calibration/C07/60/0.985/joint | 0.985 | true | 10000 | 345 | 593 | true | 0.0345 | 0.031009, 0.0382654 | null | null |
| calibration/C07/60/0.990/A-F | 0.99 | true | 10000 | 128 | 317 | true | 0.0128 | 0.0106896, 0.0152008 | null | null |
| calibration/C07/60/0.990/P-F | 0.99 | true | 10000 | 121 | 317 | true | 0.0121 | 0.0100501, 0.0144409 | null | null |
| calibration/C07/60/0.990/joint | 0.99 | true | 10000 | 249 | 593 | true | 0.0249 | 0.021935, 0.0281459 | null | null |
| calibration/C07/75/0.975/A-F | 0.975 | true | 10000 | 244 | 317 | true | 0.0244 | 0.0214652, 0.0276161 | null | null |
| calibration/C07/75/0.975/P-F | 0.975 | true | 10000 | 251 | 317 | true | 0.0251 | 0.022123, 0.0283578 | null | null |
| calibration/C07/75/0.975/joint | 0.975 | true | 10000 | 490 | 593 | true | 0.049 | 0.0448509, 0.0534142 | null | null |
| calibration/C07/75/0.980/A-F | 0.98 | true | 10000 | 197 | 317 | true | 0.0197 | 0.0170669, 0.0226178 | null | null |
| calibration/C07/75/0.980/P-F | 0.98 | true | 10000 | 215 | 317 | true | 0.0215 | 0.0187472, 0.0245361 | null | null |
| calibration/C07/75/0.980/joint | 0.98 | true | 10000 | 409 | 593 | true | 0.0409 | 0.0371024, 0.0449679 | null | null |
| calibration/C07/75/0.985/A-F | 0.985 | true | 10000 | 168 | 317 | true | 0.0168 | 0.0143725, 0.0195144 | null | null |
| calibration/C07/75/0.985/P-F | 0.985 | true | 10000 | 164 | 317 | true | 0.0164 | 0.0140023, 0.019085 | null | null |
| calibration/C07/75/0.985/joint | 0.985 | true | 10000 | 329 | 593 | true | 0.0329 | 0.0294905, 0.0365849 | null | null |
| calibration/C07/75/0.990/A-F | 0.99 | true | 10000 | 126 | 317 | true | 0.0126 | 0.0105067, 0.0149838 | null | null |
| calibration/C07/75/0.990/P-F | 0.99 | true | 10000 | 114 | 317 | true | 0.0114 | 0.00941254, 0.0136791 | null | null |
| calibration/C07/75/0.990/joint | 0.99 | true | 10000 | 239 | 593 | true | 0.0239 | 0.0209957, 0.0270859 | null | null |
| calibration/C07/90/0.975/A-F | 0.975 | true | 10000 | 229 | 317 | true | 0.0229 | 0.0200578, 0.0260245 | null | null |
| calibration/C07/90/0.975/P-F | 0.975 | true | 10000 | 235 | 317 | true | 0.0235 | 0.0206204, 0.0266615 | null | null |
| calibration/C07/90/0.975/joint | 0.975 | true | 10000 | 459 | 593 | true | 0.0459 | 0.0418812, 0.0501858 | null | null |
| calibration/C07/90/0.980/A-F | 0.98 | true | 10000 | 192 | 317 | true | 0.0192 | 0.0166011, 0.0220839 | null | null |
| calibration/C07/90/0.980/P-F | 0.98 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| calibration/C07/90/0.980/joint | 0.98 | true | 10000 | 379 | 593 | true | 0.0379 | 0.0342425, 0.0418297 | null | null |
| calibration/C07/90/0.985/A-F | 0.985 | true | 10000 | 142 | 317 | true | 0.0142 | 0.0119734, 0.0167157 | null | null |
| calibration/C07/90/0.985/P-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| calibration/C07/90/0.985/joint | 0.985 | true | 10000 | 292 | 593 | true | 0.0292 | 0.0259878, 0.0326901 | null | null |
| calibration/C07/90/0.990/A-F | 0.99 | true | 10000 | 105 | 317 | true | 0.0105 | 0.00859577, 0.0126968 | null | null |
| calibration/C07/90/0.990/P-F | 0.99 | true | 10000 | 99 | 317 | true | 0.0099 | 0.00805331, 0.0120399 | null | null |
| calibration/C07/90/0.990/joint | 0.99 | true | 10000 | 204 | 593 | true | 0.0204 | 0.0177197, 0.0233645 | null | null |
| calibration/C07/99/0.975/A-F | 0.975 | true | 10000 | 246 | 317 | true | 0.0246 | 0.021653, 0.0278281 | null | null |
| calibration/C07/99/0.975/P-F | 0.975 | true | 10000 | 243 | 317 | true | 0.0243 | 0.0213712, 0.0275101 | null | null |
| calibration/C07/99/0.975/joint | 0.975 | true | 10000 | 486 | 593 | true | 0.0486 | 0.0444675, 0.0529979 | null | null |
| calibration/C07/99/0.980/A-F | 0.98 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| calibration/C07/99/0.980/P-F | 0.98 | true | 10000 | 189 | 317 | true | 0.0189 | 0.0163219, 0.0217634 | null | null |
| calibration/C07/99/0.980/joint | 0.98 | true | 10000 | 388 | 593 | true | 0.0388 | 0.0350998, 0.0427718 | null | null |
| calibration/C07/99/0.985/A-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| calibration/C07/99/0.985/P-F | 0.985 | true | 10000 | 136 | 317 | true | 0.0136 | 0.0114224, 0.0160672 | null | null |
| calibration/C07/99/0.985/joint | 0.985 | true | 10000 | 294 | 593 | true | 0.0294 | 0.0261768, 0.032901 | null | null |
| calibration/C07/99/0.990/A-F | 0.99 | true | 10000 | 101 | 317 | true | 0.0101 | 0.00823393, 0.0122591 | null | null |
| calibration/C07/99/0.990/P-F | 0.99 | true | 10000 | 93 | 317 | true | 0.0093 | 0.00751268, 0.0113812 | null | null |
| calibration/C07/99/0.990/joint | 0.99 | true | 10000 | 194 | 593 | true | 0.0194 | 0.0167874, 0.0222975 | null | null |
| calibration/C07/cap/0.975/A-F | 0.975 | true | 10000 | 260 | 317 | true | 0.026 | 0.0229697, 0.0293105 | null | 60 |
| calibration/C07/cap/0.975/P-F | 0.975 | true | 10000 | 243 | 317 | true | 0.0243 | 0.0213712, 0.0275101 | null | 60 |
| calibration/C07/cap/0.975/joint | 0.975 | true | 10000 | 500 | 593 | true | 0.05 | 0.0458099, 0.0544545 | null | 60 |
| calibration/C07/cap/0.980/A-F | 0.98 | true | 10000 | 229 | 317 | true | 0.0229 | 0.0200578, 0.0260245 | null | 60 |
| calibration/C07/cap/0.980/P-F | 0.98 | true | 10000 | 206 | 317 | true | 0.0206 | 0.0179063, 0.0235777 | null | 60 |
| calibration/C07/cap/0.980/joint | 0.98 | true | 10000 | 433 | 593 | true | 0.0433 | 0.0393944, 0.0474743 | null | 60 |
| calibration/C07/cap/0.985/A-F | 0.985 | true | 10000 | 174 | 317 | true | 0.0174 | 0.0149285, 0.020158 | null | 60 |
| calibration/C07/cap/0.985/P-F | 0.985 | true | 10000 | 161 | 317 | true | 0.0161 | 0.0137249, 0.0187626 | null | 60 |
| calibration/C07/cap/0.985/joint | 0.985 | true | 10000 | 334 | 593 | true | 0.0334 | 0.0299648, 0.0371103 | null | 60 |
| calibration/C07/cap/0.990/A-F | 0.99 | true | 10000 | 131 | 317 | true | 0.0131 | 0.0109641, 0.0155259 | null | 60 |
| calibration/C07/cap/0.990/P-F | 0.99 | true | 10000 | 118 | 317 | true | 0.0118 | 0.00977665, 0.0141146 | null | 60 |
| calibration/C07/cap/0.990/joint | 0.99 | true | 10000 | 251 | 593 | true | 0.0251 | 0.022123, 0.0283578 | null | 60 |
| calibration/C08/36/0.975/A-F | 0.975 | true | 10000 | 335 | 317 | false | 0.0335 | 0.0300597, 0.0372153 | null | null |
| calibration/C08/36/0.975/P-F | 0.975 | true | 10000 | 304 | 317 | true | 0.0304 | 0.0271224, 0.0339547 | null | null |
| calibration/C08/36/0.975/joint | 0.975 | true | 10000 | 626 | 593 | false | 0.0626 | 0.0579301, 0.0675266 | null | null |
| calibration/C08/36/0.980/A-F | 0.98 | true | 10000 | 310 | 317 | true | 0.031 | 0.0276902, 0.0345865 | null | null |
| calibration/C08/36/0.980/P-F | 0.98 | true | 10000 | 282 | 317 | true | 0.0282 | 0.0250434, 0.0316352 | null | null |
| calibration/C08/36/0.980/joint | 0.98 | true | 10000 | 579 | 593 | true | 0.0579 | 0.0534017, 0.0626579 | null | null |
| calibration/C08/36/0.985/A-F | 0.985 | true | 10000 | 287 | 317 | true | 0.0287 | 0.0255155, 0.0321628 | null | null |
| calibration/C08/36/0.985/P-F | 0.985 | true | 10000 | 264 | 317 | true | 0.0264 | 0.0233463, 0.0297336 | null | null |
| calibration/C08/36/0.985/joint | 0.985 | true | 10000 | 539 | 593 | true | 0.0539 | 0.0495545, 0.0585076 | null | null |
| calibration/C08/36/0.990/A-F | 0.99 | true | 10000 | 272 | 317 | true | 0.0272 | 0.0241001, 0.0305792 | null | null |
| calibration/C08/36/0.990/P-F | 0.99 | true | 10000 | 236 | 317 | true | 0.0236 | 0.0207142, 0.0267676 | null | null |
| calibration/C08/36/0.990/joint | 0.99 | true | 10000 | 498 | 593 | true | 0.0498 | 0.0456181, 0.0542465 | null | null |
| calibration/C08/45/0.975/A-F | 0.975 | true | 10000 | 293 | 317 | true | 0.0293 | 0.0260823, 0.0327956 | null | null |
| calibration/C08/45/0.975/P-F | 0.975 | true | 10000 | 267 | 317 | true | 0.0267 | 0.0236289, 0.0300508 | null | null |
| calibration/C08/45/0.975/joint | 0.975 | true | 10000 | 554 | 593 | true | 0.0554 | 0.0509964, 0.0600647 | null | null |
| calibration/C08/45/0.980/A-F | 0.98 | true | 10000 | 243 | 317 | true | 0.0243 | 0.0213712, 0.0275101 | null | null |
| calibration/C08/45/0.980/P-F | 0.98 | true | 10000 | 241 | 317 | true | 0.0241 | 0.0211834, 0.027298 | null | null |
| calibration/C08/45/0.980/joint | 0.98 | true | 10000 | 477 | 593 | true | 0.0477 | 0.043605, 0.0520609 | null | null |
| calibration/C08/45/0.985/A-F | 0.985 | true | 10000 | 212 | 317 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| calibration/C08/45/0.985/P-F | 0.985 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| calibration/C08/45/0.985/joint | 0.985 | true | 10000 | 409 | 593 | true | 0.0409 | 0.0371024, 0.0449679 | null | null |
| calibration/C08/45/0.990/A-F | 0.99 | true | 10000 | 176 | 317 | true | 0.0176 | 0.015114, 0.0203723 | null | null |
| calibration/C08/45/0.990/P-F | 0.99 | true | 10000 | 154 | 317 | true | 0.0154 | 0.0130785, 0.0180096 | null | null |
| calibration/C08/45/0.990/joint | 0.99 | true | 10000 | 327 | 593 | true | 0.0327 | 0.0293009, 0.0363747 | null | null |
| calibration/C08/60/0.975/A-F | 0.975 | true | 10000 | 217 | 317 | true | 0.0217 | 0.0189343, 0.0247489 | null | null |
| calibration/C08/60/0.975/P-F | 0.975 | true | 10000 | 217 | 317 | true | 0.0217 | 0.0189343, 0.0247489 | null | null |
| calibration/C08/60/0.975/joint | 0.975 | true | 10000 | 432 | 593 | true | 0.0432 | 0.0392988, 0.0473699 | null | null |
| calibration/C08/60/0.980/A-F | 0.98 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| calibration/C08/60/0.980/P-F | 0.98 | true | 10000 | 184 | 317 | true | 0.0184 | 0.0158569, 0.0212288 | null | null |
| calibration/C08/60/0.980/joint | 0.98 | true | 10000 | 373 | 593 | true | 0.0373 | 0.0336712, 0.0412013 | null | null |
| calibration/C08/60/0.985/A-F | 0.985 | true | 10000 | 162 | 317 | true | 0.0162 | 0.0138173, 0.0188701 | null | null |
| calibration/C08/60/0.985/P-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| calibration/C08/60/0.985/joint | 0.985 | true | 10000 | 322 | 593 | true | 0.0322 | 0.0288269, 0.035849 | null | null |
| calibration/C08/60/0.990/A-F | 0.99 | true | 10000 | 135 | 317 | true | 0.0135 | 0.0113307, 0.015959 | null | null |
| calibration/C08/60/0.990/P-F | 0.99 | true | 10000 | 124 | 317 | true | 0.0124 | 0.010324, 0.0147668 | null | null |
| calibration/C08/60/0.990/joint | 0.99 | true | 10000 | 259 | 593 | true | 0.0259 | 0.0228755, 0.0292047 | null | null |
| calibration/C08/75/0.975/A-F | 0.975 | true | 10000 | 199 | 317 | true | 0.0199 | 0.0172533, 0.0228312 | null | null |
| calibration/C08/75/0.975/P-F | 0.975 | true | 10000 | 170 | 317 | true | 0.017 | 0.0145578, 0.019729 | null | null |
| calibration/C08/75/0.975/joint | 0.975 | true | 10000 | 366 | 593 | true | 0.0366 | 0.0330051, 0.0404679 | null | null |
| calibration/C08/75/0.980/A-F | 0.98 | true | 10000 | 156 | 317 | true | 0.0156 | 0.013263, 0.0182249 | null | null |
| calibration/C08/75/0.980/P-F | 0.98 | true | 10000 | 142 | 317 | true | 0.0142 | 0.0119734, 0.0167157 | null | null |
| calibration/C08/75/0.980/joint | 0.98 | true | 10000 | 295 | 593 | true | 0.0295 | 0.0262713, 0.0330064 | null | null |
| calibration/C08/75/0.985/A-F | 0.985 | true | 10000 | 122 | 317 | true | 0.0122 | 0.0101414, 0.0145495 | null | null |
| calibration/C08/75/0.985/P-F | 0.985 | true | 10000 | 107 | 317 | true | 0.0107 | 0.00877696, 0.0129154 | null | null |
| calibration/C08/75/0.985/joint | 0.985 | true | 10000 | 228 | 593 | true | 0.0228 | 0.0199641, 0.0259183 | null | null |
| calibration/C08/75/0.990/A-F | 0.99 | true | 10000 | 82 | 317 | true | 0.0082 | 0.00652686, 0.0101683 | null | null |
| calibration/C08/75/0.990/P-F | 0.99 | true | 10000 | 79 | 317 | true | 0.0079 | 0.00625935, 0.00983617 | null | null |
| calibration/C08/75/0.990/joint | 0.99 | true | 10000 | 160 | 593 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| calibration/C08/90/0.975/A-F | 0.975 | true | 10000 | 169 | 317 | true | 0.0169 | 0.0144651, 0.0196217 | null | null |
| calibration/C08/90/0.975/P-F | 0.975 | true | 10000 | 183 | 317 | true | 0.0183 | 0.015764, 0.0211218 | null | null |
| calibration/C08/90/0.975/joint | 0.975 | true | 10000 | 348 | 593 | true | 0.0348 | 0.031294, 0.0385802 | null | null |
| calibration/C08/90/0.980/A-F | 0.98 | true | 10000 | 141 | 317 | true | 0.0141 | 0.0118815, 0.0166077 | null | null |
| calibration/C08/90/0.980/P-F | 0.98 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | null |
| calibration/C08/90/0.980/joint | 0.98 | true | 10000 | 286 | 593 | true | 0.0286 | 0.025421, 0.0320573 | null | null |
| calibration/C08/90/0.985/A-F | 0.985 | true | 10000 | 106 | 317 | true | 0.0106 | 0.00868634, 0.0128061 | null | null |
| calibration/C08/90/0.985/P-F | 0.985 | true | 10000 | 110 | 317 | true | 0.011 | 0.0090491, 0.013243 | null | null |
| calibration/C08/90/0.985/joint | 0.985 | true | 10000 | 214 | 593 | true | 0.0214 | 0.0186537, 0.0244297 | null | null |
| calibration/C08/90/0.990/A-F | 0.99 | true | 10000 | 76 | 317 | true | 0.0076 | 0.00599247, 0.00950343 | null | null |
| calibration/C08/90/0.990/P-F | 0.99 | true | 10000 | 70 | 317 | true | 0.007 | 0.00546079, 0.00883592 | null | null |
| calibration/C08/90/0.990/joint | 0.99 | true | 10000 | 145 | 593 | true | 0.0145 | 0.0122493, 0.0170396 | null | null |
| calibration/C08/99/0.975/A-F | 0.975 | true | 10000 | 169 | 317 | true | 0.0169 | 0.0144651, 0.0196217 | null | null |
| calibration/C08/99/0.975/P-F | 0.975 | true | 10000 | 177 | 317 | true | 0.0177 | 0.0152068, 0.0204794 | null | null |
| calibration/C08/99/0.975/joint | 0.975 | true | 10000 | 344 | 593 | true | 0.0344 | 0.030914, 0.0381604 | null | null |
| calibration/C08/99/0.980/A-F | 0.98 | true | 10000 | 129 | 317 | true | 0.0129 | 0.0107811, 0.0153092 | null | null |
| calibration/C08/99/0.980/P-F | 0.98 | true | 10000 | 136 | 317 | true | 0.0136 | 0.0114224, 0.0160672 | null | null |
| calibration/C08/99/0.980/joint | 0.98 | true | 10000 | 264 | 593 | true | 0.0264 | 0.0233463, 0.0297336 | null | null |
| calibration/C08/99/0.985/A-F | 0.985 | true | 10000 | 92 | 317 | true | 0.0092 | 0.00742276, 0.0112712 | null | null |
| calibration/C08/99/0.985/P-F | 0.985 | true | 10000 | 90 | 317 | true | 0.009 | 0.0072431, 0.0110511 | null | null |
| calibration/C08/99/0.985/joint | 0.985 | true | 10000 | 182 | 593 | true | 0.0182 | 0.0156711, 0.0210148 | null | null |
| calibration/C08/99/0.990/A-F | 0.99 | true | 10000 | 66 | 317 | true | 0.0066 | 0.00510801, 0.00838927 | null | null |
| calibration/C08/99/0.990/P-F | 0.99 | true | 10000 | 61 | 317 | true | 0.0061 | 0.00466914, 0.0078289 | null | null |
| calibration/C08/99/0.990/joint | 0.99 | true | 10000 | 127 | 593 | true | 0.0127 | 0.0105981, 0.0150923 | null | null |
| calibration/C08/cap/0.975/A-F | 0.975 | true | 10000 | 177 | 317 | true | 0.0177 | 0.0152068, 0.0204794 | null | 60 |
| calibration/C08/cap/0.975/P-F | 0.975 | true | 10000 | 179 | 317 | true | 0.0179 | 0.0153925, 0.0206936 | null | 60 |
| calibration/C08/cap/0.975/joint | 0.975 | true | 10000 | 354 | 593 | true | 0.0354 | 0.0318641, 0.0392097 | null | 60 |
| calibration/C08/cap/0.980/A-F | 0.98 | true | 10000 | 131 | 317 | true | 0.0131 | 0.0109641, 0.0155259 | null | 60 |
| calibration/C08/cap/0.980/P-F | 0.98 | true | 10000 | 149 | 317 | true | 0.0149 | 0.0126175, 0.017471 | null | 60 |
| calibration/C08/cap/0.980/joint | 0.98 | true | 10000 | 280 | 593 | true | 0.028 | 0.0248547, 0.0314241 | null | 60 |
| calibration/C08/cap/0.985/A-F | 0.985 | true | 10000 | 112 | 317 | true | 0.0112 | 0.00923074, 0.0134611 | null | 60 |
| calibration/C08/cap/0.985/P-F | 0.985 | true | 10000 | 116 | 317 | true | 0.0116 | 0.00959451, 0.013897 | null | 60 |
| calibration/C08/cap/0.985/joint | 0.985 | true | 10000 | 229 | 593 | true | 0.0229 | 0.0200578, 0.0260245 | null | 60 |
| calibration/C08/cap/0.990/A-F | 0.99 | true | 10000 | 81 | 317 | true | 0.0081 | 0.00643762, 0.0100576 | null | 60 |
| calibration/C08/cap/0.990/P-F | 0.99 | true | 10000 | 88 | 317 | true | 0.0088 | 0.00706367, 0.0108308 | null | 60 |
| calibration/C08/cap/0.990/joint | 0.99 | true | 10000 | 169 | 593 | true | 0.0169 | 0.0144651, 0.0196217 | null | 60 |
| calibration/C09/36/0.975/A-F | 0.975 | true | 10000 | 266 | 317 | true | 0.0266 | 0.0235347, 0.029945 | null | null |
| calibration/C09/36/0.975/P-F | 0.975 | true | 10000 | 274 | 317 | true | 0.0274 | 0.0242887, 0.0307905 | null | null |
| calibration/C09/36/0.975/joint | 0.975 | true | 10000 | 532 | 593 | true | 0.0532 | 0.0488819, 0.0577806 | null | null |
| calibration/C09/36/0.980/A-F | 0.98 | true | 10000 | 223 | 317 | true | 0.0223 | 0.0194957, 0.025387 | null | null |
| calibration/C09/36/0.980/P-F | 0.98 | true | 10000 | 217 | 317 | true | 0.0217 | 0.0189343, 0.0247489 | null | null |
| calibration/C09/36/0.980/joint | 0.98 | true | 10000 | 436 | 593 | true | 0.0436 | 0.0396811, 0.0477874 | null | null |
| calibration/C09/36/0.985/A-F | 0.985 | true | 10000 | 161 | 317 | true | 0.0161 | 0.0137249, 0.0187626 | null | null |
| calibration/C09/36/0.985/P-F | 0.985 | true | 10000 | 170 | 317 | true | 0.017 | 0.0145578, 0.019729 | null | null |
| calibration/C09/36/0.985/joint | 0.985 | true | 10000 | 330 | 593 | true | 0.033 | 0.0295854, 0.03669 | null | null |
| calibration/C09/36/0.990/A-F | 0.99 | true | 10000 | 116 | 317 | true | 0.0116 | 0.00959451, 0.013897 | null | null |
| calibration/C09/36/0.990/P-F | 0.99 | true | 10000 | 113 | 317 | true | 0.0113 | 0.00932162, 0.0135702 | null | null |
| calibration/C09/36/0.990/joint | 0.99 | true | 10000 | 229 | 593 | true | 0.0229 | 0.0200578, 0.0260245 | null | null |
| calibration/C09/45/0.975/A-F | 0.975 | true | 10000 | 262 | 317 | true | 0.0262 | 0.023158, 0.029522 | null | null |
| calibration/C09/45/0.975/P-F | 0.975 | true | 10000 | 258 | 317 | true | 0.0258 | 0.0227814, 0.0290988 | null | null |
| calibration/C09/45/0.975/joint | 0.975 | true | 10000 | 516 | 593 | true | 0.0516 | 0.0473453, 0.0561181 | null | null |
| calibration/C09/45/0.980/A-F | 0.98 | true | 10000 | 213 | 317 | true | 0.0213 | 0.0185602, 0.0243233 | null | null |
| calibration/C09/45/0.980/P-F | 0.98 | true | 10000 | 212 | 317 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| calibration/C09/45/0.980/joint | 0.98 | true | 10000 | 423 | 593 | true | 0.0423 | 0.038439, 0.0464304 | null | null |
| calibration/C09/45/0.985/A-F | 0.985 | true | 10000 | 159 | 317 | true | 0.0159 | 0.0135401, 0.0185476 | null | null |
| calibration/C09/45/0.985/P-F | 0.985 | true | 10000 | 161 | 317 | true | 0.0161 | 0.0137249, 0.0187626 | null | null |
| calibration/C09/45/0.985/joint | 0.985 | true | 10000 | 320 | 593 | true | 0.032 | 0.0286373, 0.0356387 | null | null |
| calibration/C09/45/0.990/A-F | 0.99 | true | 10000 | 110 | 317 | true | 0.011 | 0.0090491, 0.013243 | null | null |
| calibration/C09/45/0.990/P-F | 0.99 | true | 10000 | 108 | 317 | true | 0.0108 | 0.00886763, 0.0130246 | null | null |
| calibration/C09/45/0.990/joint | 0.99 | true | 10000 | 218 | 593 | true | 0.0218 | 0.0190278, 0.0248553 | null | null |
| calibration/C09/60/0.975/A-F | 0.975 | true | 10000 | 272 | 317 | true | 0.0272 | 0.0241001, 0.0305792 | null | null |
| calibration/C09/60/0.975/P-F | 0.975 | true | 10000 | 276 | 317 | true | 0.0276 | 0.0244773, 0.0310017 | null | null |
| calibration/C09/60/0.975/joint | 0.975 | true | 10000 | 545 | 593 | true | 0.0545 | 0.0501311, 0.0591305 | null | null |
| calibration/C09/60/0.980/A-F | 0.98 | true | 10000 | 215 | 317 | true | 0.0215 | 0.0187472, 0.0245361 | null | null |
| calibration/C09/60/0.980/P-F | 0.98 | true | 10000 | 216 | 317 | true | 0.0216 | 0.0188407, 0.0246425 | null | null |
| calibration/C09/60/0.980/joint | 0.98 | true | 10000 | 428 | 593 | true | 0.0428 | 0.0389166, 0.0469524 | null | null |
| calibration/C09/60/0.985/A-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| calibration/C09/60/0.985/P-F | 0.985 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | null |
| calibration/C09/60/0.985/joint | 0.985 | true | 10000 | 314 | 593 | true | 0.0314 | 0.0280689, 0.0350075 | null | null |
| calibration/C09/60/0.990/A-F | 0.99 | true | 10000 | 105 | 317 | true | 0.0105 | 0.00859577, 0.0126968 | null | null |
| calibration/C09/60/0.990/P-F | 0.99 | true | 10000 | 104 | 317 | true | 0.0104 | 0.00850524, 0.0125874 | null | null |
| calibration/C09/60/0.990/joint | 0.99 | true | 10000 | 208 | 593 | true | 0.0208 | 0.0180931, 0.0237908 | null | null |
| calibration/C09/75/0.975/A-F | 0.975 | true | 10000 | 274 | 317 | true | 0.0274 | 0.0242887, 0.0307905 | null | null |
| calibration/C09/75/0.975/P-F | 0.975 | true | 10000 | 225 | 317 | true | 0.0225 | 0.019683, 0.0255996 | null | null |
| calibration/C09/75/0.975/joint | 0.975 | true | 10000 | 490 | 593 | true | 0.049 | 0.0448509, 0.0534142 | null | null |
| calibration/C09/75/0.980/A-F | 0.98 | true | 10000 | 216 | 317 | true | 0.0216 | 0.0188407, 0.0246425 | null | null |
| calibration/C09/75/0.980/P-F | 0.98 | true | 10000 | 179 | 317 | true | 0.0179 | 0.0153925, 0.0206936 | null | null |
| calibration/C09/75/0.980/joint | 0.98 | true | 10000 | 388 | 593 | true | 0.0388 | 0.0350998, 0.0427718 | null | null |
| calibration/C09/75/0.985/A-F | 0.985 | true | 10000 | 169 | 317 | true | 0.0169 | 0.0144651, 0.0196217 | null | null |
| calibration/C09/75/0.985/P-F | 0.985 | true | 10000 | 140 | 317 | true | 0.014 | 0.0117896, 0.0164997 | null | null |
| calibration/C09/75/0.985/joint | 0.985 | true | 10000 | 307 | 593 | true | 0.0307 | 0.0274062, 0.0342707 | null | null |
| calibration/C09/75/0.990/A-F | 0.99 | true | 10000 | 109 | 317 | true | 0.0109 | 0.00895834, 0.0131338 | null | null |
| calibration/C09/75/0.990/P-F | 0.99 | true | 10000 | 87 | 317 | true | 0.0087 | 0.00697404, 0.0107205 | null | null |
| calibration/C09/75/0.990/joint | 0.99 | true | 10000 | 196 | 593 | true | 0.0196 | 0.0169737, 0.0225111 | null | null |
| calibration/C09/90/0.975/A-F | 0.975 | true | 10000 | 256 | 317 | true | 0.0256 | 0.0225932, 0.0288872 | null | null |
| calibration/C09/90/0.975/P-F | 0.975 | true | 10000 | 245 | 317 | true | 0.0245 | 0.0215591, 0.0277221 | null | null |
| calibration/C09/90/0.975/joint | 0.975 | true | 10000 | 492 | 593 | true | 0.0492 | 0.0450427, 0.0536223 | null | null |
| calibration/C09/90/0.980/A-F | 0.98 | true | 10000 | 206 | 317 | true | 0.0206 | 0.0179063, 0.0235777 | null | null |
| calibration/C09/90/0.980/P-F | 0.98 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| calibration/C09/90/0.980/joint | 0.98 | true | 10000 | 401 | 593 | true | 0.0401 | 0.0363392, 0.0441316 | null | null |
| calibration/C09/90/0.985/A-F | 0.985 | true | 10000 | 148 | 317 | true | 0.0148 | 0.0125254, 0.0173632 | null | null |
| calibration/C09/90/0.985/P-F | 0.985 | true | 10000 | 147 | 317 | true | 0.0147 | 0.0124334, 0.0172553 | null | null |
| calibration/C09/90/0.985/joint | 0.985 | true | 10000 | 292 | 593 | true | 0.0292 | 0.0259878, 0.0326901 | null | null |
| calibration/C09/90/0.990/A-F | 0.99 | true | 10000 | 104 | 317 | true | 0.0104 | 0.00850524, 0.0125874 | null | null |
| calibration/C09/90/0.990/P-F | 0.99 | true | 10000 | 96 | 317 | true | 0.0096 | 0.00778275, 0.0117108 | null | null |
| calibration/C09/90/0.990/joint | 0.99 | true | 10000 | 199 | 593 | true | 0.0199 | 0.0172533, 0.0228312 | null | null |
| calibration/C09/99/0.975/A-F | 0.975 | true | 10000 | 260 | 317 | true | 0.026 | 0.0229697, 0.0293105 | null | null |
| calibration/C09/99/0.975/P-F | 0.975 | true | 10000 | 248 | 317 | true | 0.0248 | 0.021841, 0.02804 | null | null |
| calibration/C09/99/0.975/joint | 0.975 | true | 10000 | 502 | 593 | true | 0.0502 | 0.0460018, 0.0546625 | null | null |
| calibration/C09/99/0.980/A-F | 0.98 | true | 10000 | 200 | 317 | true | 0.02 | 0.0173465, 0.0229379 | null | null |
| calibration/C09/99/0.980/P-F | 0.98 | true | 10000 | 200 | 317 | true | 0.02 | 0.0173465, 0.0229379 | null | null |
| calibration/C09/99/0.980/joint | 0.98 | true | 10000 | 394 | 593 | true | 0.0394 | 0.0356717, 0.0433995 | null | null |
| calibration/C09/99/0.985/A-F | 0.985 | true | 10000 | 155 | 317 | true | 0.0155 | 0.0131707, 0.0181173 | null | null |
| calibration/C09/99/0.985/P-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| calibration/C09/99/0.985/joint | 0.985 | true | 10000 | 304 | 593 | true | 0.0304 | 0.0271224, 0.0339547 | null | null |
| calibration/C09/99/0.990/A-F | 0.99 | true | 10000 | 100 | 317 | true | 0.01 | 0.0081436, 0.0121495 | null | null |
| calibration/C09/99/0.990/P-F | 0.99 | true | 10000 | 92 | 317 | true | 0.0092 | 0.00742276, 0.0112712 | null | null |
| calibration/C09/99/0.990/joint | 0.99 | true | 10000 | 192 | 593 | true | 0.0192 | 0.0166011, 0.0220839 | null | null |
| calibration/C09/cap/0.975/A-F | 0.975 | true | 10000 | 269 | 317 | true | 0.0269 | 0.0238173, 0.0302622 | null | 60 |
| calibration/C09/cap/0.975/P-F | 0.975 | true | 10000 | 232 | 317 | true | 0.0232 | 0.020339, 0.0263431 | null | 60 |
| calibration/C09/cap/0.975/joint | 0.975 | true | 10000 | 491 | 593 | true | 0.0491 | 0.0449468, 0.0535182 | null | 60 |
| calibration/C09/cap/0.980/A-F | 0.98 | true | 10000 | 201 | 317 | true | 0.0201 | 0.0174398, 0.0230446 | null | 60 |
| calibration/C09/cap/0.980/P-F | 0.98 | true | 10000 | 189 | 317 | true | 0.0189 | 0.0163219, 0.0217634 | null | 60 |
| calibration/C09/cap/0.980/joint | 0.98 | true | 10000 | 383 | 593 | true | 0.0383 | 0.0346235, 0.0422484 | null | 60 |
| calibration/C09/cap/0.985/A-F | 0.985 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | 60 |
| calibration/C09/cap/0.985/P-F | 0.985 | true | 10000 | 136 | 317 | true | 0.0136 | 0.0114224, 0.0160672 | null | 60 |
| calibration/C09/cap/0.985/joint | 0.985 | true | 10000 | 286 | 593 | true | 0.0286 | 0.025421, 0.0320573 | null | 60 |
| calibration/C09/cap/0.990/A-F | 0.99 | true | 10000 | 94 | 317 | true | 0.0094 | 0.00760265, 0.0114911 | null | 60 |
| calibration/C09/cap/0.990/P-F | 0.99 | true | 10000 | 76 | 317 | true | 0.0076 | 0.00599247, 0.00950343 | null | 60 |
| calibration/C09/cap/0.990/joint | 0.99 | true | 10000 | 168 | 593 | true | 0.0168 | 0.0143725, 0.0195144 | null | 60 |
| calibration/C10/36/0.975/A-F | 0.975 | true | 10000 | 260 | 317 | true | 0.026 | 0.0229697, 0.0293105 | null | null |
| calibration/C10/36/0.975/P-F | 0.975 | true | 10000 | 225 | 317 | true | 0.0225 | 0.019683, 0.0255996 | null | null |
| calibration/C10/36/0.975/joint | 0.975 | true | 10000 | 479 | 593 | true | 0.0479 | 0.0437966, 0.0522692 | null | null |
| calibration/C10/36/0.980/A-F | 0.98 | true | 10000 | 207 | 317 | true | 0.0207 | 0.0179997, 0.0236842 | null | null |
| calibration/C10/36/0.980/P-F | 0.98 | true | 10000 | 175 | 317 | true | 0.0175 | 0.0150213, 0.0202651 | null | null |
| calibration/C10/36/0.980/joint | 0.98 | true | 10000 | 380 | 593 | true | 0.038 | 0.0343377, 0.0419344 | null | null |
| calibration/C10/36/0.985/A-F | 0.985 | true | 10000 | 155 | 317 | true | 0.0155 | 0.0131707, 0.0181173 | null | null |
| calibration/C10/36/0.985/P-F | 0.985 | true | 10000 | 137 | 317 | true | 0.0137 | 0.0115142, 0.0161754 | null | null |
| calibration/C10/36/0.985/joint | 0.985 | true | 10000 | 290 | 593 | true | 0.029 | 0.0257988, 0.0324792 | null | null |
| calibration/C10/36/0.990/A-F | 0.99 | true | 10000 | 109 | 317 | true | 0.0109 | 0.00895834, 0.0131338 | null | null |
| calibration/C10/36/0.990/P-F | 0.99 | true | 10000 | 91 | 317 | true | 0.0091 | 0.0073329, 0.0111612 | null | null |
| calibration/C10/36/0.990/joint | 0.99 | true | 10000 | 199 | 593 | true | 0.0199 | 0.0172533, 0.0228312 | null | null |
| calibration/C10/45/0.975/A-F | 0.975 | true | 10000 | 263 | 317 | true | 0.0263 | 0.0232521, 0.0296278 | null | null |
| calibration/C10/45/0.975/P-F | 0.975 | true | 10000 | 228 | 317 | true | 0.0228 | 0.0199641, 0.0259183 | null | null |
| calibration/C10/45/0.975/joint | 0.975 | true | 10000 | 487 | 593 | true | 0.0487 | 0.0445633, 0.0531019 | null | null |
| calibration/C10/45/0.980/A-F | 0.98 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| calibration/C10/45/0.980/P-F | 0.98 | true | 10000 | 179 | 317 | true | 0.0179 | 0.0153925, 0.0206936 | null | null |
| calibration/C10/45/0.980/joint | 0.98 | true | 10000 | 380 | 593 | true | 0.038 | 0.0343377, 0.0419344 | null | null |
| calibration/C10/45/0.985/A-F | 0.985 | true | 10000 | 148 | 317 | true | 0.0148 | 0.0125254, 0.0173632 | null | null |
| calibration/C10/45/0.985/P-F | 0.985 | true | 10000 | 130 | 317 | true | 0.013 | 0.0108726, 0.0154176 | null | null |
| calibration/C10/45/0.985/joint | 0.985 | true | 10000 | 278 | 593 | true | 0.0278 | 0.0246659, 0.0312129 | null | null |
| calibration/C10/45/0.990/A-F | 0.99 | true | 10000 | 104 | 317 | true | 0.0104 | 0.00850524, 0.0125874 | null | null |
| calibration/C10/45/0.990/P-F | 0.99 | true | 10000 | 86 | 317 | true | 0.0086 | 0.00688448, 0.0106102 | null | null |
| calibration/C10/45/0.990/joint | 0.99 | true | 10000 | 190 | 593 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| calibration/C10/60/0.975/A-F | 0.975 | true | 10000 | 256 | 317 | true | 0.0256 | 0.0225932, 0.0288872 | null | null |
| calibration/C10/60/0.975/P-F | 0.975 | true | 10000 | 212 | 317 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| calibration/C10/60/0.975/joint | 0.975 | true | 10000 | 462 | 593 | true | 0.0462 | 0.0421684, 0.0504985 | null | null |
| calibration/C10/60/0.980/A-F | 0.98 | true | 10000 | 206 | 317 | true | 0.0206 | 0.0179063, 0.0235777 | null | null |
| calibration/C10/60/0.980/P-F | 0.98 | true | 10000 | 168 | 317 | true | 0.0168 | 0.0143725, 0.0195144 | null | null |
| calibration/C10/60/0.980/joint | 0.98 | true | 10000 | 371 | 593 | true | 0.0371 | 0.0334809, 0.0409918 | null | null |
| calibration/C10/60/0.985/A-F | 0.985 | true | 10000 | 175 | 317 | true | 0.0175 | 0.0150213, 0.0202651 | null | null |
| calibration/C10/60/0.985/P-F | 0.985 | true | 10000 | 121 | 317 | true | 0.0121 | 0.0100501, 0.0144409 | null | null |
| calibration/C10/60/0.985/joint | 0.985 | true | 10000 | 296 | 593 | true | 0.0296 | 0.0263658, 0.0331118 | null | null |
| calibration/C10/60/0.990/A-F | 0.99 | true | 10000 | 123 | 317 | true | 0.0123 | 0.0102326, 0.0146582 | null | null |
| calibration/C10/60/0.990/P-F | 0.99 | true | 10000 | 84 | 317 | true | 0.0084 | 0.00670554, 0.0103894 | null | null |
| calibration/C10/60/0.990/joint | 0.99 | true | 10000 | 207 | 593 | true | 0.0207 | 0.0179997, 0.0236842 | null | null |
| calibration/C10/75/0.975/A-F | 0.975 | true | 10000 | 216 | 317 | true | 0.0216 | 0.0188407, 0.0246425 | null | null |
| calibration/C10/75/0.975/P-F | 0.975 | true | 10000 | 213 | 317 | true | 0.0213 | 0.0185602, 0.0243233 | null | null |
| calibration/C10/75/0.975/joint | 0.975 | true | 10000 | 424 | 593 | true | 0.0424 | 0.0385345, 0.0465348 | null | null |
| calibration/C10/75/0.980/A-F | 0.98 | true | 10000 | 175 | 317 | true | 0.0175 | 0.0150213, 0.0202651 | null | null |
| calibration/C10/75/0.980/P-F | 0.98 | true | 10000 | 169 | 317 | true | 0.0169 | 0.0144651, 0.0196217 | null | null |
| calibration/C10/75/0.980/joint | 0.98 | true | 10000 | 341 | 593 | true | 0.0341 | 0.0306292, 0.0378455 | null | null |
| calibration/C10/75/0.985/A-F | 0.985 | true | 10000 | 130 | 317 | true | 0.013 | 0.0108726, 0.0154176 | null | null |
| calibration/C10/75/0.985/P-F | 0.985 | true | 10000 | 127 | 317 | true | 0.0127 | 0.0105981, 0.0150923 | null | null |
| calibration/C10/75/0.985/joint | 0.985 | true | 10000 | 255 | 593 | true | 0.0255 | 0.0224992, 0.0287813 | null | null |
| calibration/C10/75/0.990/A-F | 0.99 | true | 10000 | 94 | 317 | true | 0.0094 | 0.00760265, 0.0114911 | null | null |
| calibration/C10/75/0.990/P-F | 0.99 | true | 10000 | 81 | 317 | true | 0.0081 | 0.00643762, 0.0100576 | null | null |
| calibration/C10/75/0.990/joint | 0.99 | true | 10000 | 175 | 593 | true | 0.0175 | 0.0150213, 0.0202651 | null | null |
| calibration/C10/90/0.975/A-F | 0.975 | true | 10000 | 224 | 317 | true | 0.0224 | 0.0195894, 0.0254933 | null | null |
| calibration/C10/90/0.975/P-F | 0.975 | true | 10000 | 216 | 317 | true | 0.0216 | 0.0188407, 0.0246425 | null | null |
| calibration/C10/90/0.975/joint | 0.975 | true | 10000 | 433 | 593 | true | 0.0433 | 0.0393944, 0.0474743 | null | null |
| calibration/C10/90/0.980/A-F | 0.98 | true | 10000 | 185 | 317 | true | 0.0185 | 0.0159499, 0.0213357 | null | null |
| calibration/C10/90/0.980/P-F | 0.98 | true | 10000 | 169 | 317 | true | 0.0169 | 0.0144651, 0.0196217 | null | null |
| calibration/C10/90/0.980/joint | 0.98 | true | 10000 | 348 | 593 | true | 0.0348 | 0.031294, 0.0385802 | null | null |
| calibration/C10/90/0.985/A-F | 0.985 | true | 10000 | 135 | 317 | true | 0.0135 | 0.0113307, 0.015959 | null | null |
| calibration/C10/90/0.985/P-F | 0.985 | true | 10000 | 123 | 317 | true | 0.0123 | 0.0102326, 0.0146582 | null | null |
| calibration/C10/90/0.985/joint | 0.985 | true | 10000 | 256 | 593 | true | 0.0256 | 0.0225932, 0.0288872 | null | null |
| calibration/C10/90/0.990/A-F | 0.99 | true | 10000 | 83 | 317 | true | 0.0083 | 0.00661617, 0.0102789 | null | null |
| calibration/C10/90/0.990/P-F | 0.99 | true | 10000 | 83 | 317 | true | 0.0083 | 0.00661617, 0.0102789 | null | null |
| calibration/C10/90/0.990/joint | 0.99 | true | 10000 | 166 | 593 | true | 0.0166 | 0.0141874, 0.0192998 | null | null |
| calibration/C10/99/0.975/A-F | 0.975 | true | 10000 | 257 | 317 | true | 0.0257 | 0.0226873, 0.028993 | null | null |
| calibration/C10/99/0.975/P-F | 0.975 | true | 10000 | 218 | 317 | true | 0.0218 | 0.0190278, 0.0248553 | null | null |
| calibration/C10/99/0.975/joint | 0.975 | true | 10000 | 472 | 593 | true | 0.0472 | 0.043126, 0.0515402 | null | null |
| calibration/C10/99/0.980/A-F | 0.98 | true | 10000 | 201 | 317 | true | 0.0201 | 0.0174398, 0.0230446 | null | null |
| calibration/C10/99/0.980/P-F | 0.98 | true | 10000 | 178 | 317 | true | 0.0178 | 0.0152996, 0.0205865 | null | null |
| calibration/C10/99/0.980/joint | 0.98 | true | 10000 | 376 | 593 | true | 0.0376 | 0.0339568, 0.0415155 | null | null |
| calibration/C10/99/0.985/A-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| calibration/C10/99/0.985/P-F | 0.985 | true | 10000 | 129 | 317 | true | 0.0129 | 0.0107811, 0.0153092 | null | null |
| calibration/C10/99/0.985/joint | 0.985 | true | 10000 | 280 | 593 | true | 0.028 | 0.0248547, 0.0314241 | null | null |
| calibration/C10/99/0.990/A-F | 0.99 | true | 10000 | 98 | 317 | true | 0.0098 | 0.00796307, 0.0119302 | null | null |
| calibration/C10/99/0.990/P-F | 0.99 | true | 10000 | 86 | 317 | true | 0.0086 | 0.00688448, 0.0106102 | null | null |
| calibration/C10/99/0.990/joint | 0.99 | true | 10000 | 184 | 593 | true | 0.0184 | 0.0158569, 0.0212288 | null | null |
| calibration/C10/cap/0.975/A-F | 0.975 | true | 10000 | 253 | 317 | true | 0.0253 | 0.022311, 0.0285696 | null | 60 |
| calibration/C10/cap/0.975/P-F | 0.975 | true | 10000 | 208 | 317 | true | 0.0208 | 0.0180931, 0.0237908 | null | 60 |
| calibration/C10/cap/0.975/joint | 0.975 | true | 10000 | 457 | 593 | true | 0.0457 | 0.0416898, 0.0499774 | null | 60 |
| calibration/C10/cap/0.980/A-F | 0.98 | true | 10000 | 194 | 317 | true | 0.0194 | 0.0167874, 0.0222975 | null | 60 |
| calibration/C10/cap/0.980/P-F | 0.98 | true | 10000 | 163 | 317 | true | 0.0163 | 0.0139098, 0.0189775 | null | 60 |
| calibration/C10/cap/0.980/joint | 0.98 | true | 10000 | 353 | 593 | true | 0.0353 | 0.031769, 0.0391048 | null | 60 |
| calibration/C10/cap/0.985/A-F | 0.985 | true | 10000 | 159 | 317 | true | 0.0159 | 0.0135401, 0.0185476 | null | 60 |
| calibration/C10/cap/0.985/P-F | 0.985 | true | 10000 | 114 | 317 | true | 0.0114 | 0.00941254, 0.0136791 | null | 60 |
| calibration/C10/cap/0.985/joint | 0.985 | true | 10000 | 271 | 593 | true | 0.0271 | 0.0240058, 0.0304735 | null | 60 |
| calibration/C10/cap/0.990/A-F | 0.99 | true | 10000 | 89 | 317 | true | 0.0089 | 0.00715335, 0.010941 | null | 60 |
| calibration/C10/cap/0.990/P-F | 0.99 | true | 10000 | 80 | 317 | true | 0.008 | 0.00634845, 0.00994694 | null | 60 |
| calibration/C10/cap/0.990/joint | 0.99 | true | 10000 | 167 | 593 | true | 0.0167 | 0.0142799, 0.0194071 | null | 60 |
| calibration/C11/36/0.975/A-F | 0.975 | true | 10000 | 211 | 317 | true | 0.0211 | 0.0183733, 0.0241103 | null | null |
| calibration/C11/36/0.975/P-F | 0.975 | true | 10000 | 255 | 317 | true | 0.0255 | 0.0224992, 0.0287813 | null | null |
| calibration/C11/36/0.975/joint | 0.975 | true | 10000 | 465 | 593 | true | 0.0465 | 0.0424556, 0.0508111 | null | null |
| calibration/C11/36/0.980/A-F | 0.98 | true | 10000 | 167 | 317 | true | 0.0167 | 0.0142799, 0.0194071 | null | null |
| calibration/C11/36/0.980/P-F | 0.98 | true | 10000 | 203 | 317 | true | 0.0203 | 0.0176264, 0.0232579 | null | null |
| calibration/C11/36/0.980/joint | 0.98 | true | 10000 | 369 | 593 | true | 0.0369 | 0.0332906, 0.0407822 | null | null |
| calibration/C11/36/0.985/A-F | 0.985 | true | 10000 | 129 | 317 | true | 0.0129 | 0.0107811, 0.0153092 | null | null |
| calibration/C11/36/0.985/P-F | 0.985 | true | 10000 | 148 | 317 | true | 0.0148 | 0.0125254, 0.0173632 | null | null |
| calibration/C11/36/0.985/joint | 0.985 | true | 10000 | 277 | 593 | true | 0.0277 | 0.0245716, 0.0311073 | null | null |
| calibration/C11/36/0.990/A-F | 0.99 | true | 10000 | 91 | 317 | true | 0.0091 | 0.0073329, 0.0111612 | null | null |
| calibration/C11/36/0.990/P-F | 0.99 | true | 10000 | 100 | 317 | true | 0.01 | 0.0081436, 0.0121495 | null | null |
| calibration/C11/36/0.990/joint | 0.99 | true | 10000 | 191 | 593 | true | 0.0191 | 0.016508, 0.0219771 | null | null |
| calibration/C11/45/0.975/A-F | 0.975 | true | 10000 | 223 | 317 | true | 0.0223 | 0.0194957, 0.025387 | null | null |
| calibration/C11/45/0.975/P-F | 0.975 | true | 10000 | 258 | 317 | true | 0.0258 | 0.0227814, 0.0290988 | null | null |
| calibration/C11/45/0.975/joint | 0.975 | true | 10000 | 473 | 593 | true | 0.0473 | 0.0432218, 0.0516444 | null | null |
| calibration/C11/45/0.980/A-F | 0.98 | true | 10000 | 181 | 317 | true | 0.0181 | 0.0155782, 0.0209077 | null | null |
| calibration/C11/45/0.980/P-F | 0.98 | true | 10000 | 209 | 317 | true | 0.0209 | 0.0181865, 0.0238973 | null | null |
| calibration/C11/45/0.980/joint | 0.98 | true | 10000 | 385 | 593 | true | 0.0385 | 0.034814, 0.0424578 | null | null |
| calibration/C11/45/0.985/A-F | 0.985 | true | 10000 | 136 | 317 | true | 0.0136 | 0.0114224, 0.0160672 | null | null |
| calibration/C11/45/0.985/P-F | 0.985 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | null |
| calibration/C11/45/0.985/joint | 0.985 | true | 10000 | 291 | 593 | true | 0.0291 | 0.0258933, 0.0325847 | null | null |
| calibration/C11/45/0.990/A-F | 0.99 | true | 10000 | 92 | 317 | true | 0.0092 | 0.00742276, 0.0112712 | null | null |
| calibration/C11/45/0.990/P-F | 0.99 | true | 10000 | 90 | 317 | true | 0.009 | 0.0072431, 0.0110511 | null | null |
| calibration/C11/45/0.990/joint | 0.99 | true | 10000 | 182 | 593 | true | 0.0182 | 0.0156711, 0.0210148 | null | null |
| calibration/C11/60/0.975/A-F | 0.975 | true | 10000 | 216 | 317 | true | 0.0216 | 0.0188407, 0.0246425 | null | null |
| calibration/C11/60/0.975/P-F | 0.975 | true | 10000 | 249 | 317 | true | 0.0249 | 0.021935, 0.0281459 | null | null |
| calibration/C11/60/0.975/joint | 0.975 | true | 10000 | 459 | 593 | true | 0.0459 | 0.0418812, 0.0501858 | null | null |
| calibration/C11/60/0.980/A-F | 0.98 | true | 10000 | 175 | 317 | true | 0.0175 | 0.0150213, 0.0202651 | null | null |
| calibration/C11/60/0.980/P-F | 0.98 | true | 10000 | 201 | 317 | true | 0.0201 | 0.0174398, 0.0230446 | null | null |
| calibration/C11/60/0.980/joint | 0.98 | true | 10000 | 372 | 593 | true | 0.0372 | 0.0335761, 0.0410965 | null | null |
| calibration/C11/60/0.985/A-F | 0.985 | true | 10000 | 130 | 317 | true | 0.013 | 0.0108726, 0.0154176 | null | null |
| calibration/C11/60/0.985/P-F | 0.985 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | null |
| calibration/C11/60/0.985/joint | 0.985 | true | 10000 | 278 | 593 | true | 0.0278 | 0.0246659, 0.0312129 | null | null |
| calibration/C11/60/0.990/A-F | 0.99 | true | 10000 | 87 | 317 | true | 0.0087 | 0.00697404, 0.0107205 | null | null |
| calibration/C11/60/0.990/P-F | 0.99 | true | 10000 | 107 | 317 | true | 0.0107 | 0.00877696, 0.0129154 | null | null |
| calibration/C11/60/0.990/joint | 0.99 | true | 10000 | 193 | 593 | true | 0.0193 | 0.0166942, 0.0221908 | null | null |
| calibration/C11/75/0.975/A-F | 0.975 | true | 10000 | 251 | 317 | true | 0.0251 | 0.022123, 0.0283578 | null | null |
| calibration/C11/75/0.975/P-F | 0.975 | true | 10000 | 217 | 317 | true | 0.0217 | 0.0189343, 0.0247489 | null | null |
| calibration/C11/75/0.975/joint | 0.975 | true | 10000 | 463 | 593 | true | 0.0463 | 0.0422641, 0.0506027 | null | null |
| calibration/C11/75/0.980/A-F | 0.98 | true | 10000 | 197 | 317 | true | 0.0197 | 0.0170669, 0.0226178 | null | null |
| calibration/C11/75/0.980/P-F | 0.98 | true | 10000 | 177 | 317 | true | 0.0177 | 0.0152068, 0.0204794 | null | null |
| calibration/C11/75/0.980/joint | 0.98 | true | 10000 | 371 | 593 | true | 0.0371 | 0.0334809, 0.0409918 | null | null |
| calibration/C11/75/0.985/A-F | 0.985 | true | 10000 | 154 | 317 | true | 0.0154 | 0.0130785, 0.0180096 | null | null |
| calibration/C11/75/0.985/P-F | 0.985 | true | 10000 | 134 | 317 | true | 0.0134 | 0.011239, 0.0158508 | null | null |
| calibration/C11/75/0.985/joint | 0.985 | true | 10000 | 285 | 593 | true | 0.0285 | 0.0253266, 0.0319518 | null | null |
| calibration/C11/75/0.990/A-F | 0.99 | true | 10000 | 95 | 317 | true | 0.0095 | 0.00769267, 0.011601 | null | null |
| calibration/C11/75/0.990/P-F | 0.99 | true | 10000 | 89 | 317 | true | 0.0089 | 0.00715335, 0.010941 | null | null |
| calibration/C11/75/0.990/joint | 0.99 | true | 10000 | 184 | 593 | true | 0.0184 | 0.0158569, 0.0212288 | null | null |
| calibration/C11/90/0.975/A-F | 0.975 | true | 10000 | 263 | 317 | true | 0.0263 | 0.0232521, 0.0296278 | null | null |
| calibration/C11/90/0.975/P-F | 0.975 | true | 10000 | 232 | 317 | true | 0.0232 | 0.020339, 0.0263431 | null | null |
| calibration/C11/90/0.975/joint | 0.975 | true | 10000 | 488 | 593 | true | 0.0488 | 0.0446592, 0.053206 | null | null |
| calibration/C11/90/0.980/A-F | 0.98 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| calibration/C11/90/0.980/P-F | 0.98 | true | 10000 | 196 | 317 | true | 0.0196 | 0.0169737, 0.0225111 | null | null |
| calibration/C11/90/0.980/joint | 0.98 | true | 10000 | 394 | 593 | true | 0.0394 | 0.0356717, 0.0433995 | null | null |
| calibration/C11/90/0.985/A-F | 0.985 | true | 10000 | 159 | 317 | true | 0.0159 | 0.0135401, 0.0185476 | null | null |
| calibration/C11/90/0.985/P-F | 0.985 | true | 10000 | 144 | 317 | true | 0.0144 | 0.0121573, 0.0169316 | null | null |
| calibration/C11/90/0.985/joint | 0.985 | true | 10000 | 301 | 593 | true | 0.0301 | 0.0268386, 0.0336387 | null | null |
| calibration/C11/90/0.990/A-F | 0.99 | true | 10000 | 110 | 317 | true | 0.011 | 0.0090491, 0.013243 | null | null |
| calibration/C11/90/0.990/P-F | 0.99 | true | 10000 | 84 | 317 | true | 0.0084 | 0.00670554, 0.0103894 | null | null |
| calibration/C11/90/0.990/joint | 0.99 | true | 10000 | 193 | 593 | true | 0.0193 | 0.0166942, 0.0221908 | null | null |
| calibration/C11/99/0.975/A-F | 0.975 | true | 10000 | 254 | 317 | true | 0.0254 | 0.0224051, 0.0286754 | null | null |
| calibration/C11/99/0.975/P-F | 0.975 | true | 10000 | 242 | 317 | true | 0.0242 | 0.0212773, 0.0274041 | null | null |
| calibration/C11/99/0.975/joint | 0.975 | true | 10000 | 488 | 593 | true | 0.0488 | 0.0446592, 0.053206 | null | null |
| calibration/C11/99/0.980/A-F | 0.98 | true | 10000 | 201 | 317 | true | 0.0201 | 0.0174398, 0.0230446 | null | null |
| calibration/C11/99/0.980/P-F | 0.98 | true | 10000 | 195 | 317 | true | 0.0195 | 0.0168805, 0.0224043 | null | null |
| calibration/C11/99/0.980/joint | 0.98 | true | 10000 | 392 | 593 | true | 0.0392 | 0.035481, 0.0431903 | null | null |
| calibration/C11/99/0.985/A-F | 0.985 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| calibration/C11/99/0.985/P-F | 0.985 | true | 10000 | 142 | 317 | true | 0.0142 | 0.0119734, 0.0167157 | null | null |
| calibration/C11/99/0.985/joint | 0.985 | true | 10000 | 290 | 593 | true | 0.029 | 0.0257988, 0.0324792 | null | null |
| calibration/C11/99/0.990/A-F | 0.99 | true | 10000 | 96 | 317 | true | 0.0096 | 0.00778275, 0.0117108 | null | null |
| calibration/C11/99/0.990/P-F | 0.99 | true | 10000 | 87 | 317 | true | 0.0087 | 0.00697404, 0.0107205 | null | null |
| calibration/C11/99/0.990/joint | 0.99 | true | 10000 | 182 | 593 | true | 0.0182 | 0.0156711, 0.0210148 | null | null |
| calibration/C11/cap/0.975/A-F | 0.975 | true | 10000 | 250 | 317 | true | 0.025 | 0.022029, 0.0282519 | null | 60 |
| calibration/C11/cap/0.975/P-F | 0.975 | true | 10000 | 230 | 317 | true | 0.023 | 0.0201515, 0.0261307 | null | 60 |
| calibration/C11/cap/0.975/joint | 0.975 | true | 10000 | 475 | 593 | true | 0.0475 | 0.0434134, 0.0518527 | null | 60 |
| calibration/C11/cap/0.980/A-F | 0.98 | true | 10000 | 208 | 317 | true | 0.0208 | 0.0180931, 0.0237908 | null | 60 |
| calibration/C11/cap/0.980/P-F | 0.98 | true | 10000 | 189 | 317 | true | 0.0189 | 0.0163219, 0.0217634 | null | 60 |
| calibration/C11/cap/0.980/joint | 0.98 | true | 10000 | 393 | 593 | true | 0.0393 | 0.0355764, 0.0432949 | null | 60 |
| calibration/C11/cap/0.985/A-F | 0.985 | true | 10000 | 162 | 317 | true | 0.0162 | 0.0138173, 0.0188701 | null | 60 |
| calibration/C11/cap/0.985/P-F | 0.985 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | 60 |
| calibration/C11/cap/0.985/joint | 0.985 | true | 10000 | 304 | 593 | true | 0.0304 | 0.0271224, 0.0339547 | null | 60 |
| calibration/C11/cap/0.990/A-F | 0.99 | true | 10000 | 101 | 317 | true | 0.0101 | 0.00823393, 0.0122591 | null | 60 |
| calibration/C11/cap/0.990/P-F | 0.99 | true | 10000 | 91 | 317 | true | 0.0091 | 0.0073329, 0.0111612 | null | 60 |
| calibration/C11/cap/0.990/joint | 0.99 | true | 10000 | 190 | 593 | true | 0.019 | 0.016415, 0.0218703 | null | 60 |
| calibration/C12/36/0.975/A-F | 0.975 | true | 10000 | 233 | 317 | true | 0.0233 | 0.0204328, 0.0264493 | null | null |
| calibration/C12/36/0.975/P-F | 0.975 | true | 10000 | 255 | 317 | true | 0.0255 | 0.0224992, 0.0287813 | null | null |
| calibration/C12/36/0.975/joint | 0.975 | true | 10000 | 482 | 593 | true | 0.0482 | 0.0440841, 0.0525815 | null | null |
| calibration/C12/36/0.980/A-F | 0.98 | true | 10000 | 181 | 317 | true | 0.0181 | 0.0155782, 0.0209077 | null | null |
| calibration/C12/36/0.980/P-F | 0.98 | true | 10000 | 207 | 317 | true | 0.0207 | 0.0179997, 0.0236842 | null | null |
| calibration/C12/36/0.980/joint | 0.98 | true | 10000 | 382 | 593 | true | 0.0382 | 0.0345282, 0.0421438 | null | null |
| calibration/C12/36/0.985/A-F | 0.985 | true | 10000 | 133 | 317 | true | 0.0133 | 0.0111474, 0.0157426 | null | null |
| calibration/C12/36/0.985/P-F | 0.985 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | null |
| calibration/C12/36/0.985/joint | 0.985 | true | 10000 | 277 | 593 | true | 0.0277 | 0.0245716, 0.0311073 | null | null |
| calibration/C12/36/0.990/A-F | 0.99 | true | 10000 | 85 | 317 | true | 0.0085 | 0.00679498, 0.0104998 | null | null |
| calibration/C12/36/0.990/P-F | 0.99 | true | 10000 | 95 | 317 | true | 0.0095 | 0.00769267, 0.011601 | null | null |
| calibration/C12/36/0.990/joint | 0.99 | true | 10000 | 178 | 593 | true | 0.0178 | 0.0152996, 0.0205865 | null | null |
| calibration/C12/45/0.975/A-F | 0.975 | true | 10000 | 256 | 317 | true | 0.0256 | 0.0225932, 0.0288872 | null | null |
| calibration/C12/45/0.975/P-F | 0.975 | true | 10000 | 273 | 317 | true | 0.0273 | 0.0241944, 0.0306848 | null | null |
| calibration/C12/45/0.975/joint | 0.975 | true | 10000 | 513 | 593 | true | 0.0513 | 0.0470573, 0.0558063 | null | null |
| calibration/C12/45/0.980/A-F | 0.98 | true | 10000 | 208 | 317 | true | 0.0208 | 0.0180931, 0.0237908 | null | null |
| calibration/C12/45/0.980/P-F | 0.98 | true | 10000 | 211 | 317 | true | 0.0211 | 0.0183733, 0.0241103 | null | null |
| calibration/C12/45/0.980/joint | 0.98 | true | 10000 | 408 | 593 | true | 0.0408 | 0.0370069, 0.0448633 | null | null |
| calibration/C12/45/0.985/A-F | 0.985 | true | 10000 | 161 | 317 | true | 0.0161 | 0.0137249, 0.0187626 | null | null |
| calibration/C12/45/0.985/P-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| calibration/C12/45/0.985/joint | 0.985 | true | 10000 | 308 | 593 | true | 0.0308 | 0.0275009, 0.034376 | null | null |
| calibration/C12/45/0.990/A-F | 0.99 | true | 10000 | 101 | 317 | true | 0.0101 | 0.00823393, 0.0122591 | null | null |
| calibration/C12/45/0.990/P-F | 0.99 | true | 10000 | 90 | 317 | true | 0.009 | 0.0072431, 0.0110511 | null | null |
| calibration/C12/45/0.990/joint | 0.99 | true | 10000 | 191 | 593 | true | 0.0191 | 0.016508, 0.0219771 | null | null |
| calibration/C12/60/0.975/A-F | 0.975 | true | 10000 | 239 | 317 | true | 0.0239 | 0.0209957, 0.0270859 | null | null |
| calibration/C12/60/0.975/P-F | 0.975 | true | 10000 | 286 | 317 | true | 0.0286 | 0.025421, 0.0320573 | null | null |
| calibration/C12/60/0.975/joint | 0.975 | true | 10000 | 517 | 593 | true | 0.0517 | 0.0474413, 0.0562221 | null | null |
| calibration/C12/60/0.980/A-F | 0.98 | true | 10000 | 198 | 317 | true | 0.0198 | 0.0171601, 0.0227245 | null | null |
| calibration/C12/60/0.980/P-F | 0.98 | true | 10000 | 225 | 317 | true | 0.0225 | 0.019683, 0.0255996 | null | null |
| calibration/C12/60/0.980/joint | 0.98 | true | 10000 | 417 | 593 | true | 0.0417 | 0.037866, 0.0458037 | null | null |
| calibration/C12/60/0.985/A-F | 0.985 | true | 10000 | 149 | 317 | true | 0.0149 | 0.0126175, 0.017471 | null | null |
| calibration/C12/60/0.985/P-F | 0.985 | true | 10000 | 169 | 317 | true | 0.0169 | 0.0144651, 0.0196217 | null | null |
| calibration/C12/60/0.985/joint | 0.985 | true | 10000 | 315 | 593 | true | 0.0315 | 0.0281637, 0.0351127 | null | null |
| calibration/C12/60/0.990/A-F | 0.99 | true | 10000 | 105 | 317 | true | 0.0105 | 0.00859577, 0.0126968 | null | null |
| calibration/C12/60/0.990/P-F | 0.99 | true | 10000 | 112 | 317 | true | 0.0112 | 0.00923074, 0.0134611 | null | null |
| calibration/C12/60/0.990/joint | 0.99 | true | 10000 | 216 | 593 | true | 0.0216 | 0.0188407, 0.0246425 | null | null |
| calibration/C12/75/0.975/A-F | 0.975 | true | 10000 | 253 | 317 | true | 0.0253 | 0.022311, 0.0285696 | null | null |
| calibration/C12/75/0.975/P-F | 0.975 | true | 10000 | 284 | 317 | true | 0.0284 | 0.0252322, 0.0318463 | null | null |
| calibration/C12/75/0.975/joint | 0.975 | true | 10000 | 530 | 593 | true | 0.053 | 0.0486897, 0.0575729 | null | null |
| calibration/C12/75/0.980/A-F | 0.98 | true | 10000 | 211 | 317 | true | 0.0211 | 0.0183733, 0.0241103 | null | null |
| calibration/C12/75/0.980/P-F | 0.98 | true | 10000 | 225 | 317 | true | 0.0225 | 0.019683, 0.0255996 | null | null |
| calibration/C12/75/0.980/joint | 0.98 | true | 10000 | 429 | 593 | true | 0.0429 | 0.0390122, 0.0470568 | null | null |
| calibration/C12/75/0.985/A-F | 0.985 | true | 10000 | 148 | 317 | true | 0.0148 | 0.0125254, 0.0173632 | null | null |
| calibration/C12/75/0.985/P-F | 0.985 | true | 10000 | 169 | 317 | true | 0.0169 | 0.0144651, 0.0196217 | null | null |
| calibration/C12/75/0.985/joint | 0.985 | true | 10000 | 314 | 593 | true | 0.0314 | 0.0280689, 0.0350075 | null | null |
| calibration/C12/75/0.990/A-F | 0.99 | true | 10000 | 99 | 317 | true | 0.0099 | 0.00805331, 0.0120399 | null | null |
| calibration/C12/75/0.990/P-F | 0.99 | true | 10000 | 119 | 317 | true | 0.0119 | 0.00986777, 0.0142234 | null | null |
| calibration/C12/75/0.990/joint | 0.99 | true | 10000 | 217 | 593 | true | 0.0217 | 0.0189343, 0.0247489 | null | null |
| calibration/C12/90/0.975/A-F | 0.975 | true | 10000 | 252 | 317 | true | 0.0252 | 0.022217, 0.0284637 | null | null |
| calibration/C12/90/0.975/P-F | 0.975 | true | 10000 | 268 | 317 | true | 0.0268 | 0.0237231, 0.0301565 | null | null |
| calibration/C12/90/0.975/joint | 0.975 | true | 10000 | 516 | 593 | true | 0.0516 | 0.0473453, 0.0561181 | null | null |
| calibration/C12/90/0.980/A-F | 0.98 | true | 10000 | 204 | 317 | true | 0.0204 | 0.0177197, 0.0233645 | null | null |
| calibration/C12/90/0.980/P-F | 0.98 | true | 10000 | 220 | 317 | true | 0.022 | 0.0192149, 0.025068 | null | null |
| calibration/C12/90/0.980/joint | 0.98 | true | 10000 | 422 | 593 | true | 0.0422 | 0.0383435, 0.0463259 | null | null |
| calibration/C12/90/0.985/A-F | 0.985 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | null |
| calibration/C12/90/0.985/P-F | 0.985 | true | 10000 | 168 | 317 | true | 0.0168 | 0.0143725, 0.0195144 | null | null |
| calibration/C12/90/0.985/joint | 0.985 | true | 10000 | 314 | 593 | true | 0.0314 | 0.0280689, 0.0350075 | null | null |
| calibration/C12/90/0.990/A-F | 0.99 | true | 10000 | 100 | 317 | true | 0.01 | 0.0081436, 0.0121495 | null | null |
| calibration/C12/90/0.990/P-F | 0.99 | true | 10000 | 114 | 317 | true | 0.0114 | 0.00941254, 0.0136791 | null | null |
| calibration/C12/90/0.990/joint | 0.99 | true | 10000 | 214 | 593 | true | 0.0214 | 0.0186537, 0.0244297 | null | null |
| calibration/C12/99/0.975/A-F | 0.975 | true | 10000 | 251 | 317 | true | 0.0251 | 0.022123, 0.0283578 | null | null |
| calibration/C12/99/0.975/P-F | 0.975 | true | 10000 | 266 | 317 | true | 0.0266 | 0.0235347, 0.029945 | null | null |
| calibration/C12/99/0.975/joint | 0.975 | true | 10000 | 513 | 593 | true | 0.0513 | 0.0470573, 0.0558063 | null | null |
| calibration/C12/99/0.980/A-F | 0.98 | true | 10000 | 203 | 317 | true | 0.0203 | 0.0176264, 0.0232579 | null | null |
| calibration/C12/99/0.980/P-F | 0.98 | true | 10000 | 214 | 317 | true | 0.0214 | 0.0186537, 0.0244297 | null | null |
| calibration/C12/99/0.980/joint | 0.98 | true | 10000 | 414 | 593 | true | 0.0414 | 0.0375796, 0.0454903 | null | null |
| calibration/C12/99/0.985/A-F | 0.985 | true | 10000 | 142 | 317 | true | 0.0142 | 0.0119734, 0.0167157 | null | null |
| calibration/C12/99/0.985/P-F | 0.985 | true | 10000 | 159 | 317 | true | 0.0159 | 0.0135401, 0.0185476 | null | null |
| calibration/C12/99/0.985/joint | 0.985 | true | 10000 | 301 | 593 | true | 0.0301 | 0.0268386, 0.0336387 | null | null |
| calibration/C12/99/0.990/A-F | 0.99 | true | 10000 | 87 | 317 | true | 0.0087 | 0.00697404, 0.0107205 | null | null |
| calibration/C12/99/0.990/P-F | 0.99 | true | 10000 | 102 | 317 | true | 0.0102 | 0.00832432, 0.0123686 | null | null |
| calibration/C12/99/0.990/joint | 0.99 | true | 10000 | 189 | 593 | true | 0.0189 | 0.0163219, 0.0217634 | null | null |
| calibration/C12/cap/0.975/A-F | 0.975 | true | 10000 | 248 | 317 | true | 0.0248 | 0.021841, 0.02804 | null | 60 |
| calibration/C12/cap/0.975/P-F | 0.975 | true | 10000 | 281 | 317 | true | 0.0281 | 0.024949, 0.0315297 | null | 60 |
| calibration/C12/cap/0.975/joint | 0.975 | true | 10000 | 517 | 593 | true | 0.0517 | 0.0474413, 0.0562221 | null | 60 |
| calibration/C12/cap/0.980/A-F | 0.98 | true | 10000 | 192 | 317 | true | 0.0192 | 0.0166011, 0.0220839 | null | 60 |
| calibration/C12/cap/0.980/P-F | 0.98 | true | 10000 | 217 | 317 | true | 0.0217 | 0.0189343, 0.0247489 | null | 60 |
| calibration/C12/cap/0.980/joint | 0.98 | true | 10000 | 402 | 593 | true | 0.0402 | 0.0364345, 0.0442361 | null | 60 |
| calibration/C12/cap/0.985/A-F | 0.985 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | 60 |
| calibration/C12/cap/0.985/P-F | 0.985 | true | 10000 | 143 | 317 | true | 0.0143 | 0.0120653, 0.0168237 | null | 60 |
| calibration/C12/cap/0.985/joint | 0.985 | true | 10000 | 287 | 593 | true | 0.0287 | 0.0255155, 0.0321628 | null | 60 |
| calibration/C12/cap/0.990/A-F | 0.99 | true | 10000 | 97 | 317 | true | 0.0097 | 0.00787289, 0.0118205 | null | 60 |
| calibration/C12/cap/0.990/P-F | 0.99 | true | 10000 | 94 | 317 | true | 0.0094 | 0.00760265, 0.0114911 | null | 60 |
| calibration/C12/cap/0.990/joint | 0.99 | true | 10000 | 191 | 593 | true | 0.0191 | 0.016508, 0.0219771 | null | 60 |
| calibration/C13/36/0.975/A-F | 0.975 | true | 10000 | 244 | 317 | true | 0.0244 | 0.0214652, 0.0276161 | null | null |
| calibration/C13/36/0.975/P-F | 0.975 | true | 10000 | 229 | 317 | true | 0.0229 | 0.0200578, 0.0260245 | null | null |
| calibration/C13/36/0.975/joint | 0.975 | true | 10000 | 466 | 593 | true | 0.0466 | 0.0425514, 0.0509153 | null | null |
| calibration/C13/36/0.980/A-F | 0.98 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| calibration/C13/36/0.980/P-F | 0.98 | true | 10000 | 188 | 317 | true | 0.0188 | 0.0162289, 0.0216565 | null | null |
| calibration/C13/36/0.980/joint | 0.98 | true | 10000 | 383 | 593 | true | 0.0383 | 0.0346235, 0.0422484 | null | null |
| calibration/C13/36/0.985/A-F | 0.985 | true | 10000 | 156 | 317 | true | 0.0156 | 0.013263, 0.0182249 | null | null |
| calibration/C13/36/0.985/P-F | 0.985 | true | 10000 | 134 | 317 | true | 0.0134 | 0.011239, 0.0158508 | null | null |
| calibration/C13/36/0.985/joint | 0.985 | true | 10000 | 285 | 593 | true | 0.0285 | 0.0253266, 0.0319518 | null | null |
| calibration/C13/36/0.990/A-F | 0.99 | true | 10000 | 106 | 317 | true | 0.0106 | 0.00868634, 0.0128061 | null | null |
| calibration/C13/36/0.990/P-F | 0.99 | true | 10000 | 80 | 317 | true | 0.008 | 0.00634845, 0.00994694 | null | null |
| calibration/C13/36/0.990/joint | 0.99 | true | 10000 | 183 | 593 | true | 0.0183 | 0.015764, 0.0211218 | null | null |
| calibration/C13/45/0.975/A-F | 0.975 | true | 10000 | 226 | 317 | true | 0.0226 | 0.0197767, 0.0257058 | null | null |
| calibration/C13/45/0.975/P-F | 0.975 | true | 10000 | 208 | 317 | true | 0.0208 | 0.0180931, 0.0237908 | null | null |
| calibration/C13/45/0.975/joint | 0.975 | true | 10000 | 426 | 593 | true | 0.0426 | 0.0387255, 0.0467436 | null | null |
| calibration/C13/45/0.980/A-F | 0.98 | true | 10000 | 185 | 317 | true | 0.0185 | 0.0159499, 0.0213357 | null | null |
| calibration/C13/45/0.980/P-F | 0.98 | true | 10000 | 167 | 317 | true | 0.0167 | 0.0142799, 0.0194071 | null | null |
| calibration/C13/45/0.980/joint | 0.98 | true | 10000 | 346 | 593 | true | 0.0346 | 0.031104, 0.0383703 | null | null |
| calibration/C13/45/0.985/A-F | 0.985 | true | 10000 | 139 | 317 | true | 0.0139 | 0.0116978, 0.0163916 | null | null |
| calibration/C13/45/0.985/P-F | 0.985 | true | 10000 | 124 | 317 | true | 0.0124 | 0.010324, 0.0147668 | null | null |
| calibration/C13/45/0.985/joint | 0.985 | true | 10000 | 259 | 593 | true | 0.0259 | 0.0228755, 0.0292047 | null | null |
| calibration/C13/45/0.990/A-F | 0.99 | true | 10000 | 90 | 317 | true | 0.009 | 0.0072431, 0.0110511 | null | null |
| calibration/C13/45/0.990/P-F | 0.99 | true | 10000 | 92 | 317 | true | 0.0092 | 0.00742276, 0.0112712 | null | null |
| calibration/C13/45/0.990/joint | 0.99 | true | 10000 | 179 | 593 | true | 0.0179 | 0.0153925, 0.0206936 | null | null |
| calibration/C13/60/0.975/A-F | 0.975 | true | 10000 | 230 | 317 | true | 0.023 | 0.0201515, 0.0261307 | null | null |
| calibration/C13/60/0.975/P-F | 0.975 | true | 10000 | 215 | 317 | true | 0.0215 | 0.0187472, 0.0245361 | null | null |
| calibration/C13/60/0.975/joint | 0.975 | true | 10000 | 440 | 593 | true | 0.044 | 0.0400635, 0.0482047 | null | null |
| calibration/C13/60/0.980/A-F | 0.98 | true | 10000 | 174 | 317 | true | 0.0174 | 0.0149285, 0.020158 | null | null |
| calibration/C13/60/0.980/P-F | 0.98 | true | 10000 | 175 | 317 | true | 0.0175 | 0.0150213, 0.0202651 | null | null |
| calibration/C13/60/0.980/joint | 0.98 | true | 10000 | 344 | 593 | true | 0.0344 | 0.030914, 0.0381604 | null | null |
| calibration/C13/60/0.985/A-F | 0.985 | true | 10000 | 137 | 317 | true | 0.0137 | 0.0115142, 0.0161754 | null | null |
| calibration/C13/60/0.985/P-F | 0.985 | true | 10000 | 137 | 317 | true | 0.0137 | 0.0115142, 0.0161754 | null | null |
| calibration/C13/60/0.985/joint | 0.985 | true | 10000 | 271 | 593 | true | 0.0271 | 0.0240058, 0.0304735 | null | null |
| calibration/C13/60/0.990/A-F | 0.99 | true | 10000 | 93 | 317 | true | 0.0093 | 0.00751268, 0.0113812 | null | null |
| calibration/C13/60/0.990/P-F | 0.99 | true | 10000 | 100 | 317 | true | 0.01 | 0.0081436, 0.0121495 | null | null |
| calibration/C13/60/0.990/joint | 0.99 | true | 10000 | 192 | 593 | true | 0.0192 | 0.0166011, 0.0220839 | null | null |
| calibration/C13/75/0.975/A-F | 0.975 | true | 10000 | 222 | 317 | true | 0.0222 | 0.0194021, 0.0252807 | null | null |
| calibration/C13/75/0.975/P-F | 0.975 | true | 10000 | 238 | 317 | true | 0.0238 | 0.0209018, 0.0269798 | null | null |
| calibration/C13/75/0.975/joint | 0.975 | true | 10000 | 454 | 593 | true | 0.0454 | 0.0414027, 0.0496647 | null | null |
| calibration/C13/75/0.980/A-F | 0.98 | true | 10000 | 177 | 317 | true | 0.0177 | 0.0152068, 0.0204794 | null | null |
| calibration/C13/75/0.980/P-F | 0.98 | true | 10000 | 183 | 317 | true | 0.0183 | 0.015764, 0.0211218 | null | null |
| calibration/C13/75/0.980/joint | 0.98 | true | 10000 | 359 | 593 | true | 0.0359 | 0.0323394, 0.0397341 | null | null |
| calibration/C13/75/0.985/A-F | 0.985 | true | 10000 | 137 | 317 | true | 0.0137 | 0.0115142, 0.0161754 | null | null |
| calibration/C13/75/0.985/P-F | 0.985 | true | 10000 | 147 | 317 | true | 0.0147 | 0.0124334, 0.0172553 | null | null |
| calibration/C13/75/0.985/joint | 0.985 | true | 10000 | 283 | 593 | true | 0.0283 | 0.0251378, 0.0317407 | null | null |
| calibration/C13/75/0.990/A-F | 0.99 | true | 10000 | 99 | 317 | true | 0.0099 | 0.00805331, 0.0120399 | null | null |
| calibration/C13/75/0.990/P-F | 0.99 | true | 10000 | 104 | 317 | true | 0.0104 | 0.00850524, 0.0125874 | null | null |
| calibration/C13/75/0.990/joint | 0.99 | true | 10000 | 203 | 593 | true | 0.0203 | 0.0176264, 0.0232579 | null | null |
| calibration/C13/90/0.975/A-F | 0.975 | true | 10000 | 233 | 317 | true | 0.0233 | 0.0204328, 0.0264493 | null | null |
| calibration/C13/90/0.975/P-F | 0.975 | true | 10000 | 220 | 317 | true | 0.022 | 0.0192149, 0.025068 | null | null |
| calibration/C13/90/0.975/joint | 0.975 | true | 10000 | 446 | 593 | true | 0.0446 | 0.0406373, 0.0488305 | null | null |
| calibration/C13/90/0.980/A-F | 0.98 | true | 10000 | 188 | 317 | true | 0.0188 | 0.0162289, 0.0216565 | null | null |
| calibration/C13/90/0.980/P-F | 0.98 | true | 10000 | 174 | 317 | true | 0.0174 | 0.0149285, 0.020158 | null | null |
| calibration/C13/90/0.980/joint | 0.98 | true | 10000 | 357 | 593 | true | 0.0357 | 0.0321492, 0.0395244 | null | null |
| calibration/C13/90/0.985/A-F | 0.985 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| calibration/C13/90/0.985/P-F | 0.985 | true | 10000 | 131 | 317 | true | 0.0131 | 0.0109641, 0.0155259 | null | null |
| calibration/C13/90/0.985/joint | 0.985 | true | 10000 | 280 | 593 | true | 0.028 | 0.0248547, 0.0314241 | null | null |
| calibration/C13/90/0.990/A-F | 0.99 | true | 10000 | 100 | 317 | true | 0.01 | 0.0081436, 0.0121495 | null | null |
| calibration/C13/90/0.990/P-F | 0.99 | true | 10000 | 93 | 317 | true | 0.0093 | 0.00751268, 0.0113812 | null | null |
| calibration/C13/90/0.990/joint | 0.99 | true | 10000 | 192 | 593 | true | 0.0192 | 0.0166011, 0.0220839 | null | null |
| calibration/C13/99/0.975/A-F | 0.975 | true | 10000 | 250 | 317 | true | 0.025 | 0.022029, 0.0282519 | null | null |
| calibration/C13/99/0.975/P-F | 0.975 | true | 10000 | 230 | 317 | true | 0.023 | 0.0201515, 0.0261307 | null | null |
| calibration/C13/99/0.975/joint | 0.975 | true | 10000 | 476 | 593 | true | 0.0476 | 0.0435092, 0.0519568 | null | null |
| calibration/C13/99/0.980/A-F | 0.98 | true | 10000 | 194 | 317 | true | 0.0194 | 0.0167874, 0.0222975 | null | null |
| calibration/C13/99/0.980/P-F | 0.98 | true | 10000 | 178 | 317 | true | 0.0178 | 0.0152996, 0.0205865 | null | null |
| calibration/C13/99/0.980/joint | 0.98 | true | 10000 | 369 | 593 | true | 0.0369 | 0.0332906, 0.0407822 | null | null |
| calibration/C13/99/0.985/A-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| calibration/C13/99/0.985/P-F | 0.985 | true | 10000 | 137 | 317 | true | 0.0137 | 0.0115142, 0.0161754 | null | null |
| calibration/C13/99/0.985/joint | 0.985 | true | 10000 | 287 | 593 | true | 0.0287 | 0.0255155, 0.0321628 | null | null |
| calibration/C13/99/0.990/A-F | 0.99 | true | 10000 | 93 | 317 | true | 0.0093 | 0.00751268, 0.0113812 | null | null |
| calibration/C13/99/0.990/P-F | 0.99 | true | 10000 | 97 | 317 | true | 0.0097 | 0.00787289, 0.0118205 | null | null |
| calibration/C13/99/0.990/joint | 0.99 | true | 10000 | 190 | 593 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| calibration/C13/cap/0.975/A-F | 0.975 | true | 10000 | 238 | 317 | true | 0.0238 | 0.0209018, 0.0269798 | null | 60 |
| calibration/C13/cap/0.975/P-F | 0.975 | true | 10000 | 236 | 317 | true | 0.0236 | 0.0207142, 0.0267676 | null | 60 |
| calibration/C13/cap/0.975/joint | 0.975 | true | 10000 | 467 | 593 | true | 0.0467 | 0.0426471, 0.0510194 | null | 60 |
| calibration/C13/cap/0.980/A-F | 0.98 | true | 10000 | 182 | 317 | true | 0.0182 | 0.0156711, 0.0210148 | null | 60 |
| calibration/C13/cap/0.980/P-F | 0.98 | true | 10000 | 175 | 317 | true | 0.0175 | 0.0150213, 0.0202651 | null | 60 |
| calibration/C13/cap/0.980/joint | 0.98 | true | 10000 | 355 | 593 | true | 0.0355 | 0.0319591, 0.0393146 | null | 60 |
| calibration/C13/cap/0.985/A-F | 0.985 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | 60 |
| calibration/C13/cap/0.985/P-F | 0.985 | true | 10000 | 137 | 317 | true | 0.0137 | 0.0115142, 0.0161754 | null | 60 |
| calibration/C13/cap/0.985/joint | 0.985 | true | 10000 | 287 | 593 | true | 0.0287 | 0.0255155, 0.0321628 | null | 60 |
| calibration/C13/cap/0.990/A-F | 0.99 | true | 10000 | 95 | 317 | true | 0.0095 | 0.00769267, 0.011601 | null | 60 |
| calibration/C13/cap/0.990/P-F | 0.99 | true | 10000 | 96 | 317 | true | 0.0096 | 0.00778275, 0.0117108 | null | 60 |
| calibration/C13/cap/0.990/joint | 0.99 | true | 10000 | 191 | 593 | true | 0.0191 | 0.016508, 0.0219771 | null | 60 |
| calibration/C14/36/0.975/A-F | 0.975 | true | 10000 | 248 | 317 | true | 0.0248 | 0.021841, 0.02804 | null | null |
| calibration/C14/36/0.975/P-F | 0.975 | true | 10000 | 246 | 317 | true | 0.0246 | 0.021653, 0.0278281 | null | null |
| calibration/C14/36/0.975/joint | 0.975 | true | 10000 | 490 | 593 | true | 0.049 | 0.0448509, 0.0534142 | null | null |
| calibration/C14/36/0.980/A-F | 0.98 | true | 10000 | 197 | 317 | true | 0.0197 | 0.0170669, 0.0226178 | null | null |
| calibration/C14/36/0.980/P-F | 0.98 | true | 10000 | 197 | 317 | true | 0.0197 | 0.0170669, 0.0226178 | null | null |
| calibration/C14/36/0.980/joint | 0.98 | true | 10000 | 394 | 593 | true | 0.0394 | 0.0356717, 0.0433995 | null | null |
| calibration/C14/36/0.985/A-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| calibration/C14/36/0.985/P-F | 0.985 | true | 10000 | 149 | 317 | true | 0.0149 | 0.0126175, 0.017471 | null | null |
| calibration/C14/36/0.985/joint | 0.985 | true | 10000 | 301 | 593 | true | 0.0301 | 0.0268386, 0.0336387 | null | null |
| calibration/C14/36/0.990/A-F | 0.99 | true | 10000 | 108 | 317 | true | 0.0108 | 0.00886763, 0.0130246 | null | null |
| calibration/C14/36/0.990/P-F | 0.99 | true | 10000 | 101 | 317 | true | 0.0101 | 0.00823393, 0.0122591 | null | null |
| calibration/C14/36/0.990/joint | 0.99 | true | 10000 | 209 | 593 | true | 0.0209 | 0.0181865, 0.0238973 | null | null |
| calibration/C14/45/0.975/A-F | 0.975 | true | 10000 | 251 | 317 | true | 0.0251 | 0.022123, 0.0283578 | null | null |
| calibration/C14/45/0.975/P-F | 0.975 | true | 10000 | 256 | 317 | true | 0.0256 | 0.0225932, 0.0288872 | null | null |
| calibration/C14/45/0.975/joint | 0.975 | true | 10000 | 500 | 593 | true | 0.05 | 0.0458099, 0.0544545 | null | null |
| calibration/C14/45/0.980/A-F | 0.98 | true | 10000 | 209 | 317 | true | 0.0209 | 0.0181865, 0.0238973 | null | null |
| calibration/C14/45/0.980/P-F | 0.98 | true | 10000 | 205 | 317 | true | 0.0205 | 0.017813, 0.0234711 | null | null |
| calibration/C14/45/0.980/joint | 0.98 | true | 10000 | 410 | 593 | true | 0.041 | 0.0371978, 0.0450724 | null | null |
| calibration/C14/45/0.985/A-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| calibration/C14/45/0.985/P-F | 0.985 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | null |
| calibration/C14/45/0.985/joint | 0.985 | true | 10000 | 303 | 593 | true | 0.0303 | 0.0270278, 0.0338494 | null | null |
| calibration/C14/45/0.990/A-F | 0.99 | true | 10000 | 121 | 317 | true | 0.0121 | 0.0100501, 0.0144409 | null | null |
| calibration/C14/45/0.990/P-F | 0.99 | true | 10000 | 101 | 317 | true | 0.0101 | 0.00823393, 0.0122591 | null | null |
| calibration/C14/45/0.990/joint | 0.99 | true | 10000 | 221 | 593 | true | 0.0221 | 0.0193085, 0.0251744 | null | null |
| calibration/C14/60/0.975/A-F | 0.975 | true | 10000 | 267 | 317 | true | 0.0267 | 0.0236289, 0.0300508 | null | null |
| calibration/C14/60/0.975/P-F | 0.975 | true | 10000 | 254 | 317 | true | 0.0254 | 0.0224051, 0.0286754 | null | null |
| calibration/C14/60/0.975/joint | 0.975 | true | 10000 | 513 | 593 | true | 0.0513 | 0.0470573, 0.0558063 | null | null |
| calibration/C14/60/0.980/A-F | 0.98 | true | 10000 | 218 | 317 | true | 0.0218 | 0.0190278, 0.0248553 | null | null |
| calibration/C14/60/0.980/P-F | 0.98 | true | 10000 | 204 | 317 | true | 0.0204 | 0.0177197, 0.0233645 | null | null |
| calibration/C14/60/0.980/joint | 0.98 | true | 10000 | 417 | 593 | true | 0.0417 | 0.037866, 0.0458037 | null | null |
| calibration/C14/60/0.985/A-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| calibration/C14/60/0.985/P-F | 0.985 | true | 10000 | 168 | 317 | true | 0.0168 | 0.0143725, 0.0195144 | null | null |
| calibration/C14/60/0.985/joint | 0.985 | true | 10000 | 325 | 593 | true | 0.0325 | 0.0291112, 0.0361645 | null | null |
| calibration/C14/60/0.990/A-F | 0.99 | true | 10000 | 119 | 317 | true | 0.0119 | 0.00986777, 0.0142234 | null | null |
| calibration/C14/60/0.990/P-F | 0.99 | true | 10000 | 100 | 317 | true | 0.01 | 0.0081436, 0.0121495 | null | null |
| calibration/C14/60/0.990/joint | 0.99 | true | 10000 | 217 | 593 | true | 0.0217 | 0.0189343, 0.0247489 | null | null |
| calibration/C14/75/0.975/A-F | 0.975 | true | 10000 | 280 | 317 | true | 0.028 | 0.0248547, 0.0314241 | null | null |
| calibration/C14/75/0.975/P-F | 0.975 | true | 10000 | 245 | 317 | true | 0.0245 | 0.0215591, 0.0277221 | null | null |
| calibration/C14/75/0.975/joint | 0.975 | true | 10000 | 519 | 593 | true | 0.0519 | 0.0476333, 0.0564299 | null | null |
| calibration/C14/75/0.980/A-F | 0.98 | true | 10000 | 225 | 317 | true | 0.0225 | 0.019683, 0.0255996 | null | null |
| calibration/C14/75/0.980/P-F | 0.98 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| calibration/C14/75/0.980/joint | 0.98 | true | 10000 | 411 | 593 | true | 0.0411 | 0.0372932, 0.0451769 | null | null |
| calibration/C14/75/0.985/A-F | 0.985 | true | 10000 | 165 | 317 | true | 0.0165 | 0.0140948, 0.0191924 | null | null |
| calibration/C14/75/0.985/P-F | 0.985 | true | 10000 | 145 | 317 | true | 0.0145 | 0.0122493, 0.0170396 | null | null |
| calibration/C14/75/0.985/joint | 0.985 | true | 10000 | 306 | 593 | true | 0.0306 | 0.0273116, 0.0341654 | null | null |
| calibration/C14/75/0.990/A-F | 0.99 | true | 10000 | 109 | 317 | true | 0.0109 | 0.00895834, 0.0131338 | null | null |
| calibration/C14/75/0.990/P-F | 0.99 | true | 10000 | 96 | 317 | true | 0.0096 | 0.00778275, 0.0117108 | null | null |
| calibration/C14/75/0.990/joint | 0.99 | true | 10000 | 202 | 593 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| calibration/C14/90/0.975/A-F | 0.975 | true | 10000 | 269 | 317 | true | 0.0269 | 0.0238173, 0.0302622 | null | null |
| calibration/C14/90/0.975/P-F | 0.975 | true | 10000 | 267 | 317 | true | 0.0267 | 0.0236289, 0.0300508 | null | null |
| calibration/C14/90/0.975/joint | 0.975 | true | 10000 | 526 | 593 | true | 0.0526 | 0.0483055, 0.0571573 | null | null |
| calibration/C14/90/0.980/A-F | 0.98 | true | 10000 | 216 | 317 | true | 0.0216 | 0.0188407, 0.0246425 | null | null |
| calibration/C14/90/0.980/P-F | 0.98 | true | 10000 | 209 | 317 | true | 0.0209 | 0.0181865, 0.0238973 | null | null |
| calibration/C14/90/0.980/joint | 0.98 | true | 10000 | 417 | 593 | true | 0.0417 | 0.037866, 0.0458037 | null | null |
| calibration/C14/90/0.985/A-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| calibration/C14/90/0.985/P-F | 0.985 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | null |
| calibration/C14/90/0.985/joint | 0.985 | true | 10000 | 313 | 593 | true | 0.0313 | 0.0279742, 0.0349023 | null | null |
| calibration/C14/90/0.990/A-F | 0.99 | true | 10000 | 110 | 317 | true | 0.011 | 0.0090491, 0.013243 | null | null |
| calibration/C14/90/0.990/P-F | 0.99 | true | 10000 | 103 | 317 | true | 0.0103 | 0.00841475, 0.012478 | null | null |
| calibration/C14/90/0.990/joint | 0.99 | true | 10000 | 212 | 593 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| calibration/C14/99/0.975/A-F | 0.975 | true | 10000 | 271 | 317 | true | 0.0271 | 0.0240058, 0.0304735 | null | null |
| calibration/C14/99/0.975/P-F | 0.975 | true | 10000 | 265 | 317 | true | 0.0265 | 0.0234405, 0.0298393 | null | null |
| calibration/C14/99/0.975/joint | 0.975 | true | 10000 | 527 | 593 | true | 0.0527 | 0.0484016, 0.0572612 | null | null |
| calibration/C14/99/0.980/A-F | 0.98 | true | 10000 | 225 | 317 | true | 0.0225 | 0.019683, 0.0255996 | null | null |
| calibration/C14/99/0.980/P-F | 0.98 | true | 10000 | 214 | 317 | true | 0.0214 | 0.0186537, 0.0244297 | null | null |
| calibration/C14/99/0.980/joint | 0.98 | true | 10000 | 433 | 593 | true | 0.0433 | 0.0393944, 0.0474743 | null | null |
| calibration/C14/99/0.985/A-F | 0.985 | true | 10000 | 163 | 317 | true | 0.0163 | 0.0139098, 0.0189775 | null | null |
| calibration/C14/99/0.985/P-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| calibration/C14/99/0.985/joint | 0.985 | true | 10000 | 319 | 593 | true | 0.0319 | 0.0285426, 0.0355335 | null | null |
| calibration/C14/99/0.990/A-F | 0.99 | true | 10000 | 109 | 317 | true | 0.0109 | 0.00895834, 0.0131338 | null | null |
| calibration/C14/99/0.990/P-F | 0.99 | true | 10000 | 105 | 317 | true | 0.0105 | 0.00859577, 0.0126968 | null | null |
| calibration/C14/99/0.990/joint | 0.99 | true | 10000 | 213 | 593 | true | 0.0213 | 0.0185602, 0.0243233 | null | null |
| calibration/C14/cap/0.975/A-F | 0.975 | true | 10000 | 279 | 317 | true | 0.0279 | 0.0247603, 0.0313185 | null | 60 |
| calibration/C14/cap/0.975/P-F | 0.975 | true | 10000 | 247 | 317 | true | 0.0247 | 0.021747, 0.027934 | null | 60 |
| calibration/C14/cap/0.975/joint | 0.975 | true | 10000 | 515 | 593 | true | 0.0515 | 0.0472493, 0.0560142 | null | 60 |
| calibration/C14/cap/0.980/A-F | 0.98 | true | 10000 | 208 | 317 | true | 0.0208 | 0.0180931, 0.0237908 | null | 60 |
| calibration/C14/cap/0.980/P-F | 0.98 | true | 10000 | 213 | 317 | true | 0.0213 | 0.0185602, 0.0243233 | null | 60 |
| calibration/C14/cap/0.980/joint | 0.98 | true | 10000 | 413 | 593 | true | 0.0413 | 0.0374841, 0.0453858 | null | 60 |
| calibration/C14/cap/0.985/A-F | 0.985 | true | 10000 | 156 | 317 | true | 0.0156 | 0.013263, 0.0182249 | null | 60 |
| calibration/C14/cap/0.985/P-F | 0.985 | true | 10000 | 167 | 317 | true | 0.0167 | 0.0142799, 0.0194071 | null | 60 |
| calibration/C14/cap/0.985/joint | 0.985 | true | 10000 | 319 | 593 | true | 0.0319 | 0.0285426, 0.0355335 | null | 60 |
| calibration/C14/cap/0.990/A-F | 0.99 | true | 10000 | 113 | 317 | true | 0.0113 | 0.00932162, 0.0135702 | null | 60 |
| calibration/C14/cap/0.990/P-F | 0.99 | true | 10000 | 112 | 317 | true | 0.0112 | 0.00923074, 0.0134611 | null | 60 |
| calibration/C14/cap/0.990/joint | 0.99 | true | 10000 | 222 | 593 | true | 0.0222 | 0.0194021, 0.0252807 | null | 60 |
| calibration/C15/36/0.975/A-F | 0.975 | true | 10000 | 242 | 317 | true | 0.0242 | 0.0212773, 0.0274041 | null | null |
| calibration/C15/36/0.975/P-F | 0.975 | true | 10000 | 260 | 317 | true | 0.026 | 0.0229697, 0.0293105 | null | null |
| calibration/C15/36/0.975/joint | 0.975 | true | 10000 | 496 | 593 | true | 0.0496 | 0.0454263, 0.0540384 | null | null |
| calibration/C15/36/0.980/A-F | 0.98 | true | 10000 | 195 | 317 | true | 0.0195 | 0.0168805, 0.0224043 | null | null |
| calibration/C15/36/0.980/P-F | 0.98 | true | 10000 | 210 | 317 | true | 0.021 | 0.0182799, 0.0240038 | null | null |
| calibration/C15/36/0.980/joint | 0.98 | true | 10000 | 400 | 593 | true | 0.04 | 0.0362438, 0.044027 | null | null |
| calibration/C15/36/0.985/A-F | 0.985 | true | 10000 | 141 | 317 | true | 0.0141 | 0.0118815, 0.0166077 | null | null |
| calibration/C15/36/0.985/P-F | 0.985 | true | 10000 | 155 | 317 | true | 0.0155 | 0.0131707, 0.0181173 | null | null |
| calibration/C15/36/0.985/joint | 0.985 | true | 10000 | 292 | 593 | true | 0.0292 | 0.0259878, 0.0326901 | null | null |
| calibration/C15/36/0.990/A-F | 0.99 | true | 10000 | 94 | 317 | true | 0.0094 | 0.00760265, 0.0114911 | null | null |
| calibration/C15/36/0.990/P-F | 0.99 | true | 10000 | 104 | 317 | true | 0.0104 | 0.00850524, 0.0125874 | null | null |
| calibration/C15/36/0.990/joint | 0.99 | true | 10000 | 196 | 593 | true | 0.0196 | 0.0169737, 0.0225111 | null | null |
| calibration/C15/45/0.975/A-F | 0.975 | true | 10000 | 256 | 317 | true | 0.0256 | 0.0225932, 0.0288872 | null | null |
| calibration/C15/45/0.975/P-F | 0.975 | true | 10000 | 247 | 317 | true | 0.0247 | 0.021747, 0.027934 | null | null |
| calibration/C15/45/0.975/joint | 0.975 | true | 10000 | 498 | 593 | true | 0.0498 | 0.0456181, 0.0542465 | null | null |
| calibration/C15/45/0.980/A-F | 0.98 | true | 10000 | 208 | 317 | true | 0.0208 | 0.0180931, 0.0237908 | null | null |
| calibration/C15/45/0.980/P-F | 0.98 | true | 10000 | 193 | 317 | true | 0.0193 | 0.0166942, 0.0221908 | null | null |
| calibration/C15/45/0.980/joint | 0.98 | true | 10000 | 398 | 593 | true | 0.0398 | 0.0360531, 0.0438179 | null | null |
| calibration/C15/45/0.985/A-F | 0.985 | true | 10000 | 165 | 317 | true | 0.0165 | 0.0140948, 0.0191924 | null | null |
| calibration/C15/45/0.985/P-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| calibration/C15/45/0.985/joint | 0.985 | true | 10000 | 316 | 593 | true | 0.0316 | 0.0282584, 0.0352179 | null | null |
| calibration/C15/45/0.990/A-F | 0.99 | true | 10000 | 111 | 317 | true | 0.0111 | 0.00913989, 0.0133521 | null | null |
| calibration/C15/45/0.990/P-F | 0.99 | true | 10000 | 105 | 317 | true | 0.0105 | 0.00859577, 0.0126968 | null | null |
| calibration/C15/45/0.990/joint | 0.99 | true | 10000 | 216 | 593 | true | 0.0216 | 0.0188407, 0.0246425 | null | null |
| calibration/C15/60/0.975/A-F | 0.975 | true | 10000 | 249 | 317 | true | 0.0249 | 0.021935, 0.0281459 | null | null |
| calibration/C15/60/0.975/P-F | 0.975 | true | 10000 | 238 | 317 | true | 0.0238 | 0.0209018, 0.0269798 | null | null |
| calibration/C15/60/0.975/joint | 0.975 | true | 10000 | 480 | 593 | true | 0.048 | 0.0438924, 0.0523733 | null | null |
| calibration/C15/60/0.980/A-F | 0.98 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| calibration/C15/60/0.980/P-F | 0.98 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| calibration/C15/60/0.980/joint | 0.98 | true | 10000 | 388 | 593 | true | 0.0388 | 0.0350998, 0.0427718 | null | null |
| calibration/C15/60/0.985/A-F | 0.985 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | null |
| calibration/C15/60/0.985/P-F | 0.985 | true | 10000 | 145 | 317 | true | 0.0145 | 0.0122493, 0.0170396 | null | null |
| calibration/C15/60/0.985/joint | 0.985 | true | 10000 | 292 | 593 | true | 0.0292 | 0.0259878, 0.0326901 | null | null |
| calibration/C15/60/0.990/A-F | 0.99 | true | 10000 | 100 | 317 | true | 0.01 | 0.0081436, 0.0121495 | null | null |
| calibration/C15/60/0.990/P-F | 0.99 | true | 10000 | 104 | 317 | true | 0.0104 | 0.00850524, 0.0125874 | null | null |
| calibration/C15/60/0.990/joint | 0.99 | true | 10000 | 200 | 593 | true | 0.02 | 0.0173465, 0.0229379 | null | null |
| calibration/C15/75/0.975/A-F | 0.975 | true | 10000 | 247 | 317 | true | 0.0247 | 0.021747, 0.027934 | null | null |
| calibration/C15/75/0.975/P-F | 0.975 | true | 10000 | 271 | 317 | true | 0.0271 | 0.0240058, 0.0304735 | null | null |
| calibration/C15/75/0.975/joint | 0.975 | true | 10000 | 511 | 593 | true | 0.0511 | 0.0468654, 0.0555984 | null | null |
| calibration/C15/75/0.980/A-F | 0.98 | true | 10000 | 205 | 317 | true | 0.0205 | 0.017813, 0.0234711 | null | null |
| calibration/C15/75/0.980/P-F | 0.98 | true | 10000 | 208 | 317 | true | 0.0208 | 0.0180931, 0.0237908 | null | null |
| calibration/C15/75/0.980/joint | 0.98 | true | 10000 | 408 | 593 | true | 0.0408 | 0.0370069, 0.0448633 | null | null |
| calibration/C15/75/0.985/A-F | 0.985 | true | 10000 | 162 | 317 | true | 0.0162 | 0.0138173, 0.0188701 | null | null |
| calibration/C15/75/0.985/P-F | 0.985 | true | 10000 | 153 | 317 | true | 0.0153 | 0.0129862, 0.0179019 | null | null |
| calibration/C15/75/0.985/joint | 0.985 | true | 10000 | 311 | 593 | true | 0.0311 | 0.0277849, 0.0346918 | null | null |
| calibration/C15/75/0.990/A-F | 0.99 | true | 10000 | 105 | 317 | true | 0.0105 | 0.00859577, 0.0126968 | null | null |
| calibration/C15/75/0.990/P-F | 0.99 | true | 10000 | 109 | 317 | true | 0.0109 | 0.00895834, 0.0131338 | null | null |
| calibration/C15/75/0.990/joint | 0.99 | true | 10000 | 212 | 593 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| calibration/C15/90/0.975/A-F | 0.975 | true | 10000 | 240 | 317 | true | 0.024 | 0.0210896, 0.027192 | null | null |
| calibration/C15/90/0.975/P-F | 0.975 | true | 10000 | 253 | 317 | true | 0.0253 | 0.022311, 0.0285696 | null | null |
| calibration/C15/90/0.975/joint | 0.975 | true | 10000 | 487 | 593 | true | 0.0487 | 0.0445633, 0.0531019 | null | null |
| calibration/C15/90/0.980/A-F | 0.98 | true | 10000 | 196 | 317 | true | 0.0196 | 0.0169737, 0.0225111 | null | null |
| calibration/C15/90/0.980/P-F | 0.98 | true | 10000 | 198 | 317 | true | 0.0198 | 0.0171601, 0.0227245 | null | null |
| calibration/C15/90/0.980/joint | 0.98 | true | 10000 | 390 | 593 | true | 0.039 | 0.0352904, 0.042981 | null | null |
| calibration/C15/90/0.985/A-F | 0.985 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| calibration/C15/90/0.985/P-F | 0.985 | true | 10000 | 138 | 317 | true | 0.0138 | 0.011606, 0.0162835 | null | null |
| calibration/C15/90/0.985/joint | 0.985 | true | 10000 | 285 | 593 | true | 0.0285 | 0.0253266, 0.0319518 | null | null |
| calibration/C15/90/0.990/A-F | 0.99 | true | 10000 | 96 | 317 | true | 0.0096 | 0.00778275, 0.0117108 | null | null |
| calibration/C15/90/0.990/P-F | 0.99 | true | 10000 | 92 | 317 | true | 0.0092 | 0.00742276, 0.0112712 | null | null |
| calibration/C15/90/0.990/joint | 0.99 | true | 10000 | 185 | 593 | true | 0.0185 | 0.0159499, 0.0213357 | null | null |
| calibration/C15/99/0.975/A-F | 0.975 | true | 10000 | 251 | 317 | true | 0.0251 | 0.022123, 0.0283578 | null | null |
| calibration/C15/99/0.975/P-F | 0.975 | true | 10000 | 252 | 317 | true | 0.0252 | 0.022217, 0.0284637 | null | null |
| calibration/C15/99/0.975/joint | 0.975 | true | 10000 | 498 | 593 | true | 0.0498 | 0.0456181, 0.0542465 | null | null |
| calibration/C15/99/0.980/A-F | 0.98 | true | 10000 | 192 | 317 | true | 0.0192 | 0.0166011, 0.0220839 | null | null |
| calibration/C15/99/0.980/P-F | 0.98 | true | 10000 | 205 | 317 | true | 0.0205 | 0.017813, 0.0234711 | null | null |
| calibration/C15/99/0.980/joint | 0.98 | true | 10000 | 394 | 593 | true | 0.0394 | 0.0356717, 0.0433995 | null | null |
| calibration/C15/99/0.985/A-F | 0.985 | true | 10000 | 154 | 317 | true | 0.0154 | 0.0130785, 0.0180096 | null | null |
| calibration/C15/99/0.985/P-F | 0.985 | true | 10000 | 158 | 317 | true | 0.0158 | 0.0134477, 0.0184401 | null | null |
| calibration/C15/99/0.985/joint | 0.985 | true | 10000 | 310 | 593 | true | 0.031 | 0.0276902, 0.0345865 | null | null |
| calibration/C15/99/0.990/A-F | 0.99 | true | 10000 | 104 | 317 | true | 0.0104 | 0.00850524, 0.0125874 | null | null |
| calibration/C15/99/0.990/P-F | 0.99 | true | 10000 | 118 | 317 | true | 0.0118 | 0.00977665, 0.0141146 | null | null |
| calibration/C15/99/0.990/joint | 0.99 | true | 10000 | 221 | 593 | true | 0.0221 | 0.0193085, 0.0251744 | null | null |
| calibration/C15/cap/0.975/A-F | 0.975 | true | 10000 | 244 | 317 | true | 0.0244 | 0.0214652, 0.0276161 | null | 60 |
| calibration/C15/cap/0.975/P-F | 0.975 | true | 10000 | 261 | 317 | true | 0.0261 | 0.0230638, 0.0294162 | null | 60 |
| calibration/C15/cap/0.975/joint | 0.975 | true | 10000 | 498 | 593 | true | 0.0498 | 0.0456181, 0.0542465 | null | 60 |
| calibration/C15/cap/0.980/A-F | 0.98 | true | 10000 | 182 | 317 | true | 0.0182 | 0.0156711, 0.0210148 | null | 60 |
| calibration/C15/cap/0.980/P-F | 0.98 | true | 10000 | 205 | 317 | true | 0.0205 | 0.017813, 0.0234711 | null | 60 |
| calibration/C15/cap/0.980/joint | 0.98 | true | 10000 | 381 | 593 | true | 0.0381 | 0.034433, 0.0420391 | null | 60 |
| calibration/C15/cap/0.985/A-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | 60 |
| calibration/C15/cap/0.985/P-F | 0.985 | true | 10000 | 170 | 317 | true | 0.017 | 0.0145578, 0.019729 | null | 60 |
| calibration/C15/cap/0.985/joint | 0.985 | true | 10000 | 319 | 593 | true | 0.0319 | 0.0285426, 0.0355335 | null | 60 |
| calibration/C15/cap/0.990/A-F | 0.99 | true | 10000 | 90 | 317 | true | 0.009 | 0.0072431, 0.0110511 | null | 60 |
| calibration/C15/cap/0.990/P-F | 0.99 | true | 10000 | 126 | 317 | true | 0.0126 | 0.0105067, 0.0149838 | null | 60 |
| calibration/C15/cap/0.990/joint | 0.99 | true | 10000 | 213 | 593 | true | 0.0213 | 0.0185602, 0.0243233 | null | 60 |
| calibration/C16/36/0.975/A-F | 0.975 | true | 10000 | 249 | 317 | true | 0.0249 | 0.021935, 0.0281459 | null | null |
| calibration/C16/36/0.975/P-F | 0.975 | true | 10000 | 283 | 317 | true | 0.0283 | 0.0251378, 0.0317407 | null | null |
| calibration/C16/36/0.975/joint | 0.975 | true | 10000 | 522 | 593 | true | 0.0522 | 0.0479214, 0.0567417 | null | null |
| calibration/C16/36/0.980/A-F | 0.98 | true | 10000 | 199 | 317 | true | 0.0199 | 0.0172533, 0.0228312 | null | null |
| calibration/C16/36/0.980/P-F | 0.98 | true | 10000 | 235 | 317 | true | 0.0235 | 0.0206204, 0.0266615 | null | null |
| calibration/C16/36/0.980/joint | 0.98 | true | 10000 | 427 | 593 | true | 0.0427 | 0.0388211, 0.046848 | null | null |
| calibration/C16/36/0.985/A-F | 0.985 | true | 10000 | 154 | 317 | true | 0.0154 | 0.0130785, 0.0180096 | null | null |
| calibration/C16/36/0.985/P-F | 0.985 | true | 10000 | 166 | 317 | true | 0.0166 | 0.0141874, 0.0192998 | null | null |
| calibration/C16/36/0.985/joint | 0.985 | true | 10000 | 315 | 593 | true | 0.0315 | 0.0281637, 0.0351127 | null | null |
| calibration/C16/36/0.990/A-F | 0.99 | true | 10000 | 105 | 317 | true | 0.0105 | 0.00859577, 0.0126968 | null | null |
| calibration/C16/36/0.990/P-F | 0.99 | true | 10000 | 117 | 317 | true | 0.0117 | 0.00968556, 0.0140058 | null | null |
| calibration/C16/36/0.990/joint | 0.99 | true | 10000 | 219 | 593 | true | 0.0219 | 0.0191214, 0.0249617 | null | null |
| calibration/C16/45/0.975/A-F | 0.975 | true | 10000 | 243 | 317 | true | 0.0243 | 0.0213712, 0.0275101 | null | null |
| calibration/C16/45/0.975/P-F | 0.975 | true | 10000 | 274 | 317 | true | 0.0274 | 0.0242887, 0.0307905 | null | null |
| calibration/C16/45/0.975/joint | 0.975 | true | 10000 | 505 | 593 | true | 0.0505 | 0.0462896, 0.0549745 | null | null |
| calibration/C16/45/0.980/A-F | 0.98 | true | 10000 | 198 | 317 | true | 0.0198 | 0.0171601, 0.0227245 | null | null |
| calibration/C16/45/0.980/P-F | 0.98 | true | 10000 | 217 | 317 | true | 0.0217 | 0.0189343, 0.0247489 | null | null |
| calibration/C16/45/0.980/joint | 0.98 | true | 10000 | 408 | 593 | true | 0.0408 | 0.0370069, 0.0448633 | null | null |
| calibration/C16/45/0.985/A-F | 0.985 | true | 10000 | 142 | 317 | true | 0.0142 | 0.0119734, 0.0167157 | null | null |
| calibration/C16/45/0.985/P-F | 0.985 | true | 10000 | 169 | 317 | true | 0.0169 | 0.0144651, 0.0196217 | null | null |
| calibration/C16/45/0.985/joint | 0.985 | true | 10000 | 306 | 593 | true | 0.0306 | 0.0273116, 0.0341654 | null | null |
| calibration/C16/45/0.990/A-F | 0.99 | true | 10000 | 94 | 317 | true | 0.0094 | 0.00760265, 0.0114911 | null | null |
| calibration/C16/45/0.990/P-F | 0.99 | true | 10000 | 118 | 317 | true | 0.0118 | 0.00977665, 0.0141146 | null | null |
| calibration/C16/45/0.990/joint | 0.99 | true | 10000 | 208 | 593 | true | 0.0208 | 0.0180931, 0.0237908 | null | null |
| calibration/C16/60/0.975/A-F | 0.975 | true | 10000 | 237 | 317 | true | 0.0237 | 0.020808, 0.0268738 | null | null |
| calibration/C16/60/0.975/P-F | 0.975 | true | 10000 | 254 | 317 | true | 0.0254 | 0.0224051, 0.0286754 | null | null |
| calibration/C16/60/0.975/joint | 0.975 | true | 10000 | 478 | 593 | true | 0.0478 | 0.0437008, 0.0521651 | null | null |
| calibration/C16/60/0.980/A-F | 0.98 | true | 10000 | 178 | 317 | true | 0.0178 | 0.0152996, 0.0205865 | null | null |
| calibration/C16/60/0.980/P-F | 0.98 | true | 10000 | 208 | 317 | true | 0.0208 | 0.0180931, 0.0237908 | null | null |
| calibration/C16/60/0.980/joint | 0.98 | true | 10000 | 380 | 593 | true | 0.038 | 0.0343377, 0.0419344 | null | null |
| calibration/C16/60/0.985/A-F | 0.985 | true | 10000 | 135 | 317 | true | 0.0135 | 0.0113307, 0.015959 | null | null |
| calibration/C16/60/0.985/P-F | 0.985 | true | 10000 | 159 | 317 | true | 0.0159 | 0.0135401, 0.0185476 | null | null |
| calibration/C16/60/0.985/joint | 0.985 | true | 10000 | 289 | 593 | true | 0.0289 | 0.0257044, 0.0323738 | null | null |
| calibration/C16/60/0.990/A-F | 0.99 | true | 10000 | 90 | 317 | true | 0.009 | 0.0072431, 0.0110511 | null | null |
| calibration/C16/60/0.990/P-F | 0.99 | true | 10000 | 108 | 317 | true | 0.0108 | 0.00886763, 0.0130246 | null | null |
| calibration/C16/60/0.990/joint | 0.99 | true | 10000 | 194 | 593 | true | 0.0194 | 0.0167874, 0.0222975 | null | null |
| calibration/C16/75/0.975/A-F | 0.975 | true | 10000 | 264 | 317 | true | 0.0264 | 0.0233463, 0.0297336 | null | null |
| calibration/C16/75/0.975/P-F | 0.975 | true | 10000 | 256 | 317 | true | 0.0256 | 0.0225932, 0.0288872 | null | null |
| calibration/C16/75/0.975/joint | 0.975 | true | 10000 | 513 | 593 | true | 0.0513 | 0.0470573, 0.0558063 | null | null |
| calibration/C16/75/0.980/A-F | 0.98 | true | 10000 | 213 | 317 | true | 0.0213 | 0.0185602, 0.0243233 | null | null |
| calibration/C16/75/0.980/P-F | 0.98 | true | 10000 | 209 | 317 | true | 0.0209 | 0.0181865, 0.0238973 | null | null |
| calibration/C16/75/0.980/joint | 0.98 | true | 10000 | 416 | 593 | true | 0.0416 | 0.0377705, 0.0456993 | null | null |
| calibration/C16/75/0.985/A-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| calibration/C16/75/0.985/P-F | 0.985 | true | 10000 | 162 | 317 | true | 0.0162 | 0.0138173, 0.0188701 | null | null |
| calibration/C16/75/0.985/joint | 0.985 | true | 10000 | 317 | 593 | true | 0.0317 | 0.0283531, 0.0353231 | null | null |
| calibration/C16/75/0.990/A-F | 0.99 | true | 10000 | 105 | 317 | true | 0.0105 | 0.00859577, 0.0126968 | null | null |
| calibration/C16/75/0.990/P-F | 0.99 | true | 10000 | 115 | 317 | true | 0.0115 | 0.00950351, 0.0137881 | null | null |
| calibration/C16/75/0.990/joint | 0.99 | true | 10000 | 217 | 593 | true | 0.0217 | 0.0189343, 0.0247489 | null | null |
| calibration/C16/90/0.975/A-F | 0.975 | true | 10000 | 270 | 317 | true | 0.027 | 0.0239116, 0.0303678 | null | null |
| calibration/C16/90/0.975/P-F | 0.975 | true | 10000 | 256 | 317 | true | 0.0256 | 0.0225932, 0.0288872 | null | null |
| calibration/C16/90/0.975/joint | 0.975 | true | 10000 | 515 | 593 | true | 0.0515 | 0.0472493, 0.0560142 | null | null |
| calibration/C16/90/0.980/A-F | 0.98 | true | 10000 | 213 | 317 | true | 0.0213 | 0.0185602, 0.0243233 | null | null |
| calibration/C16/90/0.980/P-F | 0.98 | true | 10000 | 207 | 317 | true | 0.0207 | 0.0179997, 0.0236842 | null | null |
| calibration/C16/90/0.980/joint | 0.98 | true | 10000 | 411 | 593 | true | 0.0411 | 0.0372932, 0.0451769 | null | null |
| calibration/C16/90/0.985/A-F | 0.985 | true | 10000 | 180 | 317 | true | 0.018 | 0.0154853, 0.0208007 | null | null |
| calibration/C16/90/0.985/P-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| calibration/C16/90/0.985/joint | 0.985 | true | 10000 | 327 | 593 | true | 0.0327 | 0.0293009, 0.0363747 | null | null |
| calibration/C16/90/0.990/A-F | 0.99 | true | 10000 | 121 | 317 | true | 0.0121 | 0.0100501, 0.0144409 | null | null |
| calibration/C16/90/0.990/P-F | 0.99 | true | 10000 | 99 | 317 | true | 0.0099 | 0.00805331, 0.0120399 | null | null |
| calibration/C16/90/0.990/joint | 0.99 | true | 10000 | 220 | 593 | true | 0.022 | 0.0192149, 0.025068 | null | null |
| calibration/C16/99/0.975/A-F | 0.975 | true | 10000 | 271 | 317 | true | 0.0271 | 0.0240058, 0.0304735 | null | null |
| calibration/C16/99/0.975/P-F | 0.975 | true | 10000 | 258 | 317 | true | 0.0258 | 0.0227814, 0.0290988 | null | null |
| calibration/C16/99/0.975/joint | 0.975 | true | 10000 | 517 | 593 | true | 0.0517 | 0.0474413, 0.0562221 | null | null |
| calibration/C16/99/0.980/A-F | 0.98 | true | 10000 | 212 | 317 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| calibration/C16/99/0.980/P-F | 0.98 | true | 10000 | 208 | 317 | true | 0.0208 | 0.0180931, 0.0237908 | null | null |
| calibration/C16/99/0.980/joint | 0.98 | true | 10000 | 412 | 593 | true | 0.0412 | 0.0373887, 0.0452814 | null | null |
| calibration/C16/99/0.985/A-F | 0.985 | true | 10000 | 158 | 317 | true | 0.0158 | 0.0134477, 0.0184401 | null | null |
| calibration/C16/99/0.985/P-F | 0.985 | true | 10000 | 158 | 317 | true | 0.0158 | 0.0134477, 0.0184401 | null | null |
| calibration/C16/99/0.985/joint | 0.985 | true | 10000 | 310 | 593 | true | 0.031 | 0.0276902, 0.0345865 | null | null |
| calibration/C16/99/0.990/A-F | 0.99 | true | 10000 | 106 | 317 | true | 0.0106 | 0.00868634, 0.0128061 | null | null |
| calibration/C16/99/0.990/P-F | 0.99 | true | 10000 | 103 | 317 | true | 0.0103 | 0.00841475, 0.012478 | null | null |
| calibration/C16/99/0.990/joint | 0.99 | true | 10000 | 206 | 593 | true | 0.0206 | 0.0179063, 0.0235777 | null | null |
| calibration/C16/cap/0.975/A-F | 0.975 | true | 10000 | 244 | 317 | true | 0.0244 | 0.0214652, 0.0276161 | null | 60 |
| calibration/C16/cap/0.975/P-F | 0.975 | true | 10000 | 254 | 317 | true | 0.0254 | 0.0224051, 0.0286754 | null | 60 |
| calibration/C16/cap/0.975/joint | 0.975 | true | 10000 | 487 | 593 | true | 0.0487 | 0.0445633, 0.0531019 | null | 60 |
| calibration/C16/cap/0.980/A-F | 0.98 | true | 10000 | 181 | 317 | true | 0.0181 | 0.0155782, 0.0209077 | null | 60 |
| calibration/C16/cap/0.980/P-F | 0.98 | true | 10000 | 203 | 317 | true | 0.0203 | 0.0176264, 0.0232579 | null | 60 |
| calibration/C16/cap/0.980/joint | 0.98 | true | 10000 | 378 | 593 | true | 0.0378 | 0.0341473, 0.041725 | null | 60 |
| calibration/C16/cap/0.985/A-F | 0.985 | true | 10000 | 138 | 317 | true | 0.0138 | 0.011606, 0.0162835 | null | 60 |
| calibration/C16/cap/0.985/P-F | 0.985 | true | 10000 | 155 | 317 | true | 0.0155 | 0.0131707, 0.0181173 | null | 60 |
| calibration/C16/cap/0.985/joint | 0.985 | true | 10000 | 288 | 593 | true | 0.0288 | 0.0256099, 0.0322683 | null | 60 |
| calibration/C16/cap/0.990/A-F | 0.99 | true | 10000 | 96 | 317 | true | 0.0096 | 0.00778275, 0.0117108 | null | 60 |
| calibration/C16/cap/0.990/P-F | 0.99 | true | 10000 | 105 | 317 | true | 0.0105 | 0.00859577, 0.0126968 | null | 60 |
| calibration/C16/cap/0.990/joint | 0.99 | true | 10000 | 197 | 593 | true | 0.0197 | 0.0170669, 0.0226178 | null | 60 |
| calibration/C17/36/0.975/A-F | 0.975 | true | 10000 | 276 | 317 | true | 0.0276 | 0.0244773, 0.0310017 | null | null |
| calibration/C17/36/0.975/P-F | 0.975 | true | 10000 | 229 | 317 | true | 0.0229 | 0.0200578, 0.0260245 | null | null |
| calibration/C17/36/0.975/joint | 0.975 | true | 10000 | 499 | 593 | true | 0.0499 | 0.045714, 0.0543505 | null | null |
| calibration/C17/36/0.980/A-F | 0.98 | true | 10000 | 220 | 317 | true | 0.022 | 0.0192149, 0.025068 | null | null |
| calibration/C17/36/0.980/P-F | 0.98 | true | 10000 | 191 | 317 | true | 0.0191 | 0.016508, 0.0219771 | null | null |
| calibration/C17/36/0.980/joint | 0.98 | true | 10000 | 408 | 593 | true | 0.0408 | 0.0370069, 0.0448633 | null | null |
| calibration/C17/36/0.985/A-F | 0.985 | true | 10000 | 159 | 317 | true | 0.0159 | 0.0135401, 0.0185476 | null | null |
| calibration/C17/36/0.985/P-F | 0.985 | true | 10000 | 134 | 317 | true | 0.0134 | 0.011239, 0.0158508 | null | null |
| calibration/C17/36/0.985/joint | 0.985 | true | 10000 | 292 | 593 | true | 0.0292 | 0.0259878, 0.0326901 | null | null |
| calibration/C17/36/0.990/A-F | 0.99 | true | 10000 | 101 | 317 | true | 0.0101 | 0.00823393, 0.0122591 | null | null |
| calibration/C17/36/0.990/P-F | 0.99 | true | 10000 | 85 | 317 | true | 0.0085 | 0.00679498, 0.0104998 | null | null |
| calibration/C17/36/0.990/joint | 0.99 | true | 10000 | 186 | 593 | true | 0.0186 | 0.0160429, 0.0214427 | null | null |
| calibration/C17/45/0.975/A-F | 0.975 | true | 10000 | 267 | 317 | true | 0.0267 | 0.0236289, 0.0300508 | null | null |
| calibration/C17/45/0.975/P-F | 0.975 | true | 10000 | 239 | 317 | true | 0.0239 | 0.0209957, 0.0270859 | null | null |
| calibration/C17/45/0.975/joint | 0.975 | true | 10000 | 502 | 593 | true | 0.0502 | 0.0460018, 0.0546625 | null | null |
| calibration/C17/45/0.980/A-F | 0.98 | true | 10000 | 226 | 317 | true | 0.0226 | 0.0197767, 0.0257058 | null | null |
| calibration/C17/45/0.980/P-F | 0.98 | true | 10000 | 197 | 317 | true | 0.0197 | 0.0170669, 0.0226178 | null | null |
| calibration/C17/45/0.980/joint | 0.98 | true | 10000 | 420 | 593 | true | 0.042 | 0.0381524, 0.0461171 | null | null |
| calibration/C17/45/0.985/A-F | 0.985 | true | 10000 | 172 | 317 | true | 0.0172 | 0.0147431, 0.0199435 | null | null |
| calibration/C17/45/0.985/P-F | 0.985 | true | 10000 | 148 | 317 | true | 0.0148 | 0.0125254, 0.0173632 | null | null |
| calibration/C17/45/0.985/joint | 0.985 | true | 10000 | 319 | 593 | true | 0.0319 | 0.0285426, 0.0355335 | null | null |
| calibration/C17/45/0.990/A-F | 0.99 | true | 10000 | 109 | 317 | true | 0.0109 | 0.00895834, 0.0131338 | null | null |
| calibration/C17/45/0.990/P-F | 0.99 | true | 10000 | 96 | 317 | true | 0.0096 | 0.00778275, 0.0117108 | null | null |
| calibration/C17/45/0.990/joint | 0.99 | true | 10000 | 204 | 593 | true | 0.0204 | 0.0177197, 0.0233645 | null | null |
| calibration/C17/60/0.975/A-F | 0.975 | true | 10000 | 255 | 317 | true | 0.0255 | 0.0224992, 0.0287813 | null | null |
| calibration/C17/60/0.975/P-F | 0.975 | true | 10000 | 231 | 317 | true | 0.0231 | 0.0202453, 0.0262369 | null | null |
| calibration/C17/60/0.975/joint | 0.975 | true | 10000 | 479 | 593 | true | 0.0479 | 0.0437966, 0.0522692 | null | null |
| calibration/C17/60/0.980/A-F | 0.98 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| calibration/C17/60/0.980/P-F | 0.98 | true | 10000 | 195 | 317 | true | 0.0195 | 0.0168805, 0.0224043 | null | null |
| calibration/C17/60/0.980/joint | 0.98 | true | 10000 | 393 | 593 | true | 0.0393 | 0.0355764, 0.0432949 | null | null |
| calibration/C17/60/0.985/A-F | 0.985 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | null |
| calibration/C17/60/0.985/P-F | 0.985 | true | 10000 | 147 | 317 | true | 0.0147 | 0.0124334, 0.0172553 | null | null |
| calibration/C17/60/0.985/joint | 0.985 | true | 10000 | 297 | 593 | true | 0.0297 | 0.0264603, 0.0332172 | null | null |
| calibration/C17/60/0.990/A-F | 0.99 | true | 10000 | 106 | 317 | true | 0.0106 | 0.00868634, 0.0128061 | null | null |
| calibration/C17/60/0.990/P-F | 0.99 | true | 10000 | 97 | 317 | true | 0.0097 | 0.00787289, 0.0118205 | null | null |
| calibration/C17/60/0.990/joint | 0.99 | true | 10000 | 203 | 593 | true | 0.0203 | 0.0176264, 0.0232579 | null | null |
| calibration/C17/75/0.975/A-F | 0.975 | true | 10000 | 264 | 317 | true | 0.0264 | 0.0233463, 0.0297336 | null | null |
| calibration/C17/75/0.975/P-F | 0.975 | true | 10000 | 254 | 317 | true | 0.0254 | 0.0224051, 0.0286754 | null | null |
| calibration/C17/75/0.975/joint | 0.975 | true | 10000 | 511 | 593 | true | 0.0511 | 0.0468654, 0.0555984 | null | null |
| calibration/C17/75/0.980/A-F | 0.98 | true | 10000 | 219 | 317 | true | 0.0219 | 0.0191214, 0.0249617 | null | null |
| calibration/C17/75/0.980/P-F | 0.98 | true | 10000 | 206 | 317 | true | 0.0206 | 0.0179063, 0.0235777 | null | null |
| calibration/C17/75/0.980/joint | 0.98 | true | 10000 | 422 | 593 | true | 0.0422 | 0.0383435, 0.0463259 | null | null |
| calibration/C17/75/0.985/A-F | 0.985 | true | 10000 | 168 | 317 | true | 0.0168 | 0.0143725, 0.0195144 | null | null |
| calibration/C17/75/0.985/P-F | 0.985 | true | 10000 | 153 | 317 | true | 0.0153 | 0.0129862, 0.0179019 | null | null |
| calibration/C17/75/0.985/joint | 0.985 | true | 10000 | 319 | 593 | true | 0.0319 | 0.0285426, 0.0355335 | null | null |
| calibration/C17/75/0.990/A-F | 0.99 | true | 10000 | 110 | 317 | true | 0.011 | 0.0090491, 0.013243 | null | null |
| calibration/C17/75/0.990/P-F | 0.99 | true | 10000 | 121 | 317 | true | 0.0121 | 0.0100501, 0.0144409 | null | null |
| calibration/C17/75/0.990/joint | 0.99 | true | 10000 | 231 | 593 | true | 0.0231 | 0.0202453, 0.0262369 | null | null |
| calibration/C17/90/0.975/A-F | 0.975 | true | 10000 | 267 | 317 | true | 0.0267 | 0.0236289, 0.0300508 | null | null |
| calibration/C17/90/0.975/P-F | 0.975 | true | 10000 | 259 | 317 | true | 0.0259 | 0.0228755, 0.0292047 | null | null |
| calibration/C17/90/0.975/joint | 0.975 | true | 10000 | 517 | 593 | true | 0.0517 | 0.0474413, 0.0562221 | null | null |
| calibration/C17/90/0.980/A-F | 0.98 | true | 10000 | 223 | 317 | true | 0.0223 | 0.0194957, 0.025387 | null | null |
| calibration/C17/90/0.980/P-F | 0.98 | true | 10000 | 216 | 317 | true | 0.0216 | 0.0188407, 0.0246425 | null | null |
| calibration/C17/90/0.980/joint | 0.98 | true | 10000 | 433 | 593 | true | 0.0433 | 0.0393944, 0.0474743 | null | null |
| calibration/C17/90/0.985/A-F | 0.985 | true | 10000 | 180 | 317 | true | 0.018 | 0.0154853, 0.0208007 | null | null |
| calibration/C17/90/0.985/P-F | 0.985 | true | 10000 | 162 | 317 | true | 0.0162 | 0.0138173, 0.0188701 | null | null |
| calibration/C17/90/0.985/joint | 0.985 | true | 10000 | 339 | 593 | true | 0.0339 | 0.0304393, 0.0376355 | null | null |
| calibration/C17/90/0.990/A-F | 0.99 | true | 10000 | 112 | 317 | true | 0.0112 | 0.00923074, 0.0134611 | null | null |
| calibration/C17/90/0.990/P-F | 0.99 | true | 10000 | 107 | 317 | true | 0.0107 | 0.00877696, 0.0129154 | null | null |
| calibration/C17/90/0.990/joint | 0.99 | true | 10000 | 218 | 593 | true | 0.0218 | 0.0190278, 0.0248553 | null | null |
| calibration/C17/99/0.975/A-F | 0.975 | true | 10000 | 263 | 317 | true | 0.0263 | 0.0232521, 0.0296278 | null | null |
| calibration/C17/99/0.975/P-F | 0.975 | true | 10000 | 252 | 317 | true | 0.0252 | 0.022217, 0.0284637 | null | null |
| calibration/C17/99/0.975/joint | 0.975 | true | 10000 | 510 | 593 | true | 0.051 | 0.0467694, 0.0554944 | null | null |
| calibration/C17/99/0.980/A-F | 0.98 | true | 10000 | 217 | 317 | true | 0.0217 | 0.0189343, 0.0247489 | null | null |
| calibration/C17/99/0.980/P-F | 0.98 | true | 10000 | 201 | 317 | true | 0.0201 | 0.0174398, 0.0230446 | null | null |
| calibration/C17/99/0.980/joint | 0.98 | true | 10000 | 415 | 593 | true | 0.0415 | 0.037675, 0.0455948 | null | null |
| calibration/C17/99/0.985/A-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| calibration/C17/99/0.985/P-F | 0.985 | true | 10000 | 155 | 317 | true | 0.0155 | 0.0131707, 0.0181173 | null | null |
| calibration/C17/99/0.985/joint | 0.985 | true | 10000 | 312 | 593 | true | 0.0312 | 0.0278796, 0.034797 | null | null |
| calibration/C17/99/0.990/A-F | 0.99 | true | 10000 | 111 | 317 | true | 0.0111 | 0.00913989, 0.0133521 | null | null |
| calibration/C17/99/0.990/P-F | 0.99 | true | 10000 | 120 | 317 | true | 0.012 | 0.00995893, 0.0143322 | null | null |
| calibration/C17/99/0.990/joint | 0.99 | true | 10000 | 230 | 593 | true | 0.023 | 0.0201515, 0.0261307 | null | null |
| calibration/C17/cap/0.975/A-F | 0.975 | true | 10000 | 267 | 317 | true | 0.0267 | 0.0236289, 0.0300508 | null | 60 |
| calibration/C17/cap/0.975/P-F | 0.975 | true | 10000 | 252 | 317 | true | 0.0252 | 0.022217, 0.0284637 | null | 60 |
| calibration/C17/cap/0.975/joint | 0.975 | true | 10000 | 515 | 593 | true | 0.0515 | 0.0472493, 0.0560142 | null | 60 |
| calibration/C17/cap/0.980/A-F | 0.98 | true | 10000 | 229 | 317 | true | 0.0229 | 0.0200578, 0.0260245 | null | 60 |
| calibration/C17/cap/0.980/P-F | 0.98 | true | 10000 | 209 | 317 | true | 0.0209 | 0.0181865, 0.0238973 | null | 60 |
| calibration/C17/cap/0.980/joint | 0.98 | true | 10000 | 430 | 593 | true | 0.043 | 0.0391077, 0.0471612 | null | 60 |
| calibration/C17/cap/0.985/A-F | 0.985 | true | 10000 | 178 | 317 | true | 0.0178 | 0.0152996, 0.0205865 | null | 60 |
| calibration/C17/cap/0.985/P-F | 0.985 | true | 10000 | 166 | 317 | true | 0.0166 | 0.0141874, 0.0192998 | null | 60 |
| calibration/C17/cap/0.985/joint | 0.985 | true | 10000 | 340 | 593 | true | 0.034 | 0.0305342, 0.0377405 | null | 60 |
| calibration/C17/cap/0.990/A-F | 0.99 | true | 10000 | 117 | 317 | true | 0.0117 | 0.00968556, 0.0140058 | null | 60 |
| calibration/C17/cap/0.990/P-F | 0.99 | true | 10000 | 116 | 317 | true | 0.0116 | 0.00959451, 0.013897 | null | 60 |
| calibration/C17/cap/0.990/joint | 0.99 | true | 10000 | 231 | 593 | true | 0.0231 | 0.0202453, 0.0262369 | null | 60 |
| verification/C01/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C01/36/0.980/A-F | 0.98 | true | 10000 | 211 | 317 | true | 0.0211 | 0.0183733, 0.0241103 | null | null |
| verification/C01/36/0.980/P-F | 0.98 | true | 10000 | 196 | 317 | true | 0.0196 | 0.0169737, 0.0225111 | null | null |
| verification/C01/36/0.980/joint | 0.98 | true | 10000 | 402 | 593 | true | 0.0402 | 0.0364345, 0.0442361 | null | null |
| verification/C01/36/0.985/A-F | 0.985 | true | 10000 | 145 | 317 | true | 0.0145 | 0.0122493, 0.0170396 | null | null |
| verification/C01/36/0.985/P-F | 0.985 | true | 10000 | 140 | 317 | true | 0.014 | 0.0117896, 0.0164997 | null | null |
| verification/C01/36/0.985/joint | 0.985 | true | 10000 | 278 | 593 | true | 0.0278 | 0.0246659, 0.0312129 | null | null |
| verification/C01/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C01/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C01/45/0.980/A-F | 0.98 | true | 10000 | 196 | 317 | true | 0.0196 | 0.0169737, 0.0225111 | null | null |
| verification/C01/45/0.980/P-F | 0.98 | true | 10000 | 195 | 317 | true | 0.0195 | 0.0168805, 0.0224043 | null | null |
| verification/C01/45/0.980/joint | 0.98 | true | 10000 | 384 | 593 | true | 0.0384 | 0.0347187, 0.0423531 | null | null |
| verification/C01/45/0.985/A-F | 0.985 | true | 10000 | 149 | 317 | true | 0.0149 | 0.0126175, 0.017471 | null | null |
| verification/C01/45/0.985/P-F | 0.985 | true | 10000 | 147 | 317 | true | 0.0147 | 0.0124334, 0.0172553 | null | null |
| verification/C01/45/0.985/joint | 0.985 | true | 10000 | 292 | 593 | true | 0.0292 | 0.0259878, 0.0326901 | null | null |
| verification/C01/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C01/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C01/60/0.980/A-F | 0.98 | true | 10000 | 227 | 317 | true | 0.0227 | 0.0198704, 0.0258121 | null | null |
| verification/C01/60/0.980/P-F | 0.98 | true | 10000 | 206 | 317 | true | 0.0206 | 0.0179063, 0.0235777 | null | null |
| verification/C01/60/0.980/joint | 0.98 | true | 10000 | 427 | 593 | true | 0.0427 | 0.0388211, 0.046848 | null | null |
| verification/C01/60/0.985/A-F | 0.985 | true | 10000 | 167 | 317 | true | 0.0167 | 0.0142799, 0.0194071 | null | null |
| verification/C01/60/0.985/P-F | 0.985 | true | 10000 | 147 | 317 | true | 0.0147 | 0.0124334, 0.0172553 | null | null |
| verification/C01/60/0.985/joint | 0.985 | true | 10000 | 308 | 593 | true | 0.0308 | 0.0275009, 0.034376 | null | null |
| verification/C01/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C01/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C01/75/0.980/A-F | 0.98 | true | 10000 | 206 | 317 | true | 0.0206 | 0.0179063, 0.0235777 | null | null |
| verification/C01/75/0.980/P-F | 0.98 | true | 10000 | 194 | 317 | true | 0.0194 | 0.0167874, 0.0222975 | null | null |
| verification/C01/75/0.980/joint | 0.98 | true | 10000 | 395 | 593 | true | 0.0395 | 0.035767, 0.0435041 | null | null |
| verification/C01/75/0.985/A-F | 0.985 | true | 10000 | 149 | 317 | true | 0.0149 | 0.0126175, 0.017471 | null | null |
| verification/C01/75/0.985/P-F | 0.985 | true | 10000 | 147 | 317 | true | 0.0147 | 0.0124334, 0.0172553 | null | null |
| verification/C01/75/0.985/joint | 0.985 | true | 10000 | 295 | 593 | true | 0.0295 | 0.0262713, 0.0330064 | null | null |
| verification/C01/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C01/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C01/90/0.980/A-F | 0.98 | true | 10000 | 194 | 317 | true | 0.0194 | 0.0167874, 0.0222975 | null | null |
| verification/C01/90/0.980/P-F | 0.98 | true | 10000 | 215 | 317 | true | 0.0215 | 0.0187472, 0.0245361 | null | null |
| verification/C01/90/0.980/joint | 0.98 | true | 10000 | 397 | 593 | true | 0.0397 | 0.0359577, 0.0437133 | null | null |
| verification/C01/90/0.985/A-F | 0.985 | true | 10000 | 155 | 317 | true | 0.0155 | 0.0131707, 0.0181173 | null | null |
| verification/C01/90/0.985/P-F | 0.985 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | null |
| verification/C01/90/0.985/joint | 0.985 | true | 10000 | 298 | 593 | true | 0.0298 | 0.0265549, 0.0333226 | null | null |
| verification/C01/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C01/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C01/99/0.980/A-F | 0.98 | true | 10000 | 188 | 317 | true | 0.0188 | 0.0162289, 0.0216565 | null | null |
| verification/C01/99/0.980/P-F | 0.98 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| verification/C01/99/0.980/joint | 0.98 | true | 10000 | 379 | 593 | true | 0.0379 | 0.0342425, 0.0418297 | null | null |
| verification/C01/99/0.985/A-F | 0.985 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| verification/C01/99/0.985/P-F | 0.985 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | null |
| verification/C01/99/0.985/joint | 0.985 | true | 10000 | 294 | 593 | true | 0.0294 | 0.0261768, 0.032901 | null | null |
| verification/C01/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C01/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C01/cap/0.980/A-F | 0.98 | true | 10000 | 235 | 317 | true | 0.0235 | 0.0206204, 0.0266615 | null | 36 |
| verification/C01/cap/0.980/P-F | 0.98 | true | 10000 | 200 | 317 | true | 0.02 | 0.0173465, 0.0229379 | null | 36 |
| verification/C01/cap/0.980/joint | 0.98 | true | 10000 | 430 | 593 | true | 0.043 | 0.0391077, 0.0471612 | null | 36 |
| verification/C01/cap/0.985/A-F | 0.985 | true | 10000 | 164 | 317 | true | 0.0164 | 0.0140023, 0.019085 | null | 36 |
| verification/C01/cap/0.985/P-F | 0.985 | true | 10000 | 149 | 317 | true | 0.0149 | 0.0126175, 0.017471 | null | 36 |
| verification/C01/cap/0.985/joint | 0.985 | true | 10000 | 308 | 593 | true | 0.0308 | 0.0275009, 0.034376 | null | 36 |
| verification/C01/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C01/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C02/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C02/36/0.980/A-F | 0.98 | true | 10000 | 196 | 317 | true | 0.0196 | 0.0169737, 0.0225111 | null | null |
| verification/C02/36/0.980/P-F | 0.98 | true | 10000 | 201 | 317 | true | 0.0201 | 0.0174398, 0.0230446 | null | null |
| verification/C02/36/0.980/joint | 0.98 | true | 10000 | 386 | 593 | true | 0.0386 | 0.0349093, 0.0425625 | null | null |
| verification/C02/36/0.985/A-F | 0.985 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| verification/C02/36/0.985/P-F | 0.985 | true | 10000 | 161 | 317 | true | 0.0161 | 0.0137249, 0.0187626 | null | null |
| verification/C02/36/0.985/joint | 0.985 | true | 10000 | 308 | 593 | true | 0.0308 | 0.0275009, 0.034376 | null | null |
| verification/C02/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C02/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C02/45/0.980/A-F | 0.98 | true | 10000 | 196 | 317 | true | 0.0196 | 0.0169737, 0.0225111 | null | null |
| verification/C02/45/0.980/P-F | 0.98 | true | 10000 | 203 | 317 | true | 0.0203 | 0.0176264, 0.0232579 | null | null |
| verification/C02/45/0.980/joint | 0.98 | true | 10000 | 393 | 593 | true | 0.0393 | 0.0355764, 0.0432949 | null | null |
| verification/C02/45/0.985/A-F | 0.985 | true | 10000 | 147 | 317 | true | 0.0147 | 0.0124334, 0.0172553 | null | null |
| verification/C02/45/0.985/P-F | 0.985 | true | 10000 | 166 | 317 | true | 0.0166 | 0.0141874, 0.0192998 | null | null |
| verification/C02/45/0.985/joint | 0.985 | true | 10000 | 306 | 593 | true | 0.0306 | 0.0273116, 0.0341654 | null | null |
| verification/C02/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C02/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C02/60/0.980/A-F | 0.98 | true | 10000 | 199 | 317 | true | 0.0199 | 0.0172533, 0.0228312 | null | null |
| verification/C02/60/0.980/P-F | 0.98 | true | 10000 | 177 | 317 | true | 0.0177 | 0.0152068, 0.0204794 | null | null |
| verification/C02/60/0.980/joint | 0.98 | true | 10000 | 369 | 593 | true | 0.0369 | 0.0332906, 0.0407822 | null | null |
| verification/C02/60/0.985/A-F | 0.985 | true | 10000 | 162 | 317 | true | 0.0162 | 0.0138173, 0.0188701 | null | null |
| verification/C02/60/0.985/P-F | 0.985 | true | 10000 | 166 | 317 | true | 0.0166 | 0.0141874, 0.0192998 | null | null |
| verification/C02/60/0.985/joint | 0.985 | true | 10000 | 321 | 593 | true | 0.0321 | 0.0287321, 0.0357439 | null | null |
| verification/C02/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C02/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C02/75/0.980/A-F | 0.98 | true | 10000 | 192 | 317 | true | 0.0192 | 0.0166011, 0.0220839 | null | null |
| verification/C02/75/0.980/P-F | 0.98 | true | 10000 | 189 | 317 | true | 0.0189 | 0.0163219, 0.0217634 | null | null |
| verification/C02/75/0.980/joint | 0.98 | true | 10000 | 372 | 593 | true | 0.0372 | 0.0335761, 0.0410965 | null | null |
| verification/C02/75/0.985/A-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| verification/C02/75/0.985/P-F | 0.985 | true | 10000 | 154 | 317 | true | 0.0154 | 0.0130785, 0.0180096 | null | null |
| verification/C02/75/0.985/joint | 0.985 | true | 10000 | 296 | 593 | true | 0.0296 | 0.0263658, 0.0331118 | null | null |
| verification/C02/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C02/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C02/90/0.980/A-F | 0.98 | true | 10000 | 197 | 317 | true | 0.0197 | 0.0170669, 0.0226178 | null | null |
| verification/C02/90/0.980/P-F | 0.98 | true | 10000 | 182 | 317 | true | 0.0182 | 0.0156711, 0.0210148 | null | null |
| verification/C02/90/0.980/joint | 0.98 | true | 10000 | 370 | 593 | true | 0.037 | 0.0333857, 0.040887 | null | null |
| verification/C02/90/0.985/A-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| verification/C02/90/0.985/P-F | 0.985 | true | 10000 | 155 | 317 | true | 0.0155 | 0.0131707, 0.0181173 | null | null |
| verification/C02/90/0.985/joint | 0.985 | true | 10000 | 301 | 593 | true | 0.0301 | 0.0268386, 0.0336387 | null | null |
| verification/C02/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C02/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C02/99/0.980/A-F | 0.98 | true | 10000 | 200 | 317 | true | 0.02 | 0.0173465, 0.0229379 | null | null |
| verification/C02/99/0.980/P-F | 0.98 | true | 10000 | 191 | 317 | true | 0.0191 | 0.016508, 0.0219771 | null | null |
| verification/C02/99/0.980/joint | 0.98 | true | 10000 | 384 | 593 | true | 0.0384 | 0.0347187, 0.0423531 | null | null |
| verification/C02/99/0.985/A-F | 0.985 | true | 10000 | 141 | 317 | true | 0.0141 | 0.0118815, 0.0166077 | null | null |
| verification/C02/99/0.985/P-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| verification/C02/99/0.985/joint | 0.985 | true | 10000 | 295 | 593 | true | 0.0295 | 0.0262713, 0.0330064 | null | null |
| verification/C02/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C02/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C02/cap/0.980/A-F | 0.98 | true | 10000 | 197 | 317 | true | 0.0197 | 0.0170669, 0.0226178 | null | 36 |
| verification/C02/cap/0.980/P-F | 0.98 | true | 10000 | 209 | 317 | true | 0.0209 | 0.0181865, 0.0238973 | null | 36 |
| verification/C02/cap/0.980/joint | 0.98 | true | 10000 | 398 | 593 | true | 0.0398 | 0.0360531, 0.0438179 | null | 36 |
| verification/C02/cap/0.985/A-F | 0.985 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | 36 |
| verification/C02/cap/0.985/P-F | 0.985 | true | 10000 | 147 | 317 | true | 0.0147 | 0.0124334, 0.0172553 | null | 36 |
| verification/C02/cap/0.985/joint | 0.985 | true | 10000 | 293 | 593 | true | 0.0293 | 0.0260823, 0.0327956 | null | 36 |
| verification/C02/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C02/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C03/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C03/36/0.980/A-F | 0.98 | true | 10000 | 256 | 317 | true | 0.0256 | 0.0225932, 0.0288872 | null | null |
| verification/C03/36/0.980/P-F | 0.98 | true | 10000 | 240 | 317 | true | 0.024 | 0.0210896, 0.027192 | null | null |
| verification/C03/36/0.980/joint | 0.98 | true | 10000 | 483 | 593 | true | 0.0483 | 0.0441799, 0.0526856 | null | null |
| verification/C03/36/0.985/A-F | 0.985 | true | 10000 | 210 | 317 | true | 0.021 | 0.0182799, 0.0240038 | null | null |
| verification/C03/36/0.985/P-F | 0.985 | true | 10000 | 200 | 317 | true | 0.02 | 0.0173465, 0.0229379 | null | null |
| verification/C03/36/0.985/joint | 0.985 | true | 10000 | 401 | 593 | true | 0.0401 | 0.0363392, 0.0441316 | null | null |
| verification/C03/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C03/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C03/45/0.980/A-F | 0.98 | true | 10000 | 262 | 317 | true | 0.0262 | 0.023158, 0.029522 | null | null |
| verification/C03/45/0.980/P-F | 0.98 | true | 10000 | 244 | 317 | true | 0.0244 | 0.0214652, 0.0276161 | null | null |
| verification/C03/45/0.980/joint | 0.98 | true | 10000 | 489 | 593 | true | 0.0489 | 0.0447551, 0.0533101 | null | null |
| verification/C03/45/0.985/A-F | 0.985 | true | 10000 | 198 | 317 | true | 0.0198 | 0.0171601, 0.0227245 | null | null |
| verification/C03/45/0.985/P-F | 0.985 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| verification/C03/45/0.985/joint | 0.985 | true | 10000 | 378 | 593 | true | 0.0378 | 0.0341473, 0.041725 | null | null |
| verification/C03/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C03/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C03/60/0.980/A-F | 0.98 | true | 10000 | 249 | 317 | true | 0.0249 | 0.021935, 0.0281459 | null | null |
| verification/C03/60/0.980/P-F | 0.98 | true | 10000 | 236 | 317 | true | 0.0236 | 0.0207142, 0.0267676 | null | null |
| verification/C03/60/0.980/joint | 0.98 | true | 10000 | 471 | 593 | true | 0.0471 | 0.0430302, 0.0514361 | null | null |
| verification/C03/60/0.985/A-F | 0.985 | true | 10000 | 188 | 317 | true | 0.0188 | 0.0162289, 0.0216565 | null | null |
| verification/C03/60/0.985/P-F | 0.985 | true | 10000 | 182 | 317 | true | 0.0182 | 0.0156711, 0.0210148 | null | null |
| verification/C03/60/0.985/joint | 0.985 | true | 10000 | 365 | 593 | true | 0.0365 | 0.03291, 0.0403631 | null | null |
| verification/C03/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C03/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C03/75/0.980/A-F | 0.98 | true | 10000 | 260 | 317 | true | 0.026 | 0.0229697, 0.0293105 | null | null |
| verification/C03/75/0.980/P-F | 0.98 | true | 10000 | 218 | 317 | true | 0.0218 | 0.0190278, 0.0248553 | null | null |
| verification/C03/75/0.980/joint | 0.98 | true | 10000 | 471 | 593 | true | 0.0471 | 0.0430302, 0.0514361 | null | null |
| verification/C03/75/0.985/A-F | 0.985 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| verification/C03/75/0.985/P-F | 0.985 | true | 10000 | 187 | 317 | true | 0.0187 | 0.0161359, 0.0215496 | null | null |
| verification/C03/75/0.985/joint | 0.985 | true | 10000 | 367 | 593 | true | 0.0367 | 0.0331003, 0.0405727 | null | null |
| verification/C03/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C03/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C03/90/0.980/A-F | 0.98 | true | 10000 | 231 | 317 | true | 0.0231 | 0.0202453, 0.0262369 | null | null |
| verification/C03/90/0.980/P-F | 0.98 | true | 10000 | 230 | 317 | true | 0.023 | 0.0201515, 0.0261307 | null | null |
| verification/C03/90/0.980/joint | 0.98 | true | 10000 | 442 | 593 | true | 0.0442 | 0.0402548, 0.0484133 | null | null |
| verification/C03/90/0.985/A-F | 0.985 | true | 10000 | 199 | 317 | true | 0.0199 | 0.0172533, 0.0228312 | null | null |
| verification/C03/90/0.985/P-F | 0.985 | true | 10000 | 206 | 317 | true | 0.0206 | 0.0179063, 0.0235777 | null | null |
| verification/C03/90/0.985/joint | 0.985 | true | 10000 | 399 | 593 | true | 0.0399 | 0.0361484, 0.0439225 | null | null |
| verification/C03/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C03/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C03/99/0.980/A-F | 0.98 | true | 10000 | 227 | 317 | true | 0.0227 | 0.0198704, 0.0258121 | null | null |
| verification/C03/99/0.980/P-F | 0.98 | true | 10000 | 241 | 317 | true | 0.0241 | 0.0211834, 0.027298 | null | null |
| verification/C03/99/0.980/joint | 0.98 | true | 10000 | 455 | 593 | true | 0.0455 | 0.0414984, 0.0497689 | null | null |
| verification/C03/99/0.985/A-F | 0.985 | true | 10000 | 189 | 317 | true | 0.0189 | 0.0163219, 0.0217634 | null | null |
| verification/C03/99/0.985/P-F | 0.985 | true | 10000 | 176 | 317 | true | 0.0176 | 0.015114, 0.0203723 | null | null |
| verification/C03/99/0.985/joint | 0.985 | true | 10000 | 360 | 593 | true | 0.036 | 0.0324345, 0.0398389 | null | null |
| verification/C03/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C03/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C03/cap/0.980/A-F | 0.98 | true | 10000 | 244 | 317 | true | 0.0244 | 0.0214652, 0.0276161 | null | 36 |
| verification/C03/cap/0.980/P-F | 0.98 | true | 10000 | 237 | 317 | true | 0.0237 | 0.020808, 0.0268738 | null | 36 |
| verification/C03/cap/0.980/joint | 0.98 | true | 10000 | 468 | 593 | true | 0.0468 | 0.0427429, 0.0511236 | null | 36 |
| verification/C03/cap/0.985/A-F | 0.985 | true | 10000 | 186 | 317 | true | 0.0186 | 0.0160429, 0.0214427 | null | 36 |
| verification/C03/cap/0.985/P-F | 0.985 | true | 10000 | 194 | 317 | true | 0.0194 | 0.0167874, 0.0222975 | null | 36 |
| verification/C03/cap/0.985/joint | 0.985 | true | 10000 | 374 | 593 | true | 0.0374 | 0.0337664, 0.041306 | null | 36 |
| verification/C03/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C03/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C04/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C04/36/0.980/A-F | 0.98 | true | 10000 | 253 | 317 | true | 0.0253 | 0.022311, 0.0285696 | null | null |
| verification/C04/36/0.980/P-F | 0.98 | true | 10000 | 272 | 317 | true | 0.0272 | 0.0241001, 0.0305792 | null | null |
| verification/C04/36/0.980/joint | 0.98 | true | 10000 | 514 | 593 | true | 0.0514 | 0.0471533, 0.0559103 | null | null |
| verification/C04/36/0.985/A-F | 0.985 | true | 10000 | 217 | 317 | true | 0.0217 | 0.0189343, 0.0247489 | null | null |
| verification/C04/36/0.985/P-F | 0.985 | true | 10000 | 225 | 317 | true | 0.0225 | 0.019683, 0.0255996 | null | null |
| verification/C04/36/0.985/joint | 0.985 | true | 10000 | 429 | 593 | true | 0.0429 | 0.0390122, 0.0470568 | null | null |
| verification/C04/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C04/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C04/45/0.980/A-F | 0.98 | true | 10000 | 240 | 317 | true | 0.024 | 0.0210896, 0.027192 | null | null |
| verification/C04/45/0.980/P-F | 0.98 | true | 10000 | 276 | 317 | true | 0.0276 | 0.0244773, 0.0310017 | null | null |
| verification/C04/45/0.980/joint | 0.98 | true | 10000 | 504 | 593 | true | 0.0504 | 0.0461937, 0.0548705 | null | null |
| verification/C04/45/0.985/A-F | 0.985 | true | 10000 | 205 | 317 | true | 0.0205 | 0.017813, 0.0234711 | null | null |
| verification/C04/45/0.985/P-F | 0.985 | true | 10000 | 212 | 317 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| verification/C04/45/0.985/joint | 0.985 | true | 10000 | 409 | 593 | true | 0.0409 | 0.0371024, 0.0449679 | null | null |
| verification/C04/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C04/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C04/60/0.980/A-F | 0.98 | true | 10000 | 248 | 317 | true | 0.0248 | 0.021841, 0.02804 | null | null |
| verification/C04/60/0.980/P-F | 0.98 | true | 10000 | 266 | 317 | true | 0.0266 | 0.0235347, 0.029945 | null | null |
| verification/C04/60/0.980/joint | 0.98 | true | 10000 | 504 | 593 | true | 0.0504 | 0.0461937, 0.0548705 | null | null |
| verification/C04/60/0.985/A-F | 0.985 | true | 10000 | 199 | 317 | true | 0.0199 | 0.0172533, 0.0228312 | null | null |
| verification/C04/60/0.985/P-F | 0.985 | true | 10000 | 214 | 317 | true | 0.0214 | 0.0186537, 0.0244297 | null | null |
| verification/C04/60/0.985/joint | 0.985 | true | 10000 | 407 | 593 | true | 0.0407 | 0.0369115, 0.0447588 | null | null |
| verification/C04/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C04/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C04/75/0.980/A-F | 0.98 | true | 10000 | 231 | 317 | true | 0.0231 | 0.0202453, 0.0262369 | null | null |
| verification/C04/75/0.980/P-F | 0.98 | true | 10000 | 247 | 317 | true | 0.0247 | 0.021747, 0.027934 | null | null |
| verification/C04/75/0.980/joint | 0.98 | true | 10000 | 469 | 593 | true | 0.0469 | 0.0428386, 0.0512278 | null | null |
| verification/C04/75/0.985/A-F | 0.985 | true | 10000 | 175 | 317 | true | 0.0175 | 0.0150213, 0.0202651 | null | null |
| verification/C04/75/0.985/P-F | 0.985 | true | 10000 | 198 | 317 | true | 0.0198 | 0.0171601, 0.0227245 | null | null |
| verification/C04/75/0.985/joint | 0.985 | true | 10000 | 365 | 593 | true | 0.0365 | 0.03291, 0.0403631 | null | null |
| verification/C04/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C04/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C04/90/0.980/A-F | 0.98 | true | 10000 | 243 | 317 | true | 0.0243 | 0.0213712, 0.0275101 | null | null |
| verification/C04/90/0.980/P-F | 0.98 | true | 10000 | 220 | 317 | true | 0.022 | 0.0192149, 0.025068 | null | null |
| verification/C04/90/0.980/joint | 0.98 | true | 10000 | 457 | 593 | true | 0.0457 | 0.0416898, 0.0499774 | null | null |
| verification/C04/90/0.985/A-F | 0.985 | true | 10000 | 164 | 317 | true | 0.0164 | 0.0140023, 0.019085 | null | null |
| verification/C04/90/0.985/P-F | 0.985 | true | 10000 | 199 | 317 | true | 0.0199 | 0.0172533, 0.0228312 | null | null |
| verification/C04/90/0.985/joint | 0.985 | true | 10000 | 356 | 593 | true | 0.0356 | 0.0320542, 0.0394195 | null | null |
| verification/C04/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C04/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C04/99/0.980/A-F | 0.98 | true | 10000 | 250 | 317 | true | 0.025 | 0.022029, 0.0282519 | null | null |
| verification/C04/99/0.980/P-F | 0.98 | true | 10000 | 246 | 317 | true | 0.0246 | 0.021653, 0.0278281 | null | null |
| verification/C04/99/0.980/joint | 0.98 | true | 10000 | 493 | 593 | true | 0.0493 | 0.0451386, 0.0537263 | null | null |
| verification/C04/99/0.985/A-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| verification/C04/99/0.985/P-F | 0.985 | true | 10000 | 206 | 317 | true | 0.0206 | 0.0179063, 0.0235777 | null | null |
| verification/C04/99/0.985/joint | 0.985 | true | 10000 | 358 | 593 | true | 0.0358 | 0.0322443, 0.0396292 | null | null |
| verification/C04/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C04/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C04/cap/0.980/A-F | 0.98 | true | 10000 | 259 | 317 | true | 0.0259 | 0.0228755, 0.0292047 | null | 36 |
| verification/C04/cap/0.980/P-F | 0.98 | true | 10000 | 247 | 317 | true | 0.0247 | 0.021747, 0.027934 | null | 36 |
| verification/C04/cap/0.980/joint | 0.98 | true | 10000 | 499 | 593 | true | 0.0499 | 0.045714, 0.0543505 | null | 36 |
| verification/C04/cap/0.985/A-F | 0.985 | true | 10000 | 153 | 317 | true | 0.0153 | 0.0129862, 0.0179019 | null | 36 |
| verification/C04/cap/0.985/P-F | 0.985 | true | 10000 | 206 | 317 | true | 0.0206 | 0.0179063, 0.0235777 | null | 36 |
| verification/C04/cap/0.985/joint | 0.985 | true | 10000 | 350 | 593 | true | 0.035 | 0.031484, 0.0387901 | null | 36 |
| verification/C04/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C04/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C05/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C05/36/0.980/A-F | 0.98 | true | 10000 | 203 | 317 | true | 0.0203 | 0.0176264, 0.0232579 | null | null |
| verification/C05/36/0.980/P-F | 0.98 | true | 10000 | 221 | 317 | true | 0.0221 | 0.0193085, 0.0251744 | null | null |
| verification/C05/36/0.980/joint | 0.98 | true | 10000 | 417 | 593 | true | 0.0417 | 0.037866, 0.0458037 | null | null |
| verification/C05/36/0.985/A-F | 0.985 | true | 10000 | 165 | 317 | true | 0.0165 | 0.0140948, 0.0191924 | null | null |
| verification/C05/36/0.985/P-F | 0.985 | true | 10000 | 179 | 317 | true | 0.0179 | 0.0153925, 0.0206936 | null | null |
| verification/C05/36/0.985/joint | 0.985 | true | 10000 | 339 | 593 | true | 0.0339 | 0.0304393, 0.0376355 | null | null |
| verification/C05/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C05/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C05/45/0.980/A-F | 0.98 | true | 10000 | 204 | 317 | true | 0.0204 | 0.0177197, 0.0233645 | null | null |
| verification/C05/45/0.980/P-F | 0.98 | true | 10000 | 198 | 317 | true | 0.0198 | 0.0171601, 0.0227245 | null | null |
| verification/C05/45/0.980/joint | 0.98 | true | 10000 | 393 | 593 | true | 0.0393 | 0.0355764, 0.0432949 | null | null |
| verification/C05/45/0.985/A-F | 0.985 | true | 10000 | 153 | 317 | true | 0.0153 | 0.0129862, 0.0179019 | null | null |
| verification/C05/45/0.985/P-F | 0.985 | true | 10000 | 168 | 317 | true | 0.0168 | 0.0143725, 0.0195144 | null | null |
| verification/C05/45/0.985/joint | 0.985 | true | 10000 | 315 | 593 | true | 0.0315 | 0.0281637, 0.0351127 | null | null |
| verification/C05/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C05/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C05/60/0.980/A-F | 0.98 | true | 10000 | 195 | 317 | true | 0.0195 | 0.0168805, 0.0224043 | null | null |
| verification/C05/60/0.980/P-F | 0.98 | true | 10000 | 224 | 317 | true | 0.0224 | 0.0195894, 0.0254933 | null | null |
| verification/C05/60/0.980/joint | 0.98 | true | 10000 | 410 | 593 | true | 0.041 | 0.0371978, 0.0450724 | null | null |
| verification/C05/60/0.985/A-F | 0.985 | true | 10000 | 176 | 317 | true | 0.0176 | 0.015114, 0.0203723 | null | null |
| verification/C05/60/0.985/P-F | 0.985 | true | 10000 | 165 | 317 | true | 0.0165 | 0.0140948, 0.0191924 | null | null |
| verification/C05/60/0.985/joint | 0.985 | true | 10000 | 334 | 593 | true | 0.0334 | 0.0299648, 0.0371103 | null | null |
| verification/C05/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C05/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C05/75/0.980/A-F | 0.98 | true | 10000 | 183 | 317 | true | 0.0183 | 0.015764, 0.0211218 | null | null |
| verification/C05/75/0.980/P-F | 0.98 | true | 10000 | 223 | 317 | true | 0.0223 | 0.0194957, 0.025387 | null | null |
| verification/C05/75/0.980/joint | 0.98 | true | 10000 | 400 | 593 | true | 0.04 | 0.0362438, 0.044027 | null | null |
| verification/C05/75/0.985/A-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| verification/C05/75/0.985/P-F | 0.985 | true | 10000 | 164 | 317 | true | 0.0164 | 0.0140023, 0.019085 | null | null |
| verification/C05/75/0.985/joint | 0.985 | true | 10000 | 316 | 593 | true | 0.0316 | 0.0282584, 0.0352179 | null | null |
| verification/C05/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C05/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C05/90/0.980/A-F | 0.98 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| verification/C05/90/0.980/P-F | 0.98 | true | 10000 | 224 | 317 | true | 0.0224 | 0.0195894, 0.0254933 | null | null |
| verification/C05/90/0.980/joint | 0.98 | true | 10000 | 419 | 593 | true | 0.0419 | 0.038057, 0.0460126 | null | null |
| verification/C05/90/0.985/A-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| verification/C05/90/0.985/P-F | 0.985 | true | 10000 | 183 | 317 | true | 0.0183 | 0.015764, 0.0211218 | null | null |
| verification/C05/90/0.985/joint | 0.985 | true | 10000 | 329 | 593 | true | 0.0329 | 0.0294905, 0.0365849 | null | null |
| verification/C05/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C05/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C05/99/0.980/A-F | 0.98 | true | 10000 | 220 | 317 | true | 0.022 | 0.0192149, 0.025068 | null | null |
| verification/C05/99/0.980/P-F | 0.98 | true | 10000 | 220 | 317 | true | 0.022 | 0.0192149, 0.025068 | null | null |
| verification/C05/99/0.980/joint | 0.98 | true | 10000 | 431 | 593 | true | 0.0431 | 0.0392033, 0.0472656 | null | null |
| verification/C05/99/0.985/A-F | 0.985 | true | 10000 | 154 | 317 | true | 0.0154 | 0.0130785, 0.0180096 | null | null |
| verification/C05/99/0.985/P-F | 0.985 | true | 10000 | 174 | 317 | true | 0.0174 | 0.0149285, 0.020158 | null | null |
| verification/C05/99/0.985/joint | 0.985 | true | 10000 | 317 | 593 | true | 0.0317 | 0.0283531, 0.0353231 | null | null |
| verification/C05/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C05/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C05/cap/0.980/A-F | 0.98 | true | 10000 | 205 | 317 | true | 0.0205 | 0.017813, 0.0234711 | null | 36 |
| verification/C05/cap/0.980/P-F | 0.98 | true | 10000 | 228 | 317 | true | 0.0228 | 0.0199641, 0.0259183 | null | 36 |
| verification/C05/cap/0.980/joint | 0.98 | true | 10000 | 426 | 593 | true | 0.0426 | 0.0387255, 0.0467436 | null | 36 |
| verification/C05/cap/0.985/A-F | 0.985 | true | 10000 | 168 | 317 | true | 0.0168 | 0.0143725, 0.0195144 | null | 36 |
| verification/C05/cap/0.985/P-F | 0.985 | true | 10000 | 173 | 317 | true | 0.0173 | 0.0148358, 0.0200508 | null | 36 |
| verification/C05/cap/0.985/joint | 0.985 | true | 10000 | 335 | 593 | true | 0.0335 | 0.0300597, 0.0372153 | null | 36 |
| verification/C05/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C05/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C06/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C06/36/0.980/A-F | 0.98 | true | 10000 | 243 | 317 | true | 0.0243 | 0.0213712, 0.0275101 | null | null |
| verification/C06/36/0.980/P-F | 0.98 | true | 10000 | 221 | 317 | true | 0.0221 | 0.0193085, 0.0251744 | null | null |
| verification/C06/36/0.980/joint | 0.98 | true | 10000 | 455 | 593 | true | 0.0455 | 0.0414984, 0.0497689 | null | null |
| verification/C06/36/0.985/A-F | 0.985 | true | 10000 | 153 | 317 | true | 0.0153 | 0.0129862, 0.0179019 | null | null |
| verification/C06/36/0.985/P-F | 0.985 | true | 10000 | 169 | 317 | true | 0.0169 | 0.0144651, 0.0196217 | null | null |
| verification/C06/36/0.985/joint | 0.985 | true | 10000 | 317 | 593 | true | 0.0317 | 0.0283531, 0.0353231 | null | null |
| verification/C06/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C06/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C06/45/0.980/A-F | 0.98 | true | 10000 | 245 | 317 | true | 0.0245 | 0.0215591, 0.0277221 | null | null |
| verification/C06/45/0.980/P-F | 0.98 | true | 10000 | 231 | 317 | true | 0.0231 | 0.0202453, 0.0262369 | null | null |
| verification/C06/45/0.980/joint | 0.98 | true | 10000 | 471 | 593 | true | 0.0471 | 0.0430302, 0.0514361 | null | null |
| verification/C06/45/0.985/A-F | 0.985 | true | 10000 | 164 | 317 | true | 0.0164 | 0.0140023, 0.019085 | null | null |
| verification/C06/45/0.985/P-F | 0.985 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| verification/C06/45/0.985/joint | 0.985 | true | 10000 | 313 | 593 | true | 0.0313 | 0.0279742, 0.0349023 | null | null |
| verification/C06/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C06/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C06/60/0.980/A-F | 0.98 | true | 10000 | 221 | 317 | true | 0.0221 | 0.0193085, 0.0251744 | null | null |
| verification/C06/60/0.980/P-F | 0.98 | true | 10000 | 208 | 317 | true | 0.0208 | 0.0180931, 0.0237908 | null | null |
| verification/C06/60/0.980/joint | 0.98 | true | 10000 | 421 | 593 | true | 0.0421 | 0.0382479, 0.0462215 | null | null |
| verification/C06/60/0.985/A-F | 0.985 | true | 10000 | 147 | 317 | true | 0.0147 | 0.0124334, 0.0172553 | null | null |
| verification/C06/60/0.985/P-F | 0.985 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | null |
| verification/C06/60/0.985/joint | 0.985 | true | 10000 | 292 | 593 | true | 0.0292 | 0.0259878, 0.0326901 | null | null |
| verification/C06/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C06/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C06/75/0.980/A-F | 0.98 | true | 10000 | 187 | 317 | true | 0.0187 | 0.0161359, 0.0215496 | null | null |
| verification/C06/75/0.980/P-F | 0.98 | true | 10000 | 208 | 317 | true | 0.0208 | 0.0180931, 0.0237908 | null | null |
| verification/C06/75/0.980/joint | 0.98 | true | 10000 | 389 | 593 | true | 0.0389 | 0.0351951, 0.0428764 | null | null |
| verification/C06/75/0.985/A-F | 0.985 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | null |
| verification/C06/75/0.985/P-F | 0.985 | true | 10000 | 165 | 317 | true | 0.0165 | 0.0140948, 0.0191924 | null | null |
| verification/C06/75/0.985/joint | 0.985 | true | 10000 | 309 | 593 | true | 0.0309 | 0.0275955, 0.0344812 | null | null |
| verification/C06/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C06/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C06/90/0.980/A-F | 0.98 | true | 10000 | 211 | 317 | true | 0.0211 | 0.0183733, 0.0241103 | null | null |
| verification/C06/90/0.980/P-F | 0.98 | true | 10000 | 223 | 317 | true | 0.0223 | 0.0194957, 0.025387 | null | null |
| verification/C06/90/0.980/joint | 0.98 | true | 10000 | 427 | 593 | true | 0.0427 | 0.0388211, 0.046848 | null | null |
| verification/C06/90/0.985/A-F | 0.985 | true | 10000 | 155 | 317 | true | 0.0155 | 0.0131707, 0.0181173 | null | null |
| verification/C06/90/0.985/P-F | 0.985 | true | 10000 | 155 | 317 | true | 0.0155 | 0.0131707, 0.0181173 | null | null |
| verification/C06/90/0.985/joint | 0.985 | true | 10000 | 306 | 593 | true | 0.0306 | 0.0273116, 0.0341654 | null | null |
| verification/C06/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C06/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C06/99/0.980/A-F | 0.98 | true | 10000 | 206 | 317 | true | 0.0206 | 0.0179063, 0.0235777 | null | null |
| verification/C06/99/0.980/P-F | 0.98 | true | 10000 | 203 | 317 | true | 0.0203 | 0.0176264, 0.0232579 | null | null |
| verification/C06/99/0.980/joint | 0.98 | true | 10000 | 403 | 593 | true | 0.0403 | 0.0365299, 0.0443407 | null | null |
| verification/C06/99/0.985/A-F | 0.985 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| verification/C06/99/0.985/P-F | 0.985 | true | 10000 | 171 | 317 | true | 0.0171 | 0.0146504, 0.0198363 | null | null |
| verification/C06/99/0.985/joint | 0.985 | true | 10000 | 316 | 593 | true | 0.0316 | 0.0282584, 0.0352179 | null | null |
| verification/C06/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C06/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C06/cap/0.980/A-F | 0.98 | true | 10000 | 198 | 317 | true | 0.0198 | 0.0171601, 0.0227245 | null | 36 |
| verification/C06/cap/0.980/P-F | 0.98 | true | 10000 | 205 | 317 | true | 0.0205 | 0.017813, 0.0234711 | null | 36 |
| verification/C06/cap/0.980/joint | 0.98 | true | 10000 | 396 | 593 | true | 0.0396 | 0.0358624, 0.0436087 | null | 36 |
| verification/C06/cap/0.985/A-F | 0.985 | true | 10000 | 141 | 317 | true | 0.0141 | 0.0118815, 0.0166077 | null | 36 |
| verification/C06/cap/0.985/P-F | 0.985 | true | 10000 | 153 | 317 | true | 0.0153 | 0.0129862, 0.0179019 | null | 36 |
| verification/C06/cap/0.985/joint | 0.985 | true | 10000 | 289 | 593 | true | 0.0289 | 0.0257044, 0.0323738 | null | 36 |
| verification/C06/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C06/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C07/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C07/36/0.980/A-F | 0.98 | true | 10000 | 348 | 317 | false | 0.0348 | 0.031294, 0.0385802 | null | null |
| verification/C07/36/0.980/P-F | 0.98 | true | 10000 | 300 | 317 | true | 0.03 | 0.026744, 0.0335334 | null | null |
| verification/C07/36/0.980/joint | 0.98 | true | 10000 | 630 | 593 | false | 0.063 | 0.0583158, 0.0679406 | null | null |
| verification/C07/36/0.985/A-F | 0.985 | true | 10000 | 244 | 317 | true | 0.0244 | 0.0214652, 0.0276161 | null | null |
| verification/C07/36/0.985/P-F | 0.985 | true | 10000 | 244 | 317 | true | 0.0244 | 0.0214652, 0.0276161 | null | null |
| verification/C07/36/0.985/joint | 0.985 | true | 10000 | 482 | 593 | true | 0.0482 | 0.0440841, 0.0525815 | null | null |
| verification/C07/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C07/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C07/45/0.980/A-F | 0.98 | true | 10000 | 277 | 317 | true | 0.0277 | 0.0245716, 0.0311073 | null | null |
| verification/C07/45/0.980/P-F | 0.98 | true | 10000 | 244 | 317 | true | 0.0244 | 0.0214652, 0.0276161 | null | null |
| verification/C07/45/0.980/joint | 0.98 | true | 10000 | 508 | 593 | true | 0.0508 | 0.0465775, 0.0552865 | null | null |
| verification/C07/45/0.985/A-F | 0.985 | true | 10000 | 189 | 317 | true | 0.0189 | 0.0163219, 0.0217634 | null | null |
| verification/C07/45/0.985/P-F | 0.985 | true | 10000 | 220 | 317 | true | 0.022 | 0.0192149, 0.025068 | null | null |
| verification/C07/45/0.985/joint | 0.985 | true | 10000 | 402 | 593 | true | 0.0402 | 0.0364345, 0.0442361 | null | null |
| verification/C07/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C07/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C07/60/0.980/A-F | 0.98 | true | 10000 | 214 | 317 | true | 0.0214 | 0.0186537, 0.0244297 | null | null |
| verification/C07/60/0.980/P-F | 0.98 | true | 10000 | 200 | 317 | true | 0.02 | 0.0173465, 0.0229379 | null | null |
| verification/C07/60/0.980/joint | 0.98 | true | 10000 | 405 | 593 | true | 0.0405 | 0.0367207, 0.0445498 | null | null |
| verification/C07/60/0.985/A-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| verification/C07/60/0.985/P-F | 0.985 | true | 10000 | 181 | 317 | true | 0.0181 | 0.0155782, 0.0209077 | null | null |
| verification/C07/60/0.985/joint | 0.985 | true | 10000 | 336 | 593 | true | 0.0336 | 0.0301546, 0.0373204 | null | null |
| verification/C07/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C07/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C07/75/0.980/A-F | 0.98 | true | 10000 | 177 | 317 | true | 0.0177 | 0.0152068, 0.0204794 | null | null |
| verification/C07/75/0.980/P-F | 0.98 | true | 10000 | 182 | 317 | true | 0.0182 | 0.0156711, 0.0210148 | null | null |
| verification/C07/75/0.980/joint | 0.98 | true | 10000 | 349 | 593 | true | 0.0349 | 0.031389, 0.0386852 | null | null |
| verification/C07/75/0.985/A-F | 0.985 | true | 10000 | 128 | 317 | true | 0.0128 | 0.0106896, 0.0152008 | null | null |
| verification/C07/75/0.985/P-F | 0.985 | true | 10000 | 156 | 317 | true | 0.0156 | 0.013263, 0.0182249 | null | null |
| verification/C07/75/0.985/joint | 0.985 | true | 10000 | 280 | 593 | true | 0.028 | 0.0248547, 0.0314241 | null | null |
| verification/C07/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C07/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C07/90/0.980/A-F | 0.98 | true | 10000 | 184 | 317 | true | 0.0184 | 0.0158569, 0.0212288 | null | null |
| verification/C07/90/0.980/P-F | 0.98 | true | 10000 | 185 | 317 | true | 0.0185 | 0.0159499, 0.0213357 | null | null |
| verification/C07/90/0.980/joint | 0.98 | true | 10000 | 361 | 593 | true | 0.0361 | 0.0325296, 0.0399438 | null | null |
| verification/C07/90/0.985/A-F | 0.985 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | null |
| verification/C07/90/0.985/P-F | 0.985 | true | 10000 | 138 | 317 | true | 0.0138 | 0.011606, 0.0162835 | null | null |
| verification/C07/90/0.985/joint | 0.985 | true | 10000 | 278 | 593 | true | 0.0278 | 0.0246659, 0.0312129 | null | null |
| verification/C07/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C07/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C07/99/0.980/A-F | 0.98 | true | 10000 | 179 | 317 | true | 0.0179 | 0.0153925, 0.0206936 | null | null |
| verification/C07/99/0.980/P-F | 0.98 | true | 10000 | 180 | 317 | true | 0.018 | 0.0154853, 0.0208007 | null | null |
| verification/C07/99/0.980/joint | 0.98 | true | 10000 | 349 | 593 | true | 0.0349 | 0.031389, 0.0386852 | null | null |
| verification/C07/99/0.985/A-F | 0.985 | true | 10000 | 128 | 317 | true | 0.0128 | 0.0106896, 0.0152008 | null | null |
| verification/C07/99/0.985/P-F | 0.985 | true | 10000 | 138 | 317 | true | 0.0138 | 0.011606, 0.0162835 | null | null |
| verification/C07/99/0.985/joint | 0.985 | true | 10000 | 262 | 593 | true | 0.0262 | 0.023158, 0.029522 | null | null |
| verification/C07/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C07/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C07/cap/0.980/A-F | 0.98 | true | 10000 | 208 | 317 | true | 0.0208 | 0.0180931, 0.0237908 | null | 36 |
| verification/C07/cap/0.980/P-F | 0.98 | true | 10000 | 195 | 317 | true | 0.0195 | 0.0168805, 0.0224043 | null | 36 |
| verification/C07/cap/0.980/joint | 0.98 | true | 10000 | 397 | 593 | true | 0.0397 | 0.0359577, 0.0437133 | null | 36 |
| verification/C07/cap/0.985/A-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | 36 |
| verification/C07/cap/0.985/P-F | 0.985 | true | 10000 | 174 | 317 | true | 0.0174 | 0.0149285, 0.020158 | null | 36 |
| verification/C07/cap/0.985/joint | 0.985 | true | 10000 | 320 | 593 | true | 0.032 | 0.0286373, 0.0356387 | null | 36 |
| verification/C07/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C07/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C08/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C08/36/0.980/A-F | 0.98 | true | 10000 | 286 | 317 | true | 0.0286 | 0.025421, 0.0320573 | null | null |
| verification/C08/36/0.980/P-F | 0.98 | true | 10000 | 272 | 317 | true | 0.0272 | 0.0241001, 0.0305792 | null | null |
| verification/C08/36/0.980/joint | 0.98 | true | 10000 | 547 | 593 | true | 0.0547 | 0.0503234, 0.0593382 | null | null |
| verification/C08/36/0.985/A-F | 0.985 | true | 10000 | 295 | 317 | true | 0.0295 | 0.0262713, 0.0330064 | null | null |
| verification/C08/36/0.985/P-F | 0.985 | true | 10000 | 267 | 317 | true | 0.0267 | 0.0236289, 0.0300508 | null | null |
| verification/C08/36/0.985/joint | 0.985 | true | 10000 | 552 | 593 | true | 0.0552 | 0.0508041, 0.0598571 | null | null |
| verification/C08/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C08/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C08/45/0.980/A-F | 0.98 | true | 10000 | 235 | 317 | true | 0.0235 | 0.0206204, 0.0266615 | null | null |
| verification/C08/45/0.980/P-F | 0.98 | true | 10000 | 221 | 317 | true | 0.0221 | 0.0193085, 0.0251744 | null | null |
| verification/C08/45/0.980/joint | 0.98 | true | 10000 | 448 | 593 | true | 0.0448 | 0.0408286, 0.0490391 | null | null |
| verification/C08/45/0.985/A-F | 0.985 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| verification/C08/45/0.985/P-F | 0.985 | true | 10000 | 178 | 317 | true | 0.0178 | 0.0152996, 0.0205865 | null | null |
| verification/C08/45/0.985/joint | 0.985 | true | 10000 | 376 | 593 | true | 0.0376 | 0.0339568, 0.0415155 | null | null |
| verification/C08/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C08/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C08/60/0.980/A-F | 0.98 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| verification/C08/60/0.980/P-F | 0.98 | true | 10000 | 191 | 317 | true | 0.0191 | 0.016508, 0.0219771 | null | null |
| verification/C08/60/0.980/joint | 0.98 | true | 10000 | 373 | 593 | true | 0.0373 | 0.0336712, 0.0412013 | null | null |
| verification/C08/60/0.985/A-F | 0.985 | true | 10000 | 182 | 317 | true | 0.0182 | 0.0156711, 0.0210148 | null | null |
| verification/C08/60/0.985/P-F | 0.985 | true | 10000 | 132 | 317 | true | 0.0132 | 0.0110557, 0.0156343 | null | null |
| verification/C08/60/0.985/joint | 0.985 | true | 10000 | 311 | 593 | true | 0.0311 | 0.0277849, 0.0346918 | null | null |
| verification/C08/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C08/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C08/75/0.980/A-F | 0.98 | true | 10000 | 142 | 317 | true | 0.0142 | 0.0119734, 0.0167157 | null | null |
| verification/C08/75/0.980/P-F | 0.98 | true | 10000 | 143 | 317 | true | 0.0143 | 0.0120653, 0.0168237 | null | null |
| verification/C08/75/0.980/joint | 0.98 | true | 10000 | 281 | 593 | true | 0.0281 | 0.024949, 0.0315297 | null | null |
| verification/C08/75/0.985/A-F | 0.985 | true | 10000 | 120 | 317 | true | 0.012 | 0.00995893, 0.0143322 | null | null |
| verification/C08/75/0.985/P-F | 0.985 | true | 10000 | 95 | 317 | true | 0.0095 | 0.00769267, 0.011601 | null | null |
| verification/C08/75/0.985/joint | 0.985 | true | 10000 | 209 | 593 | true | 0.0209 | 0.0181865, 0.0238973 | null | null |
| verification/C08/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C08/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C08/90/0.980/A-F | 0.98 | true | 10000 | 122 | 317 | true | 0.0122 | 0.0101414, 0.0145495 | null | null |
| verification/C08/90/0.980/P-F | 0.98 | true | 10000 | 128 | 317 | true | 0.0128 | 0.0106896, 0.0152008 | null | null |
| verification/C08/90/0.980/joint | 0.98 | true | 10000 | 244 | 593 | true | 0.0244 | 0.0214652, 0.0276161 | null | null |
| verification/C08/90/0.985/A-F | 0.985 | true | 10000 | 98 | 317 | true | 0.0098 | 0.00796307, 0.0119302 | null | null |
| verification/C08/90/0.985/P-F | 0.985 | true | 10000 | 82 | 317 | true | 0.0082 | 0.00652686, 0.0101683 | null | null |
| verification/C08/90/0.985/joint | 0.985 | true | 10000 | 176 | 593 | true | 0.0176 | 0.015114, 0.0203723 | null | null |
| verification/C08/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C08/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C08/99/0.980/A-F | 0.98 | true | 10000 | 111 | 317 | true | 0.0111 | 0.00913989, 0.0133521 | null | null |
| verification/C08/99/0.980/P-F | 0.98 | true | 10000 | 115 | 317 | true | 0.0115 | 0.00950351, 0.0137881 | null | null |
| verification/C08/99/0.980/joint | 0.98 | true | 10000 | 219 | 593 | true | 0.0219 | 0.0191214, 0.0249617 | null | null |
| verification/C08/99/0.985/A-F | 0.985 | true | 10000 | 103 | 317 | true | 0.0103 | 0.00841475, 0.012478 | null | null |
| verification/C08/99/0.985/P-F | 0.985 | true | 10000 | 79 | 317 | true | 0.0079 | 0.00625935, 0.00983617 | null | null |
| verification/C08/99/0.985/joint | 0.985 | true | 10000 | 178 | 593 | true | 0.0178 | 0.0152996, 0.0205865 | null | null |
| verification/C08/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C08/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C08/cap/0.980/A-F | 0.98 | true | 10000 | 153 | 317 | true | 0.0153 | 0.0129862, 0.0179019 | null | 36 |
| verification/C08/cap/0.980/P-F | 0.98 | true | 10000 | 141 | 317 | true | 0.0141 | 0.0118815, 0.0166077 | null | 36 |
| verification/C08/cap/0.980/joint | 0.98 | true | 10000 | 289 | 593 | true | 0.0289 | 0.0257044, 0.0323738 | null | 36 |
| verification/C08/cap/0.985/A-F | 0.985 | true | 10000 | 140 | 317 | true | 0.014 | 0.0117896, 0.0164997 | null | 36 |
| verification/C08/cap/0.985/P-F | 0.985 | true | 10000 | 97 | 317 | true | 0.0097 | 0.00787289, 0.0118205 | null | 36 |
| verification/C08/cap/0.985/joint | 0.985 | true | 10000 | 234 | 593 | true | 0.0234 | 0.0205266, 0.0265554 | null | 36 |
| verification/C08/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C08/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C09/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C09/36/0.980/A-F | 0.98 | true | 10000 | 184 | 317 | true | 0.0184 | 0.0158569, 0.0212288 | null | null |
| verification/C09/36/0.980/P-F | 0.98 | true | 10000 | 194 | 317 | true | 0.0194 | 0.0167874, 0.0222975 | null | null |
| verification/C09/36/0.980/joint | 0.98 | true | 10000 | 367 | 593 | true | 0.0367 | 0.0331003, 0.0405727 | null | null |
| verification/C09/36/0.985/A-F | 0.985 | true | 10000 | 143 | 317 | true | 0.0143 | 0.0120653, 0.0168237 | null | null |
| verification/C09/36/0.985/P-F | 0.985 | true | 10000 | 123 | 317 | true | 0.0123 | 0.0102326, 0.0146582 | null | null |
| verification/C09/36/0.985/joint | 0.985 | true | 10000 | 258 | 593 | true | 0.0258 | 0.0227814, 0.0290988 | null | null |
| verification/C09/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C09/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C09/45/0.980/A-F | 0.98 | true | 10000 | 200 | 317 | true | 0.02 | 0.0173465, 0.0229379 | null | null |
| verification/C09/45/0.980/P-F | 0.98 | true | 10000 | 197 | 317 | true | 0.0197 | 0.0170669, 0.0226178 | null | null |
| verification/C09/45/0.980/joint | 0.98 | true | 10000 | 385 | 593 | true | 0.0385 | 0.034814, 0.0424578 | null | null |
| verification/C09/45/0.985/A-F | 0.985 | true | 10000 | 130 | 317 | true | 0.013 | 0.0108726, 0.0154176 | null | null |
| verification/C09/45/0.985/P-F | 0.985 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | null |
| verification/C09/45/0.985/joint | 0.985 | true | 10000 | 269 | 593 | true | 0.0269 | 0.0238173, 0.0302622 | null | null |
| verification/C09/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C09/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C09/60/0.980/A-F | 0.98 | true | 10000 | 195 | 317 | true | 0.0195 | 0.0168805, 0.0224043 | null | null |
| verification/C09/60/0.980/P-F | 0.98 | true | 10000 | 179 | 317 | true | 0.0179 | 0.0153925, 0.0206936 | null | null |
| verification/C09/60/0.980/joint | 0.98 | true | 10000 | 365 | 593 | true | 0.0365 | 0.03291, 0.0403631 | null | null |
| verification/C09/60/0.985/A-F | 0.985 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | null |
| verification/C09/60/0.985/P-F | 0.985 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | null |
| verification/C09/60/0.985/joint | 0.985 | true | 10000 | 300 | 593 | true | 0.03 | 0.026744, 0.0335334 | null | null |
| verification/C09/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C09/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C09/75/0.980/A-F | 0.98 | true | 10000 | 207 | 317 | true | 0.0207 | 0.0179997, 0.0236842 | null | null |
| verification/C09/75/0.980/P-F | 0.98 | true | 10000 | 186 | 317 | true | 0.0186 | 0.0160429, 0.0214427 | null | null |
| verification/C09/75/0.980/joint | 0.98 | true | 10000 | 387 | 593 | true | 0.0387 | 0.0350045, 0.0426671 | null | null |
| verification/C09/75/0.985/A-F | 0.985 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | null |
| verification/C09/75/0.985/P-F | 0.985 | true | 10000 | 141 | 317 | true | 0.0141 | 0.0118815, 0.0166077 | null | null |
| verification/C09/75/0.985/joint | 0.985 | true | 10000 | 291 | 593 | true | 0.0291 | 0.0258933, 0.0325847 | null | null |
| verification/C09/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C09/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C09/90/0.980/A-F | 0.98 | true | 10000 | 214 | 317 | true | 0.0214 | 0.0186537, 0.0244297 | null | null |
| verification/C09/90/0.980/P-F | 0.98 | true | 10000 | 196 | 317 | true | 0.0196 | 0.0169737, 0.0225111 | null | null |
| verification/C09/90/0.980/joint | 0.98 | true | 10000 | 403 | 593 | true | 0.0403 | 0.0365299, 0.0443407 | null | null |
| verification/C09/90/0.985/A-F | 0.985 | true | 10000 | 142 | 317 | true | 0.0142 | 0.0119734, 0.0167157 | null | null |
| verification/C09/90/0.985/P-F | 0.985 | true | 10000 | 138 | 317 | true | 0.0138 | 0.011606, 0.0162835 | null | null |
| verification/C09/90/0.985/joint | 0.985 | true | 10000 | 272 | 593 | true | 0.0272 | 0.0241001, 0.0305792 | null | null |
| verification/C09/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C09/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C09/99/0.980/A-F | 0.98 | true | 10000 | 208 | 317 | true | 0.0208 | 0.0180931, 0.0237908 | null | null |
| verification/C09/99/0.980/P-F | 0.98 | true | 10000 | 199 | 317 | true | 0.0199 | 0.0172533, 0.0228312 | null | null |
| verification/C09/99/0.980/joint | 0.98 | true | 10000 | 396 | 593 | true | 0.0396 | 0.0358624, 0.0436087 | null | null |
| verification/C09/99/0.985/A-F | 0.985 | true | 10000 | 139 | 317 | true | 0.0139 | 0.0116978, 0.0163916 | null | null |
| verification/C09/99/0.985/P-F | 0.985 | true | 10000 | 156 | 317 | true | 0.0156 | 0.013263, 0.0182249 | null | null |
| verification/C09/99/0.985/joint | 0.985 | true | 10000 | 288 | 593 | true | 0.0288 | 0.0256099, 0.0322683 | null | null |
| verification/C09/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C09/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C09/cap/0.980/A-F | 0.98 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | 36 |
| verification/C09/cap/0.980/P-F | 0.98 | true | 10000 | 194 | 317 | true | 0.0194 | 0.0167874, 0.0222975 | null | 36 |
| verification/C09/cap/0.980/joint | 0.98 | true | 10000 | 380 | 593 | true | 0.038 | 0.0343377, 0.0419344 | null | 36 |
| verification/C09/cap/0.985/A-F | 0.985 | true | 10000 | 153 | 317 | true | 0.0153 | 0.0129862, 0.0179019 | null | 36 |
| verification/C09/cap/0.985/P-F | 0.985 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | 36 |
| verification/C09/cap/0.985/joint | 0.985 | true | 10000 | 295 | 593 | true | 0.0295 | 0.0262713, 0.0330064 | null | 36 |
| verification/C09/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C09/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C10/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C10/36/0.980/A-F | 0.98 | true | 10000 | 196 | 317 | true | 0.0196 | 0.0169737, 0.0225111 | null | null |
| verification/C10/36/0.980/P-F | 0.98 | true | 10000 | 194 | 317 | true | 0.0194 | 0.0167874, 0.0222975 | null | null |
| verification/C10/36/0.980/joint | 0.98 | true | 10000 | 384 | 593 | true | 0.0384 | 0.0347187, 0.0423531 | null | null |
| verification/C10/36/0.985/A-F | 0.985 | true | 10000 | 144 | 317 | true | 0.0144 | 0.0121573, 0.0169316 | null | null |
| verification/C10/36/0.985/P-F | 0.985 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | null |
| verification/C10/36/0.985/joint | 0.985 | true | 10000 | 289 | 593 | true | 0.0289 | 0.0257044, 0.0323738 | null | null |
| verification/C10/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C10/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C10/45/0.980/A-F | 0.98 | true | 10000 | 213 | 317 | true | 0.0213 | 0.0185602, 0.0243233 | null | null |
| verification/C10/45/0.980/P-F | 0.98 | true | 10000 | 196 | 317 | true | 0.0196 | 0.0169737, 0.0225111 | null | null |
| verification/C10/45/0.980/joint | 0.98 | true | 10000 | 402 | 593 | true | 0.0402 | 0.0364345, 0.0442361 | null | null |
| verification/C10/45/0.985/A-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| verification/C10/45/0.985/P-F | 0.985 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | null |
| verification/C10/45/0.985/joint | 0.985 | true | 10000 | 311 | 593 | true | 0.0311 | 0.0277849, 0.0346918 | null | null |
| verification/C10/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C10/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C10/60/0.980/A-F | 0.98 | true | 10000 | 206 | 317 | true | 0.0206 | 0.0179063, 0.0235777 | null | null |
| verification/C10/60/0.980/P-F | 0.98 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| verification/C10/60/0.980/joint | 0.98 | true | 10000 | 390 | 593 | true | 0.039 | 0.0352904, 0.042981 | null | null |
| verification/C10/60/0.985/A-F | 0.985 | true | 10000 | 138 | 317 | true | 0.0138 | 0.011606, 0.0162835 | null | null |
| verification/C10/60/0.985/P-F | 0.985 | true | 10000 | 145 | 317 | true | 0.0145 | 0.0122493, 0.0170396 | null | null |
| verification/C10/60/0.985/joint | 0.985 | true | 10000 | 278 | 593 | true | 0.0278 | 0.0246659, 0.0312129 | null | null |
| verification/C10/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C10/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C10/75/0.980/A-F | 0.98 | true | 10000 | 199 | 317 | true | 0.0199 | 0.0172533, 0.0228312 | null | null |
| verification/C10/75/0.980/P-F | 0.98 | true | 10000 | 188 | 317 | true | 0.0188 | 0.0162289, 0.0216565 | null | null |
| verification/C10/75/0.980/joint | 0.98 | true | 10000 | 384 | 593 | true | 0.0384 | 0.0347187, 0.0423531 | null | null |
| verification/C10/75/0.985/A-F | 0.985 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| verification/C10/75/0.985/P-F | 0.985 | true | 10000 | 138 | 317 | true | 0.0138 | 0.011606, 0.0162835 | null | null |
| verification/C10/75/0.985/joint | 0.985 | true | 10000 | 286 | 593 | true | 0.0286 | 0.025421, 0.0320573 | null | null |
| verification/C10/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C10/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C10/90/0.980/A-F | 0.98 | true | 10000 | 213 | 317 | true | 0.0213 | 0.0185602, 0.0243233 | null | null |
| verification/C10/90/0.980/P-F | 0.98 | true | 10000 | 208 | 317 | true | 0.0208 | 0.0180931, 0.0237908 | null | null |
| verification/C10/90/0.980/joint | 0.98 | true | 10000 | 418 | 593 | true | 0.0418 | 0.0379615, 0.0459082 | null | null |
| verification/C10/90/0.985/A-F | 0.985 | true | 10000 | 137 | 317 | true | 0.0137 | 0.0115142, 0.0161754 | null | null |
| verification/C10/90/0.985/P-F | 0.985 | true | 10000 | 116 | 317 | true | 0.0116 | 0.00959451, 0.013897 | null | null |
| verification/C10/90/0.985/joint | 0.985 | true | 10000 | 250 | 593 | true | 0.025 | 0.022029, 0.0282519 | null | null |
| verification/C10/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C10/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C10/99/0.980/A-F | 0.98 | true | 10000 | 210 | 317 | true | 0.021 | 0.0182799, 0.0240038 | null | null |
| verification/C10/99/0.980/P-F | 0.98 | true | 10000 | 181 | 317 | true | 0.0181 | 0.0155782, 0.0209077 | null | null |
| verification/C10/99/0.980/joint | 0.98 | true | 10000 | 388 | 593 | true | 0.0388 | 0.0350998, 0.0427718 | null | null |
| verification/C10/99/0.985/A-F | 0.985 | true | 10000 | 143 | 317 | true | 0.0143 | 0.0120653, 0.0168237 | null | null |
| verification/C10/99/0.985/P-F | 0.985 | true | 10000 | 122 | 317 | true | 0.0122 | 0.0101414, 0.0145495 | null | null |
| verification/C10/99/0.985/joint | 0.985 | true | 10000 | 264 | 593 | true | 0.0264 | 0.0233463, 0.0297336 | null | null |
| verification/C10/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C10/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C10/cap/0.980/A-F | 0.98 | true | 10000 | 227 | 317 | true | 0.0227 | 0.0198704, 0.0258121 | null | 36 |
| verification/C10/cap/0.980/P-F | 0.98 | true | 10000 | 191 | 317 | true | 0.0191 | 0.016508, 0.0219771 | null | 36 |
| verification/C10/cap/0.980/joint | 0.98 | true | 10000 | 415 | 593 | true | 0.0415 | 0.037675, 0.0455948 | null | 36 |
| verification/C10/cap/0.985/A-F | 0.985 | true | 10000 | 145 | 317 | true | 0.0145 | 0.0122493, 0.0170396 | null | 36 |
| verification/C10/cap/0.985/P-F | 0.985 | true | 10000 | 125 | 317 | true | 0.0125 | 0.0104153, 0.0148753 | null | 36 |
| verification/C10/cap/0.985/joint | 0.985 | true | 10000 | 267 | 593 | true | 0.0267 | 0.0236289, 0.0300508 | null | 36 |
| verification/C10/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C10/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C11/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C11/36/0.980/A-F | 0.98 | true | 10000 | 197 | 317 | true | 0.0197 | 0.0170669, 0.0226178 | null | null |
| verification/C11/36/0.980/P-F | 0.98 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| verification/C11/36/0.980/joint | 0.98 | true | 10000 | 388 | 593 | true | 0.0388 | 0.0350998, 0.0427718 | null | null |
| verification/C11/36/0.985/A-F | 0.985 | true | 10000 | 143 | 317 | true | 0.0143 | 0.0120653, 0.0168237 | null | null |
| verification/C11/36/0.985/P-F | 0.985 | true | 10000 | 142 | 317 | true | 0.0142 | 0.0119734, 0.0167157 | null | null |
| verification/C11/36/0.985/joint | 0.985 | true | 10000 | 281 | 593 | true | 0.0281 | 0.024949, 0.0315297 | null | null |
| verification/C11/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C11/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C11/45/0.980/A-F | 0.98 | true | 10000 | 180 | 317 | true | 0.018 | 0.0154853, 0.0208007 | null | null |
| verification/C11/45/0.980/P-F | 0.98 | true | 10000 | 198 | 317 | true | 0.0198 | 0.0171601, 0.0227245 | null | null |
| verification/C11/45/0.980/joint | 0.98 | true | 10000 | 368 | 593 | true | 0.0368 | 0.0331954, 0.0406775 | null | null |
| verification/C11/45/0.985/A-F | 0.985 | true | 10000 | 147 | 317 | true | 0.0147 | 0.0124334, 0.0172553 | null | null |
| verification/C11/45/0.985/P-F | 0.985 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | null |
| verification/C11/45/0.985/joint | 0.985 | true | 10000 | 302 | 593 | true | 0.0302 | 0.0269332, 0.0337441 | null | null |
| verification/C11/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C11/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C11/60/0.980/A-F | 0.98 | true | 10000 | 193 | 317 | true | 0.0193 | 0.0166942, 0.0221908 | null | null |
| verification/C11/60/0.980/P-F | 0.98 | true | 10000 | 213 | 317 | true | 0.0213 | 0.0185602, 0.0243233 | null | null |
| verification/C11/60/0.980/joint | 0.98 | true | 10000 | 393 | 593 | true | 0.0393 | 0.0355764, 0.0432949 | null | null |
| verification/C11/60/0.985/A-F | 0.985 | true | 10000 | 148 | 317 | true | 0.0148 | 0.0125254, 0.0173632 | null | null |
| verification/C11/60/0.985/P-F | 0.985 | true | 10000 | 144 | 317 | true | 0.0144 | 0.0121573, 0.0169316 | null | null |
| verification/C11/60/0.985/joint | 0.985 | true | 10000 | 290 | 593 | true | 0.029 | 0.0257988, 0.0324792 | null | null |
| verification/C11/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C11/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C11/75/0.980/A-F | 0.98 | true | 10000 | 195 | 317 | true | 0.0195 | 0.0168805, 0.0224043 | null | null |
| verification/C11/75/0.980/P-F | 0.98 | true | 10000 | 209 | 317 | true | 0.0209 | 0.0181865, 0.0238973 | null | null |
| verification/C11/75/0.980/joint | 0.98 | true | 10000 | 391 | 593 | true | 0.0391 | 0.0353857, 0.0430857 | null | null |
| verification/C11/75/0.985/A-F | 0.985 | true | 10000 | 149 | 317 | true | 0.0149 | 0.0126175, 0.017471 | null | null |
| verification/C11/75/0.985/P-F | 0.985 | true | 10000 | 133 | 317 | true | 0.0133 | 0.0111474, 0.0157426 | null | null |
| verification/C11/75/0.985/joint | 0.985 | true | 10000 | 280 | 593 | true | 0.028 | 0.0248547, 0.0314241 | null | null |
| verification/C11/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C11/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C11/90/0.980/A-F | 0.98 | true | 10000 | 187 | 317 | true | 0.0187 | 0.0161359, 0.0215496 | null | null |
| verification/C11/90/0.980/P-F | 0.98 | true | 10000 | 205 | 317 | true | 0.0205 | 0.017813, 0.0234711 | null | null |
| verification/C11/90/0.980/joint | 0.98 | true | 10000 | 382 | 593 | true | 0.0382 | 0.0345282, 0.0421438 | null | null |
| verification/C11/90/0.985/A-F | 0.985 | true | 10000 | 156 | 317 | true | 0.0156 | 0.013263, 0.0182249 | null | null |
| verification/C11/90/0.985/P-F | 0.985 | true | 10000 | 140 | 317 | true | 0.014 | 0.0117896, 0.0164997 | null | null |
| verification/C11/90/0.985/joint | 0.985 | true | 10000 | 294 | 593 | true | 0.0294 | 0.0261768, 0.032901 | null | null |
| verification/C11/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C11/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C11/99/0.980/A-F | 0.98 | true | 10000 | 184 | 317 | true | 0.0184 | 0.0158569, 0.0212288 | null | null |
| verification/C11/99/0.980/P-F | 0.98 | true | 10000 | 224 | 317 | true | 0.0224 | 0.0195894, 0.0254933 | null | null |
| verification/C11/99/0.980/joint | 0.98 | true | 10000 | 399 | 593 | true | 0.0399 | 0.0361484, 0.0439225 | null | null |
| verification/C11/99/0.985/A-F | 0.985 | true | 10000 | 138 | 317 | true | 0.0138 | 0.011606, 0.0162835 | null | null |
| verification/C11/99/0.985/P-F | 0.985 | true | 10000 | 150 | 317 | true | 0.015 | 0.0127097, 0.0175788 | null | null |
| verification/C11/99/0.985/joint | 0.985 | true | 10000 | 285 | 593 | true | 0.0285 | 0.0253266, 0.0319518 | null | null |
| verification/C11/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C11/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C11/cap/0.980/A-F | 0.98 | true | 10000 | 191 | 317 | true | 0.0191 | 0.016508, 0.0219771 | null | 36 |
| verification/C11/cap/0.980/P-F | 0.98 | true | 10000 | 207 | 317 | true | 0.0207 | 0.0179997, 0.0236842 | null | 36 |
| verification/C11/cap/0.980/joint | 0.98 | true | 10000 | 387 | 593 | true | 0.0387 | 0.0350045, 0.0426671 | null | 36 |
| verification/C11/cap/0.985/A-F | 0.985 | true | 10000 | 144 | 317 | true | 0.0144 | 0.0121573, 0.0169316 | null | 36 |
| verification/C11/cap/0.985/P-F | 0.985 | true | 10000 | 155 | 317 | true | 0.0155 | 0.0131707, 0.0181173 | null | 36 |
| verification/C11/cap/0.985/joint | 0.985 | true | 10000 | 297 | 593 | true | 0.0297 | 0.0264603, 0.0332172 | null | 36 |
| verification/C11/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C11/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C12/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C12/36/0.980/A-F | 0.98 | true | 10000 | 235 | 317 | true | 0.0235 | 0.0206204, 0.0266615 | null | null |
| verification/C12/36/0.980/P-F | 0.98 | true | 10000 | 194 | 317 | true | 0.0194 | 0.0167874, 0.0222975 | null | null |
| verification/C12/36/0.980/joint | 0.98 | true | 10000 | 425 | 593 | true | 0.0425 | 0.03863, 0.0466392 | null | null |
| verification/C12/36/0.985/A-F | 0.985 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | null |
| verification/C12/36/0.985/P-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| verification/C12/36/0.985/joint | 0.985 | true | 10000 | 302 | 593 | true | 0.0302 | 0.0269332, 0.0337441 | null | null |
| verification/C12/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C12/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C12/45/0.980/A-F | 0.98 | true | 10000 | 225 | 317 | true | 0.0225 | 0.019683, 0.0255996 | null | null |
| verification/C12/45/0.980/P-F | 0.98 | true | 10000 | 204 | 317 | true | 0.0204 | 0.0177197, 0.0233645 | null | null |
| verification/C12/45/0.980/joint | 0.98 | true | 10000 | 424 | 593 | true | 0.0424 | 0.0385345, 0.0465348 | null | null |
| verification/C12/45/0.985/A-F | 0.985 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| verification/C12/45/0.985/P-F | 0.985 | true | 10000 | 145 | 317 | true | 0.0145 | 0.0122493, 0.0170396 | null | null |
| verification/C12/45/0.985/joint | 0.985 | true | 10000 | 290 | 593 | true | 0.029 | 0.0257988, 0.0324792 | null | null |
| verification/C12/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C12/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C12/60/0.980/A-F | 0.98 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| verification/C12/60/0.980/P-F | 0.98 | true | 10000 | 214 | 317 | true | 0.0214 | 0.0186537, 0.0244297 | null | null |
| verification/C12/60/0.980/joint | 0.98 | true | 10000 | 407 | 593 | true | 0.0407 | 0.0369115, 0.0447588 | null | null |
| verification/C12/60/0.985/A-F | 0.985 | true | 10000 | 154 | 317 | true | 0.0154 | 0.0130785, 0.0180096 | null | null |
| verification/C12/60/0.985/P-F | 0.985 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | null |
| verification/C12/60/0.985/joint | 0.985 | true | 10000 | 293 | 593 | true | 0.0293 | 0.0260823, 0.0327956 | null | null |
| verification/C12/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C12/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C12/75/0.980/A-F | 0.98 | true | 10000 | 210 | 317 | true | 0.021 | 0.0182799, 0.0240038 | null | null |
| verification/C12/75/0.980/P-F | 0.98 | true | 10000 | 176 | 317 | true | 0.0176 | 0.015114, 0.0203723 | null | null |
| verification/C12/75/0.980/joint | 0.98 | true | 10000 | 382 | 593 | true | 0.0382 | 0.0345282, 0.0421438 | null | null |
| verification/C12/75/0.985/A-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| verification/C12/75/0.985/P-F | 0.985 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| verification/C12/75/0.985/joint | 0.985 | true | 10000 | 302 | 593 | true | 0.0302 | 0.0269332, 0.0337441 | null | null |
| verification/C12/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C12/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C12/90/0.980/A-F | 0.98 | true | 10000 | 209 | 317 | true | 0.0209 | 0.0181865, 0.0238973 | null | null |
| verification/C12/90/0.980/P-F | 0.98 | true | 10000 | 176 | 317 | true | 0.0176 | 0.015114, 0.0203723 | null | null |
| verification/C12/90/0.980/joint | 0.98 | true | 10000 | 380 | 593 | true | 0.038 | 0.0343377, 0.0419344 | null | null |
| verification/C12/90/0.985/A-F | 0.985 | true | 10000 | 144 | 317 | true | 0.0144 | 0.0121573, 0.0169316 | null | null |
| verification/C12/90/0.985/P-F | 0.985 | true | 10000 | 167 | 317 | true | 0.0167 | 0.0142799, 0.0194071 | null | null |
| verification/C12/90/0.985/joint | 0.985 | true | 10000 | 300 | 593 | true | 0.03 | 0.026744, 0.0335334 | null | null |
| verification/C12/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C12/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C12/99/0.980/A-F | 0.98 | true | 10000 | 214 | 317 | true | 0.0214 | 0.0186537, 0.0244297 | null | null |
| verification/C12/99/0.980/P-F | 0.98 | true | 10000 | 183 | 317 | true | 0.0183 | 0.015764, 0.0211218 | null | null |
| verification/C12/99/0.980/joint | 0.98 | true | 10000 | 390 | 593 | true | 0.039 | 0.0352904, 0.042981 | null | null |
| verification/C12/99/0.985/A-F | 0.985 | true | 10000 | 153 | 317 | true | 0.0153 | 0.0129862, 0.0179019 | null | null |
| verification/C12/99/0.985/P-F | 0.985 | true | 10000 | 169 | 317 | true | 0.0169 | 0.0144651, 0.0196217 | null | null |
| verification/C12/99/0.985/joint | 0.985 | true | 10000 | 309 | 593 | true | 0.0309 | 0.0275955, 0.0344812 | null | null |
| verification/C12/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C12/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C12/cap/0.980/A-F | 0.98 | true | 10000 | 200 | 317 | true | 0.02 | 0.0173465, 0.0229379 | null | 36 |
| verification/C12/cap/0.980/P-F | 0.98 | true | 10000 | 180 | 317 | true | 0.018 | 0.0154853, 0.0208007 | null | 36 |
| verification/C12/cap/0.980/joint | 0.98 | true | 10000 | 373 | 593 | true | 0.0373 | 0.0336712, 0.0412013 | null | 36 |
| verification/C12/cap/0.985/A-F | 0.985 | true | 10000 | 149 | 317 | true | 0.0149 | 0.0126175, 0.017471 | null | 36 |
| verification/C12/cap/0.985/P-F | 0.985 | true | 10000 | 146 | 317 | true | 0.0146 | 0.0123413, 0.0171475 | null | 36 |
| verification/C12/cap/0.985/joint | 0.985 | true | 10000 | 288 | 593 | true | 0.0288 | 0.0256099, 0.0322683 | null | 36 |
| verification/C12/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C12/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C13/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C13/36/0.980/A-F | 0.98 | true | 10000 | 207 | 317 | true | 0.0207 | 0.0179997, 0.0236842 | null | null |
| verification/C13/36/0.980/P-F | 0.98 | true | 10000 | 182 | 317 | true | 0.0182 | 0.0156711, 0.0210148 | null | null |
| verification/C13/36/0.980/joint | 0.98 | true | 10000 | 379 | 593 | true | 0.0379 | 0.0342425, 0.0418297 | null | null |
| verification/C13/36/0.985/A-F | 0.985 | true | 10000 | 148 | 317 | true | 0.0148 | 0.0125254, 0.0173632 | null | null |
| verification/C13/36/0.985/P-F | 0.985 | true | 10000 | 144 | 317 | true | 0.0144 | 0.0121573, 0.0169316 | null | null |
| verification/C13/36/0.985/joint | 0.985 | true | 10000 | 290 | 593 | true | 0.029 | 0.0257988, 0.0324792 | null | null |
| verification/C13/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C13/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C13/45/0.980/A-F | 0.98 | true | 10000 | 197 | 317 | true | 0.0197 | 0.0170669, 0.0226178 | null | null |
| verification/C13/45/0.980/P-F | 0.98 | true | 10000 | 180 | 317 | true | 0.018 | 0.0154853, 0.0208007 | null | null |
| verification/C13/45/0.980/joint | 0.98 | true | 10000 | 360 | 593 | true | 0.036 | 0.0324345, 0.0398389 | null | null |
| verification/C13/45/0.985/A-F | 0.985 | true | 10000 | 161 | 317 | true | 0.0161 | 0.0137249, 0.0187626 | null | null |
| verification/C13/45/0.985/P-F | 0.985 | true | 10000 | 162 | 317 | true | 0.0162 | 0.0138173, 0.0188701 | null | null |
| verification/C13/45/0.985/joint | 0.985 | true | 10000 | 319 | 593 | true | 0.0319 | 0.0285426, 0.0355335 | null | null |
| verification/C13/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C13/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C13/60/0.980/A-F | 0.98 | true | 10000 | 229 | 317 | true | 0.0229 | 0.0200578, 0.0260245 | null | null |
| verification/C13/60/0.980/P-F | 0.98 | true | 10000 | 230 | 317 | true | 0.023 | 0.0201515, 0.0261307 | null | null |
| verification/C13/60/0.980/joint | 0.98 | true | 10000 | 447 | 593 | true | 0.0447 | 0.040733, 0.0489348 | null | null |
| verification/C13/60/0.985/A-F | 0.985 | true | 10000 | 149 | 317 | true | 0.0149 | 0.0126175, 0.017471 | null | null |
| verification/C13/60/0.985/P-F | 0.985 | true | 10000 | 145 | 317 | true | 0.0145 | 0.0122493, 0.0170396 | null | null |
| verification/C13/60/0.985/joint | 0.985 | true | 10000 | 291 | 593 | true | 0.0291 | 0.0258933, 0.0325847 | null | null |
| verification/C13/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C13/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C13/75/0.980/A-F | 0.98 | true | 10000 | 213 | 317 | true | 0.0213 | 0.0185602, 0.0243233 | null | null |
| verification/C13/75/0.980/P-F | 0.98 | true | 10000 | 212 | 317 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| verification/C13/75/0.980/joint | 0.98 | true | 10000 | 410 | 593 | true | 0.041 | 0.0371978, 0.0450724 | null | null |
| verification/C13/75/0.985/A-F | 0.985 | true | 10000 | 139 | 317 | true | 0.0139 | 0.0116978, 0.0163916 | null | null |
| verification/C13/75/0.985/P-F | 0.985 | true | 10000 | 143 | 317 | true | 0.0143 | 0.0120653, 0.0168237 | null | null |
| verification/C13/75/0.985/joint | 0.985 | true | 10000 | 277 | 593 | true | 0.0277 | 0.0245716, 0.0311073 | null | null |
| verification/C13/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C13/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C13/90/0.980/A-F | 0.98 | true | 10000 | 205 | 317 | true | 0.0205 | 0.017813, 0.0234711 | null | null |
| verification/C13/90/0.980/P-F | 0.98 | true | 10000 | 201 | 317 | true | 0.0201 | 0.0174398, 0.0230446 | null | null |
| verification/C13/90/0.980/joint | 0.98 | true | 10000 | 389 | 593 | true | 0.0389 | 0.0351951, 0.0428764 | null | null |
| verification/C13/90/0.985/A-F | 0.985 | true | 10000 | 143 | 317 | true | 0.0143 | 0.0120653, 0.0168237 | null | null |
| verification/C13/90/0.985/P-F | 0.985 | true | 10000 | 143 | 317 | true | 0.0143 | 0.0120653, 0.0168237 | null | null |
| verification/C13/90/0.985/joint | 0.985 | true | 10000 | 284 | 593 | true | 0.0284 | 0.0252322, 0.0318463 | null | null |
| verification/C13/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C13/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C13/99/0.980/A-F | 0.98 | true | 10000 | 201 | 317 | true | 0.0201 | 0.0174398, 0.0230446 | null | null |
| verification/C13/99/0.980/P-F | 0.98 | true | 10000 | 230 | 317 | true | 0.023 | 0.0201515, 0.0261307 | null | null |
| verification/C13/99/0.980/joint | 0.98 | true | 10000 | 416 | 593 | true | 0.0416 | 0.0377705, 0.0456993 | null | null |
| verification/C13/99/0.985/A-F | 0.985 | true | 10000 | 154 | 317 | true | 0.0154 | 0.0130785, 0.0180096 | null | null |
| verification/C13/99/0.985/P-F | 0.985 | true | 10000 | 136 | 317 | true | 0.0136 | 0.0114224, 0.0160672 | null | null |
| verification/C13/99/0.985/joint | 0.985 | true | 10000 | 284 | 593 | true | 0.0284 | 0.0252322, 0.0318463 | null | null |
| verification/C13/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C13/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C13/cap/0.980/A-F | 0.98 | true | 10000 | 197 | 317 | true | 0.0197 | 0.0170669, 0.0226178 | null | 36 |
| verification/C13/cap/0.980/P-F | 0.98 | true | 10000 | 205 | 317 | true | 0.0205 | 0.017813, 0.0234711 | null | 36 |
| verification/C13/cap/0.980/joint | 0.98 | true | 10000 | 388 | 593 | true | 0.0388 | 0.0350998, 0.0427718 | null | 36 |
| verification/C13/cap/0.985/A-F | 0.985 | true | 10000 | 136 | 317 | true | 0.0136 | 0.0114224, 0.0160672 | null | 36 |
| verification/C13/cap/0.985/P-F | 0.985 | true | 10000 | 141 | 317 | true | 0.0141 | 0.0118815, 0.0166077 | null | 36 |
| verification/C13/cap/0.985/joint | 0.985 | true | 10000 | 275 | 593 | true | 0.0275 | 0.024383, 0.0308961 | null | 36 |
| verification/C13/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C13/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C14/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C14/36/0.980/A-F | 0.98 | true | 10000 | 181 | 317 | true | 0.0181 | 0.0155782, 0.0209077 | null | null |
| verification/C14/36/0.980/P-F | 0.98 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| verification/C14/36/0.980/joint | 0.98 | true | 10000 | 378 | 593 | true | 0.0378 | 0.0341473, 0.041725 | null | null |
| verification/C14/36/0.985/A-F | 0.985 | true | 10000 | 148 | 317 | true | 0.0148 | 0.0125254, 0.0173632 | null | null |
| verification/C14/36/0.985/P-F | 0.985 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| verification/C14/36/0.985/joint | 0.985 | true | 10000 | 296 | 593 | true | 0.0296 | 0.0263658, 0.0331118 | null | null |
| verification/C14/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C14/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C14/45/0.980/A-F | 0.98 | true | 10000 | 176 | 317 | true | 0.0176 | 0.015114, 0.0203723 | null | null |
| verification/C14/45/0.980/P-F | 0.98 | true | 10000 | 205 | 317 | true | 0.0205 | 0.017813, 0.0234711 | null | null |
| verification/C14/45/0.980/joint | 0.98 | true | 10000 | 377 | 593 | true | 0.0377 | 0.034052, 0.0416202 | null | null |
| verification/C14/45/0.985/A-F | 0.985 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | null |
| verification/C14/45/0.985/P-F | 0.985 | true | 10000 | 139 | 317 | true | 0.0139 | 0.0116978, 0.0163916 | null | null |
| verification/C14/45/0.985/joint | 0.985 | true | 10000 | 294 | 593 | true | 0.0294 | 0.0261768, 0.032901 | null | null |
| verification/C14/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C14/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C14/60/0.980/A-F | 0.98 | true | 10000 | 183 | 317 | true | 0.0183 | 0.015764, 0.0211218 | null | null |
| verification/C14/60/0.980/P-F | 0.98 | true | 10000 | 199 | 317 | true | 0.0199 | 0.0172533, 0.0228312 | null | null |
| verification/C14/60/0.980/joint | 0.98 | true | 10000 | 379 | 593 | true | 0.0379 | 0.0342425, 0.0418297 | null | null |
| verification/C14/60/0.985/A-F | 0.985 | true | 10000 | 154 | 317 | true | 0.0154 | 0.0130785, 0.0180096 | null | null |
| verification/C14/60/0.985/P-F | 0.985 | true | 10000 | 140 | 317 | true | 0.014 | 0.0117896, 0.0164997 | null | null |
| verification/C14/60/0.985/joint | 0.985 | true | 10000 | 291 | 593 | true | 0.0291 | 0.0258933, 0.0325847 | null | null |
| verification/C14/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C14/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C14/75/0.980/A-F | 0.98 | true | 10000 | 184 | 317 | true | 0.0184 | 0.0158569, 0.0212288 | null | null |
| verification/C14/75/0.980/P-F | 0.98 | true | 10000 | 197 | 317 | true | 0.0197 | 0.0170669, 0.0226178 | null | null |
| verification/C14/75/0.980/joint | 0.98 | true | 10000 | 376 | 593 | true | 0.0376 | 0.0339568, 0.0415155 | null | null |
| verification/C14/75/0.985/A-F | 0.985 | true | 10000 | 149 | 317 | true | 0.0149 | 0.0126175, 0.017471 | null | null |
| verification/C14/75/0.985/P-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | null |
| verification/C14/75/0.985/joint | 0.985 | true | 10000 | 304 | 593 | true | 0.0304 | 0.0271224, 0.0339547 | null | null |
| verification/C14/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C14/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C14/90/0.980/A-F | 0.98 | true | 10000 | 184 | 317 | true | 0.0184 | 0.0158569, 0.0212288 | null | null |
| verification/C14/90/0.980/P-F | 0.98 | true | 10000 | 193 | 317 | true | 0.0193 | 0.0166942, 0.0221908 | null | null |
| verification/C14/90/0.980/joint | 0.98 | true | 10000 | 368 | 593 | true | 0.0368 | 0.0331954, 0.0406775 | null | null |
| verification/C14/90/0.985/A-F | 0.985 | true | 10000 | 142 | 317 | true | 0.0142 | 0.0119734, 0.0167157 | null | null |
| verification/C14/90/0.985/P-F | 0.985 | true | 10000 | 162 | 317 | true | 0.0162 | 0.0138173, 0.0188701 | null | null |
| verification/C14/90/0.985/joint | 0.985 | true | 10000 | 300 | 593 | true | 0.03 | 0.026744, 0.0335334 | null | null |
| verification/C14/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C14/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C14/99/0.980/A-F | 0.98 | true | 10000 | 184 | 317 | true | 0.0184 | 0.0158569, 0.0212288 | null | null |
| verification/C14/99/0.980/P-F | 0.98 | true | 10000 | 188 | 317 | true | 0.0188 | 0.0162289, 0.0216565 | null | null |
| verification/C14/99/0.980/joint | 0.98 | true | 10000 | 367 | 593 | true | 0.0367 | 0.0331003, 0.0405727 | null | null |
| verification/C14/99/0.985/A-F | 0.985 | true | 10000 | 144 | 317 | true | 0.0144 | 0.0121573, 0.0169316 | null | null |
| verification/C14/99/0.985/P-F | 0.985 | true | 10000 | 162 | 317 | true | 0.0162 | 0.0138173, 0.0188701 | null | null |
| verification/C14/99/0.985/joint | 0.985 | true | 10000 | 301 | 593 | true | 0.0301 | 0.0268386, 0.0336387 | null | null |
| verification/C14/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C14/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C14/cap/0.980/A-F | 0.98 | true | 10000 | 177 | 317 | true | 0.0177 | 0.0152068, 0.0204794 | null | 36 |
| verification/C14/cap/0.980/P-F | 0.98 | true | 10000 | 200 | 317 | true | 0.02 | 0.0173465, 0.0229379 | null | 36 |
| verification/C14/cap/0.980/joint | 0.98 | true | 10000 | 371 | 593 | true | 0.0371 | 0.0334809, 0.0409918 | null | 36 |
| verification/C14/cap/0.985/A-F | 0.985 | true | 10000 | 145 | 317 | true | 0.0145 | 0.0122493, 0.0170396 | null | 36 |
| verification/C14/cap/0.985/P-F | 0.985 | true | 10000 | 176 | 317 | true | 0.0176 | 0.015114, 0.0203723 | null | 36 |
| verification/C14/cap/0.985/joint | 0.985 | true | 10000 | 317 | 593 | true | 0.0317 | 0.0283531, 0.0353231 | null | 36 |
| verification/C14/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C14/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C15/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C15/36/0.980/A-F | 0.98 | true | 10000 | 213 | 317 | true | 0.0213 | 0.0185602, 0.0243233 | null | null |
| verification/C15/36/0.980/P-F | 0.98 | true | 10000 | 202 | 317 | true | 0.0202 | 0.0175331, 0.0231512 | null | null |
| verification/C15/36/0.980/joint | 0.98 | true | 10000 | 412 | 593 | true | 0.0412 | 0.0373887, 0.0452814 | null | null |
| verification/C15/36/0.985/A-F | 0.985 | true | 10000 | 164 | 317 | true | 0.0164 | 0.0140023, 0.019085 | null | null |
| verification/C15/36/0.985/P-F | 0.985 | true | 10000 | 152 | 317 | true | 0.0152 | 0.012894, 0.0177942 | null | null |
| verification/C15/36/0.985/joint | 0.985 | true | 10000 | 310 | 593 | true | 0.031 | 0.0276902, 0.0345865 | null | null |
| verification/C15/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C15/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C15/45/0.980/A-F | 0.98 | true | 10000 | 216 | 317 | true | 0.0216 | 0.0188407, 0.0246425 | null | null |
| verification/C15/45/0.980/P-F | 0.98 | true | 10000 | 203 | 317 | true | 0.0203 | 0.0176264, 0.0232579 | null | null |
| verification/C15/45/0.980/joint | 0.98 | true | 10000 | 415 | 593 | true | 0.0415 | 0.037675, 0.0455948 | null | null |
| verification/C15/45/0.985/A-F | 0.985 | true | 10000 | 172 | 317 | true | 0.0172 | 0.0147431, 0.0199435 | null | null |
| verification/C15/45/0.985/P-F | 0.985 | true | 10000 | 147 | 317 | true | 0.0147 | 0.0124334, 0.0172553 | null | null |
| verification/C15/45/0.985/joint | 0.985 | true | 10000 | 318 | 593 | true | 0.0318 | 0.0284478, 0.0354283 | null | null |
| verification/C15/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C15/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C15/60/0.980/A-F | 0.98 | true | 10000 | 209 | 317 | true | 0.0209 | 0.0181865, 0.0238973 | null | null |
| verification/C15/60/0.980/P-F | 0.98 | true | 10000 | 177 | 317 | true | 0.0177 | 0.0152068, 0.0204794 | null | null |
| verification/C15/60/0.980/joint | 0.98 | true | 10000 | 383 | 593 | true | 0.0383 | 0.0346235, 0.0422484 | null | null |
| verification/C15/60/0.985/A-F | 0.985 | true | 10000 | 173 | 317 | true | 0.0173 | 0.0148358, 0.0200508 | null | null |
| verification/C15/60/0.985/P-F | 0.985 | true | 10000 | 141 | 317 | true | 0.0141 | 0.0118815, 0.0166077 | null | null |
| verification/C15/60/0.985/joint | 0.985 | true | 10000 | 308 | 593 | true | 0.0308 | 0.0275009, 0.034376 | null | null |
| verification/C15/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C15/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C15/75/0.980/A-F | 0.98 | true | 10000 | 201 | 317 | true | 0.0201 | 0.0174398, 0.0230446 | null | null |
| verification/C15/75/0.980/P-F | 0.98 | true | 10000 | 181 | 317 | true | 0.0181 | 0.0155782, 0.0209077 | null | null |
| verification/C15/75/0.980/joint | 0.98 | true | 10000 | 379 | 593 | true | 0.0379 | 0.0342425, 0.0418297 | null | null |
| verification/C15/75/0.985/A-F | 0.985 | true | 10000 | 157 | 317 | true | 0.0157 | 0.0133554, 0.0183325 | null | null |
| verification/C15/75/0.985/P-F | 0.985 | true | 10000 | 147 | 317 | true | 0.0147 | 0.0124334, 0.0172553 | null | null |
| verification/C15/75/0.985/joint | 0.985 | true | 10000 | 299 | 593 | true | 0.0299 | 0.0266494, 0.033428 | null | null |
| verification/C15/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C15/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C15/90/0.980/A-F | 0.98 | true | 10000 | 195 | 317 | true | 0.0195 | 0.0168805, 0.0224043 | null | null |
| verification/C15/90/0.980/P-F | 0.98 | true | 10000 | 168 | 317 | true | 0.0168 | 0.0143725, 0.0195144 | null | null |
| verification/C15/90/0.980/joint | 0.98 | true | 10000 | 358 | 593 | true | 0.0358 | 0.0322443, 0.0396292 | null | null |
| verification/C15/90/0.985/A-F | 0.985 | true | 10000 | 159 | 317 | true | 0.0159 | 0.0135401, 0.0185476 | null | null |
| verification/C15/90/0.985/P-F | 0.985 | true | 10000 | 154 | 317 | true | 0.0154 | 0.0130785, 0.0180096 | null | null |
| verification/C15/90/0.985/joint | 0.985 | true | 10000 | 310 | 593 | true | 0.031 | 0.0276902, 0.0345865 | null | null |
| verification/C15/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C15/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C15/99/0.980/A-F | 0.98 | true | 10000 | 196 | 317 | true | 0.0196 | 0.0169737, 0.0225111 | null | null |
| verification/C15/99/0.980/P-F | 0.98 | true | 10000 | 175 | 317 | true | 0.0175 | 0.0150213, 0.0202651 | null | null |
| verification/C15/99/0.980/joint | 0.98 | true | 10000 | 364 | 593 | true | 0.0364 | 0.0328149, 0.0402583 | null | null |
| verification/C15/99/0.985/A-F | 0.985 | true | 10000 | 158 | 317 | true | 0.0158 | 0.0134477, 0.0184401 | null | null |
| verification/C15/99/0.985/P-F | 0.985 | true | 10000 | 155 | 317 | true | 0.0155 | 0.0131707, 0.0181173 | null | null |
| verification/C15/99/0.985/joint | 0.985 | true | 10000 | 311 | 593 | true | 0.0311 | 0.0277849, 0.0346918 | null | null |
| verification/C15/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C15/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C15/cap/0.980/A-F | 0.98 | true | 10000 | 195 | 317 | true | 0.0195 | 0.0168805, 0.0224043 | null | 36 |
| verification/C15/cap/0.980/P-F | 0.98 | true | 10000 | 169 | 317 | true | 0.0169 | 0.0144651, 0.0196217 | null | 36 |
| verification/C15/cap/0.980/joint | 0.98 | true | 10000 | 358 | 593 | true | 0.0358 | 0.0322443, 0.0396292 | null | 36 |
| verification/C15/cap/0.985/A-F | 0.985 | true | 10000 | 173 | 317 | true | 0.0173 | 0.0148358, 0.0200508 | null | 36 |
| verification/C15/cap/0.985/P-F | 0.985 | true | 10000 | 160 | 317 | true | 0.016 | 0.0136325, 0.0186551 | null | 36 |
| verification/C15/cap/0.985/joint | 0.985 | true | 10000 | 329 | 593 | true | 0.0329 | 0.0294905, 0.0365849 | null | 36 |
| verification/C15/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C15/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C16/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C16/36/0.980/A-F | 0.98 | true | 10000 | 199 | 317 | true | 0.0199 | 0.0172533, 0.0228312 | null | null |
| verification/C16/36/0.980/P-F | 0.98 | true | 10000 | 198 | 317 | true | 0.0198 | 0.0171601, 0.0227245 | null | null |
| verification/C16/36/0.980/joint | 0.98 | true | 10000 | 386 | 593 | true | 0.0386 | 0.0349093, 0.0425625 | null | null |
| verification/C16/36/0.985/A-F | 0.985 | true | 10000 | 123 | 317 | true | 0.0123 | 0.0102326, 0.0146582 | null | null |
| verification/C16/36/0.985/P-F | 0.985 | true | 10000 | 164 | 317 | true | 0.0164 | 0.0140023, 0.019085 | null | null |
| verification/C16/36/0.985/joint | 0.985 | true | 10000 | 280 | 593 | true | 0.028 | 0.0248547, 0.0314241 | null | null |
| verification/C16/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C16/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C16/45/0.980/A-F | 0.98 | true | 10000 | 173 | 317 | true | 0.0173 | 0.0148358, 0.0200508 | null | null |
| verification/C16/45/0.980/P-F | 0.98 | true | 10000 | 207 | 317 | true | 0.0207 | 0.0179997, 0.0236842 | null | null |
| verification/C16/45/0.980/joint | 0.98 | true | 10000 | 370 | 593 | true | 0.037 | 0.0333857, 0.040887 | null | null |
| verification/C16/45/0.985/A-F | 0.985 | true | 10000 | 136 | 317 | true | 0.0136 | 0.0114224, 0.0160672 | null | null |
| verification/C16/45/0.985/P-F | 0.985 | true | 10000 | 153 | 317 | true | 0.0153 | 0.0129862, 0.0179019 | null | null |
| verification/C16/45/0.985/joint | 0.985 | true | 10000 | 284 | 593 | true | 0.0284 | 0.0252322, 0.0318463 | null | null |
| verification/C16/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C16/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C16/60/0.980/A-F | 0.98 | true | 10000 | 194 | 317 | true | 0.0194 | 0.0167874, 0.0222975 | null | null |
| verification/C16/60/0.980/P-F | 0.98 | true | 10000 | 216 | 317 | true | 0.0216 | 0.0188407, 0.0246425 | null | null |
| verification/C16/60/0.980/joint | 0.98 | true | 10000 | 401 | 593 | true | 0.0401 | 0.0363392, 0.0441316 | null | null |
| verification/C16/60/0.985/A-F | 0.985 | true | 10000 | 131 | 317 | true | 0.0131 | 0.0109641, 0.0155259 | null | null |
| verification/C16/60/0.985/P-F | 0.985 | true | 10000 | 124 | 317 | true | 0.0124 | 0.010324, 0.0147668 | null | null |
| verification/C16/60/0.985/joint | 0.985 | true | 10000 | 247 | 593 | true | 0.0247 | 0.021747, 0.027934 | null | null |
| verification/C16/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C16/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C16/75/0.980/A-F | 0.98 | true | 10000 | 201 | 317 | true | 0.0201 | 0.0174398, 0.0230446 | null | null |
| verification/C16/75/0.980/P-F | 0.98 | true | 10000 | 212 | 317 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| verification/C16/75/0.980/joint | 0.98 | true | 10000 | 400 | 593 | true | 0.04 | 0.0362438, 0.044027 | null | null |
| verification/C16/75/0.985/A-F | 0.985 | true | 10000 | 121 | 317 | true | 0.0121 | 0.0100501, 0.0144409 | null | null |
| verification/C16/75/0.985/P-F | 0.985 | true | 10000 | 135 | 317 | true | 0.0135 | 0.0113307, 0.015959 | null | null |
| verification/C16/75/0.985/joint | 0.985 | true | 10000 | 251 | 593 | true | 0.0251 | 0.022123, 0.0283578 | null | null |
| verification/C16/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C16/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C16/90/0.980/A-F | 0.98 | true | 10000 | 221 | 317 | true | 0.0221 | 0.0193085, 0.0251744 | null | null |
| verification/C16/90/0.980/P-F | 0.98 | true | 10000 | 218 | 317 | true | 0.0218 | 0.0190278, 0.0248553 | null | null |
| verification/C16/90/0.980/joint | 0.98 | true | 10000 | 425 | 593 | true | 0.0425 | 0.03863, 0.0466392 | null | null |
| verification/C16/90/0.985/A-F | 0.985 | true | 10000 | 115 | 317 | true | 0.0115 | 0.00950351, 0.0137881 | null | null |
| verification/C16/90/0.985/P-F | 0.985 | true | 10000 | 135 | 317 | true | 0.0135 | 0.0113307, 0.015959 | null | null |
| verification/C16/90/0.985/joint | 0.985 | true | 10000 | 246 | 593 | true | 0.0246 | 0.021653, 0.0278281 | null | null |
| verification/C16/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C16/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C16/99/0.980/A-F | 0.98 | true | 10000 | 212 | 317 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| verification/C16/99/0.980/P-F | 0.98 | true | 10000 | 209 | 317 | true | 0.0209 | 0.0181865, 0.0238973 | null | null |
| verification/C16/99/0.980/joint | 0.98 | true | 10000 | 412 | 593 | true | 0.0412 | 0.0373887, 0.0452814 | null | null |
| verification/C16/99/0.985/A-F | 0.985 | true | 10000 | 134 | 317 | true | 0.0134 | 0.011239, 0.0158508 | null | null |
| verification/C16/99/0.985/P-F | 0.985 | true | 10000 | 135 | 317 | true | 0.0135 | 0.0113307, 0.015959 | null | null |
| verification/C16/99/0.985/joint | 0.985 | true | 10000 | 264 | 593 | true | 0.0264 | 0.0233463, 0.0297336 | null | null |
| verification/C16/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C16/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C16/cap/0.980/A-F | 0.98 | true | 10000 | 194 | 317 | true | 0.0194 | 0.0167874, 0.0222975 | null | 36 |
| verification/C16/cap/0.980/P-F | 0.98 | true | 10000 | 216 | 317 | true | 0.0216 | 0.0188407, 0.0246425 | null | 36 |
| verification/C16/cap/0.980/joint | 0.98 | true | 10000 | 400 | 593 | true | 0.04 | 0.0362438, 0.044027 | null | 36 |
| verification/C16/cap/0.985/A-F | 0.985 | true | 10000 | 129 | 317 | true | 0.0129 | 0.0107811, 0.0153092 | null | 36 |
| verification/C16/cap/0.985/P-F | 0.985 | true | 10000 | 134 | 317 | true | 0.0134 | 0.011239, 0.0158508 | null | 36 |
| verification/C16/cap/0.985/joint | 0.985 | true | 10000 | 255 | 593 | true | 0.0255 | 0.0224992, 0.0287813 | null | 36 |
| verification/C16/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C16/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C17/36/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/36/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/36/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C17/36/0.980/A-F | 0.98 | true | 10000 | 201 | 317 | true | 0.0201 | 0.0174398, 0.0230446 | null | null |
| verification/C17/36/0.980/P-F | 0.98 | true | 10000 | 218 | 317 | true | 0.0218 | 0.0190278, 0.0248553 | null | null |
| verification/C17/36/0.980/joint | 0.98 | true | 10000 | 408 | 593 | true | 0.0408 | 0.0370069, 0.0448633 | null | null |
| verification/C17/36/0.985/A-F | 0.985 | true | 10000 | 148 | 317 | true | 0.0148 | 0.0125254, 0.0173632 | null | null |
| verification/C17/36/0.985/P-F | 0.985 | true | 10000 | 153 | 317 | true | 0.0153 | 0.0129862, 0.0179019 | null | null |
| verification/C17/36/0.985/joint | 0.985 | true | 10000 | 297 | 593 | true | 0.0297 | 0.0264603, 0.0332172 | null | null |
| verification/C17/36/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/36/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/36/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C17/45/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/45/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/45/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C17/45/0.980/A-F | 0.98 | true | 10000 | 206 | 317 | true | 0.0206 | 0.0179063, 0.0235777 | null | null |
| verification/C17/45/0.980/P-F | 0.98 | true | 10000 | 195 | 317 | true | 0.0195 | 0.0168805, 0.0224043 | null | null |
| verification/C17/45/0.980/joint | 0.98 | true | 10000 | 393 | 593 | true | 0.0393 | 0.0355764, 0.0432949 | null | null |
| verification/C17/45/0.985/A-F | 0.985 | true | 10000 | 144 | 317 | true | 0.0144 | 0.0121573, 0.0169316 | null | null |
| verification/C17/45/0.985/P-F | 0.985 | true | 10000 | 155 | 317 | true | 0.0155 | 0.0131707, 0.0181173 | null | null |
| verification/C17/45/0.985/joint | 0.985 | true | 10000 | 291 | 593 | true | 0.0291 | 0.0258933, 0.0325847 | null | null |
| verification/C17/45/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/45/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/45/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C17/60/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/60/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/60/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C17/60/0.980/A-F | 0.98 | true | 10000 | 201 | 317 | true | 0.0201 | 0.0174398, 0.0230446 | null | null |
| verification/C17/60/0.980/P-F | 0.98 | true | 10000 | 181 | 317 | true | 0.0181 | 0.0155782, 0.0209077 | null | null |
| verification/C17/60/0.980/joint | 0.98 | true | 10000 | 372 | 593 | true | 0.0372 | 0.0335761, 0.0410965 | null | null |
| verification/C17/60/0.985/A-F | 0.985 | true | 10000 | 147 | 317 | true | 0.0147 | 0.0124334, 0.0172553 | null | null |
| verification/C17/60/0.985/P-F | 0.985 | true | 10000 | 169 | 317 | true | 0.0169 | 0.0144651, 0.0196217 | null | null |
| verification/C17/60/0.985/joint | 0.985 | true | 10000 | 312 | 593 | true | 0.0312 | 0.0278796, 0.034797 | null | null |
| verification/C17/60/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/60/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/60/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C17/75/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/75/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/75/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C17/75/0.980/A-F | 0.98 | true | 10000 | 178 | 317 | true | 0.0178 | 0.0152996, 0.0205865 | null | null |
| verification/C17/75/0.980/P-F | 0.98 | true | 10000 | 212 | 317 | true | 0.0212 | 0.0184668, 0.0242168 | null | null |
| verification/C17/75/0.980/joint | 0.98 | true | 10000 | 379 | 593 | true | 0.0379 | 0.0342425, 0.0418297 | null | null |
| verification/C17/75/0.985/A-F | 0.985 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| verification/C17/75/0.985/P-F | 0.985 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | null |
| verification/C17/75/0.985/joint | 0.985 | true | 10000 | 296 | 593 | true | 0.0296 | 0.0263658, 0.0331118 | null | null |
| verification/C17/75/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/75/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/75/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C17/90/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/90/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/90/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C17/90/0.980/A-F | 0.98 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | null |
| verification/C17/90/0.980/P-F | 0.98 | true | 10000 | 194 | 317 | true | 0.0194 | 0.0167874, 0.0222975 | null | null |
| verification/C17/90/0.980/joint | 0.98 | true | 10000 | 376 | 593 | true | 0.0376 | 0.0339568, 0.0415155 | null | null |
| verification/C17/90/0.985/A-F | 0.985 | true | 10000 | 166 | 317 | true | 0.0166 | 0.0141874, 0.0192998 | null | null |
| verification/C17/90/0.985/P-F | 0.985 | true | 10000 | 141 | 317 | true | 0.0141 | 0.0118815, 0.0166077 | null | null |
| verification/C17/90/0.985/joint | 0.985 | true | 10000 | 303 | 593 | true | 0.0303 | 0.0270278, 0.0338494 | null | null |
| verification/C17/90/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/90/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/90/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C17/99/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/99/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/99/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C17/99/0.980/A-F | 0.98 | true | 10000 | 201 | 317 | true | 0.0201 | 0.0174398, 0.0230446 | null | null |
| verification/C17/99/0.980/P-F | 0.98 | true | 10000 | 192 | 317 | true | 0.0192 | 0.0166011, 0.0220839 | null | null |
| verification/C17/99/0.980/joint | 0.98 | true | 10000 | 385 | 593 | true | 0.0385 | 0.034814, 0.0424578 | null | null |
| verification/C17/99/0.985/A-F | 0.985 | true | 10000 | 165 | 317 | true | 0.0165 | 0.0140948, 0.0191924 | null | null |
| verification/C17/99/0.985/P-F | 0.985 | true | 10000 | 135 | 317 | true | 0.0135 | 0.0113307, 0.015959 | null | null |
| verification/C17/99/0.985/joint | 0.985 | true | 10000 | 297 | 593 | true | 0.0297 | 0.0264603, 0.0332172 | null | null |
| verification/C17/99/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/99/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/99/0.990/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C17/cap/0.975/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/cap/0.975/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/cap/0.975/joint | null | false | null | null | null | null | null | null | null | null |
| verification/C17/cap/0.980/A-F | 0.98 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | 36 |
| verification/C17/cap/0.980/P-F | 0.98 | true | 10000 | 190 | 317 | true | 0.019 | 0.016415, 0.0218703 | null | 36 |
| verification/C17/cap/0.980/joint | 0.98 | true | 10000 | 371 | 593 | true | 0.0371 | 0.0334809, 0.0409918 | null | 36 |
| verification/C17/cap/0.985/A-F | 0.985 | true | 10000 | 151 | 317 | true | 0.0151 | 0.0128018, 0.0176865 | null | 36 |
| verification/C17/cap/0.985/P-F | 0.985 | true | 10000 | 174 | 317 | true | 0.0174 | 0.0149285, 0.020158 | null | 36 |
| verification/C17/cap/0.985/joint | 0.985 | true | 10000 | 322 | 593 | true | 0.0322 | 0.0288269, 0.035849 | null | 36 |
| verification/C17/cap/0.990/A-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/cap/0.990/P-F | null | false | null | null | null | null | null | null | null | null |
| verification/C17/cap/0.990/joint | null | false | null | null | null | null | null | null | null | null |


## qualification

| candidate | rate | arm | lr | state | fallbacks | warnings |
|---|---|---|---|---|---|---|
| c1 | adamw-1.45e-05 | A | 1.45e-05 | qualified |  |  |
| c1 | adamw-2.11e-05 | A | 2.11e-05 | qualified |  |  |
| c1 | adam_frozen-1.3e-05 | F | 1.3e-05 | failing |  |  |
| c1 | adam_frozen-1.83e-05 | F | 1.83e-05 | failing |  |  |
| c1 | adamw_prewarmed-1.3e-05 | P | 1.3e-05 | qualified |  |  |
| c1 | adamw_prewarmed-1.85e-05 | P | 1.85e-05 | failing |  |  |

| candidate | supersession | sha256 | superseded verdict | its sha256 | cases re-qualified |
|---|---|---|---|---|---|
| c1 | $RUNS_DIR/qualification/c1/supersede-1.json | 63d04957876e5d15fd0b6d0faccfcdd63dcb3b5864bd011ffbaa1b3649ad3051 | /workspace/runs/zero-field-refusal/qualification/c1/engineering/s10002/adamw_prewarmed-1.3e-05/rest-attempt-1.json | b695f67b87b6d69f3d4151f0486dfb7d022973c3e7100cee72cc55c56112a3ba | {"case": "s10000/adam_frozen-1.3e-05", "stage": "engineering"}, {"case": "s10000/adam_frozen-1.83e-05", "stage": "engineering"}, {"case": "s10000/adamw-1.45e-05", "stage": "engineering"}, {"case": "s10000/adamw_prewarmed-1.3e-05", "stage": "engineering"}, {"case": "s10000/adamw_prewarmed-1.85e-05", "stage": "engineering"}, {"case": "s10001/adam_frozen-1.3e-05", "stage": "engineering"}, {"case": "s10001/adam_frozen-1.83e-05", "stage": "engineering"}, {"case": "s10001/adamw-1.45e-05", "stage": "engineering"}, {"case": "s10001/adamw-2.11e-05", "stage": "engineering"}, {"case": "s10001/adamw_prewarmed-1.3e-05", "stage": "engineering"}, {"case": "s10001/adamw_prewarmed-1.85e-05", "stage": "engineering"}, {"case": "s10002/adam_frozen-1.3e-05", "stage": "engineering"}, {"case": "s10002/adam_frozen-1.83e-05", "stage": "engineering"}, {"case": "s10002/adamw-1.45e-05", "stage": "engineering"}, {"case": "s10002/adamw-2.11e-05", "stage": "engineering"}, {"case": "s10002/adamw_prewarmed-1.3e-05", "stage": "engineering"} |

| candidate | case restart | sha256 | case | launcher run |
|---|---|---|---|---|

| field | value |
|---|---|
| c1 problems | 0 |


## not_counted

| run_id | label | candidate | step | reason | commit |
|---|---|---|---|---|---|


## validation

| comparison | supported | minimum | utility | passes |
|---|---|---|---|---|
| AF | 27 | 22 | true | true |
| PF | 26 | 22 | true | true |

| comparison | arm | metric | n | mean (pp) | upper (pp) | limit (pp) | passes |
|---|---|---|---|---|---|---|---|
| AF | A | mmlu | 27 | 2.62242 | 2.95635 | 5 | true |
| AF | A | factual | 27 | 0.330401 | 0.422949 | 10 | true |
| AF | F | mmlu | 27 | 2.696 | 2.99756 | 5 | true |
| AF | F | factual | 27 | 0.451464 | 0.560044 | 10 | true |
| PF | P | mmlu | 26 | 2.73612 | 3.07632 | 5 | true |
| PF | P | factual | 26 | 0.397473 | 0.493562 | 10 | true |
| PF | F | mmlu | 26 | 2.71771 | 3.02919 | 5 | true |
| PF | F | factual | 26 | 0.44605 | 0.558647 | 10 | true |

| comparison | excluded blocks |
|---|---|
| AF | 10305 |
| PF | 10305, 10317 |

| field | value |
|---|---|
| blocks | 28 |
| registered spacing | 0.223144 |

| arm | lr | blocks | SD of ln K | spacing / SD |
|---|---|---|---|---|
| A | 1.45e-05 | 28 | 0.0787271 | 2.83439 |
| A | 2.11e-05 | 28 | 0.0632341 | 3.52885 |
| F | 1.3e-05 | 28 | 0.0802818 | 2.7795 |
| F | 1.83e-05 | 27 | 0.0712649 | 3.13119 |
| P | 1.3e-05 | 28 | 0.113032 | 1.97416 |
| P | 1.85e-05 | 28 | 0.07131 | 3.12921 |


## n

| field | value |
|---|---|
| record | study3/decisions/n-c1.json |
| sha256 | e3d3bcc1315fa702fcb5778feee12883ff6d2d4071c7a39b82aa9e5d97948a7b |
| state | passed |
| N | 60 |
| cap | 74 |
| spread (pp) | 3.5 |
| L | 0.985 |
| active | A-F, P-F |
| cohort blocks | 27 |
| P-F power | 0.8596 |
| budget record | null |
| budget decision | null |


## confirmation

| field | value |
|---|---|
| candidate | 1 |
| N | 60 |
| cap | 74 |
| started | 64 |
| seeds | 10400, 10463 |
| counters | {"A-F": 61, "P-F": 60} |
| supported | {"A-F": 61, "P-F": 60} |
| stop | {"stopped": true, "why": "counters"} |
| readings | {"A-F": "confirmatory", "P-F": "confirmatory"} |

| comparison | seed | state | reason | arms |
|---|---|---|---|---|
| A-F | 10428 | unsupported | unsupported | {"F": "too wide"} |
| A-F | 10447 | unsupported | unsupported | {"F": "failed endpoint"} |
| A-F | 10460 | unsupported | unsupported | {"F": "failed endpoint"} |
| A-F | 10463 | supported | beyond_cohort | {} |
| P-F | 10414 | unsupported | unsupported | {"P": "failed endpoint"} |
| P-F | 10428 | unsupported | unsupported | {"F": "too wide"} |
| P-F | 10447 | unsupported | unsupported | {"F": "failed endpoint", "P": "failed endpoint"} |
| P-F | 10460 | unsupported | unsupported | {"F": "failed endpoint"} |


## utility

| run_id | step | candidate | arm | lr | seed | attempt | state | unusable | MMLU loss (pp) | factual loss (pp) |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261103T192819Z-c1-validation-adamw-1.45e-05-s10300-4b0be9c1-a1 | validation | 1 | A | 1.45e-05 | 10300 | 1 | usable | null | 3 | 0 |
| 20261103T193353Z-c1-validation-adamw-2.11e-05-s10300-4e1b700c-a1 | validation | 1 | A | 2.11e-05 | 10300 | 1 | usable | null | 3.4 | 0 |
| 20261103T193932Z-c1-validation-adam_frozen-1.-s10300-33bbba32-a1 | validation | 1 | F | 1.3e-05 | 10300 | 1 | usable | null | 2.4 | 0 |
| 20261103T194605Z-c1-validation-adam_frozen-1.-s10300-b30aaf76-a1 | validation | 1 | F | 1.83e-05 | 10300 | 1 | usable | null | 2.4 | 0 |
| 20261103T195239Z-c1-validation-adamw_prewarme-s10300-b3b766f9-a1 | validation | 1 | P | 1.3e-05 | 10300 | 1 | usable | null | 2.4 | 0 |
| 20261103T195921Z-c1-validation-adamw_prewarme-s10300-2bae657d-a1 | validation | 1 | P | 1.85e-05 | 10300 | 1 | usable | null | 2.6 | 0 |
| 20261103T200603Z-c1-validation-adamw-1.45e-05-s10301-322cbf49-a1 | validation | 1 | A | 1.45e-05 | 10301 | 1 | usable | null | 0.4 | 0.45045 |
| 20261103T201101Z-c1-validation-adamw-2.11e-05-s10301-174571b2-a1 | validation | 1 | A | 2.11e-05 | 10301 | 1 | usable | null | 4.4 | 0.900901 |
| 20261103T201600Z-c1-validation-adam_frozen-1.-s10301-a7037464-a1 | validation | 1 | F | 1.3e-05 | 10301 | 1 | usable | null | 1.4 | 0.45045 |
| 20261103T202218Z-c1-validation-adam_frozen-1.-s10301-fb43c5bb-a1 | validation | 1 | F | 1.83e-05 | 10301 | 1 | usable | null | 6.4 | 0.900901 |
| 20261103T202838Z-c1-validation-adamw_prewarme-s10301-256eb909-a1 | validation | 1 | P | 1.3e-05 | 10301 | 1 | usable | null | 1.4 | 0.45045 |
| 20261103T203503Z-c1-validation-adamw_prewarme-s10301-9e6cf3a0-a1 | validation | 1 | P | 1.85e-05 | 10301 | 1 | usable | null | 6.6 | 0.900901 |
| 20261103T204126Z-c1-validation-adamw-1.45e-05-s10302-8e10cef6-a1 | validation | 1 | A | 1.45e-05 | 10302 | 1 | usable | null | 1 | 0.45045 |
| 20261103T204651Z-c1-validation-adamw-2.11e-05-s10302-663e20f0-a1 | validation | 1 | A | 2.11e-05 | 10302 | 1 | usable | null | 2.8 | 0.45045 |
| 20261103T205232Z-c1-validation-adam_frozen-1.-s10302-a2817236-a1 | validation | 1 | F | 1.3e-05 | 10302 | 1 | usable | null | 2.8 | 0 |
| 20261103T205900Z-c1-validation-adam_frozen-1.-s10302-238df8e0-a1 | validation | 1 | F | 1.83e-05 | 10302 | 1 | usable | null | 2.8 | 0.45045 |
| 20261103T210529Z-c1-validation-adamw_prewarme-s10302-aeff00b0-a1 | validation | 1 | P | 1.3e-05 | 10302 | 1 | usable | null | 2.6 | 0 |
| 20261103T211202Z-c1-validation-adamw_prewarme-s10302-a90d29a6-a1 | validation | 1 | P | 1.85e-05 | 10302 | 1 | usable | null | 2.6 | 0.45045 |
| 20261103T211835Z-c1-validation-adamw-1.45e-05-s10303-cd927701-a1 | validation | 1 | A | 1.45e-05 | 10303 | 1 | usable | null | 2.4 | 0 |
| 20261103T212407Z-c1-validation-adamw-2.11e-05-s10303-660ec186-a1 | validation | 1 | A | 2.11e-05 | 10303 | 1 | usable | null | 5 | 0 |
| 20261103T212937Z-c1-validation-adam_frozen-1.-s10303-7e0071d5-a1 | validation | 1 | F | 1.3e-05 | 10303 | 1 | usable | null | 2.8 | 0.45045 |
| 20261103T213557Z-c1-validation-adam_frozen-1.-s10303-59a9e6d4-a1 | validation | 1 | F | 1.83e-05 | 10303 | 1 | usable | null | 4.2 | 0 |
| 20261103T214227Z-c1-validation-adamw_prewarme-s10303-836189bd-a1 | validation | 1 | P | 1.3e-05 | 10303 | 1 | usable | null | 2.4 | 0.45045 |
| 20261103T214849Z-c1-validation-adamw_prewarme-s10303-977d4904-a1 | validation | 1 | P | 1.85e-05 | 10303 | 1 | usable | null | 4.2 | 0 |
| 20261103T215527Z-c1-validation-adamw-1.45e-05-s10304-8fd4d4d0-a1 | validation | 1 | A | 1.45e-05 | 10304 | 1 | usable | null | 2 | 0.45045 |
| 20261103T220054Z-c1-validation-adamw-2.11e-05-s10304-47feb002-a1 | validation | 1 | A | 2.11e-05 | 10304 | 1 | usable | null | 2.4 | 0.45045 |
| 20261103T220626Z-c1-validation-adam_frozen-1.-s10304-dba16aaa-a1 | validation | 1 | F | 1.3e-05 | 10304 | 1 | usable | null | 2.4 | 0.45045 |
| 20261103T221241Z-c1-validation-adam_frozen-1.-s10304-ec68b574-a1 | validation | 1 | F | 1.83e-05 | 10304 | 1 | usable | null | 2 | 0 |
| 20261103T221914Z-c1-validation-adamw_prewarme-s10304-ce9b9a05-a1 | validation | 1 | P | 1.3e-05 | 10304 | 1 | usable | null | 2.6 | 0.45045 |
| 20261103T222539Z-c1-validation-adamw_prewarme-s10304-3e1f7d9b-a1 | validation | 1 | P | 1.85e-05 | 10304 | 1 | usable | null | 1.4 | 0 |
| 20261103T223221Z-c1-validation-adamw-1.45e-05-s10305-286c2ed8-a1 | validation | 1 | A | 1.45e-05 | 10305 | 1 | usable | null | 2 | 0 |
| 20261103T223746Z-c1-validation-adamw-2.11e-05-s10305-5369eb55-a1 | validation | 1 | A | 2.11e-05 | 10305 | 1 | usable | null | 2 | 0 |
| 20261103T224322Z-c1-validation-adam_frozen-1.-s10305-842f5156-a1 | validation | 1 | F | 1.3e-05 | 10305 | 1 | usable | null | 2.2 | 0 |
| 20261103T224950Z-c1-validation-adam_frozen-1.-s10305-bc1bfa73-a1 | validation | 1 | F | 1.83e-05 | 10305 | 1 | scientific | null | null | null |
| 20261103T225314Z-c1-validation-adamw_prewarme-s10305-0f4bb253-a1 | validation | 1 | P | 1.3e-05 | 10305 | 1 | usable | null | 2.6 | 0 |
| 20261103T225952Z-c1-validation-adamw_prewarme-s10305-c5568527-a1 | validation | 1 | P | 1.85e-05 | 10305 | 1 | usable | null | 4.6 | 0 |
| 20261103T230634Z-c1-validation-adamw-1.45e-05-s10306-d49a8f11-a1 | validation | 1 | A | 1.45e-05 | 10306 | 1 | usable | null | 1.6 | 0 |
| 20261103T231209Z-c1-validation-adamw-2.11e-05-s10306-e574b367-a1 | validation | 1 | A | 2.11e-05 | 10306 | 1 | usable | null | 2.2 | 0 |
| 20261103T231746Z-c1-validation-adam_frozen-1.-s10306-167cdb48-a1 | validation | 1 | F | 1.3e-05 | 10306 | 1 | usable | null | 0.6 | 0 |
| 20261103T232409Z-c1-validation-adam_frozen-1.-s10306-e2718b4e-a1 | validation | 1 | F | 1.83e-05 | 10306 | 1 | usable | null | 1 | 0 |
| 20261103T233039Z-c1-validation-adamw_prewarme-s10306-5432bc82-a1 | validation | 1 | P | 1.3e-05 | 10306 | 1 | usable | null | 0.2 | 0 |
| 20261103T233704Z-c1-validation-adamw_prewarme-s10306-b4fa8a0e-a1 | validation | 1 | P | 1.85e-05 | 10306 | 1 | usable | null | 1.4 | 0 |
| 20261103T234343Z-c1-validation-adamw-1.45e-05-s10307-7179b773-a1 | validation | 1 | A | 1.45e-05 | 10307 | 1 | usable | null | 2 | 0 |
| 20261103T234919Z-c1-validation-adamw-2.11e-05-s10307-6f31ec7c-a1 | validation | 1 | A | 2.11e-05 | 10307 | 1 | usable | null | 4.4 | 0.45045 |
| 20261103T235500Z-c1-validation-adam_frozen-1.-s10307-a6114a39-a1 | validation | 1 | F | 1.3e-05 | 10307 | 1 | usable | null | 2.2 | 0 |
| 20261104T000117Z-c1-validation-adam_frozen-1.-s10307-ae10d8b6-a1 | validation | 1 | F | 1.83e-05 | 10307 | 1 | usable | null | 3.2 | 0.900901 |
| 20261104T000753Z-c1-validation-adamw_prewarme-s10307-dd1fecf4-a1 | validation | 1 | P | 1.3e-05 | 10307 | 1 | usable | null | 2.2 | 0 |
| 20261104T001417Z-c1-validation-adamw_prewarme-s10307-6a78cc1d-a1 | validation | 1 | P | 1.85e-05 | 10307 | 1 | usable | null | 3.6 | 0.45045 |
| 20261104T002100Z-c1-validation-adamw-1.45e-05-s10308-abf018d7-a1 | validation | 1 | A | 1.45e-05 | 10308 | 1 | usable | null | 1.6 | 0 |
| 20261104T002636Z-c1-validation-adamw-2.11e-05-s10308-46004861-a1 | validation | 1 | A | 2.11e-05 | 10308 | 1 | usable | null | 2 | 0.45045 |
| 20261104T003208Z-c1-validation-adam_frozen-1.-s10308-6c4c6d85-a1 | validation | 1 | F | 1.3e-05 | 10308 | 1 | usable | null | 2 | 0 |
| 20261104T003843Z-c1-validation-adam_frozen-1.-s10308-16fcbb41-a1 | validation | 1 | F | 1.83e-05 | 10308 | 1 | usable | null | 3.4 | 0.45045 |
| 20261104T004517Z-c1-validation-adamw_prewarme-s10308-63100799-a1 | validation | 1 | P | 1.3e-05 | 10308 | 1 | usable | null | 2 | 0 |
| 20261104T005202Z-c1-validation-adamw_prewarme-s10308-0aeefe8f-a1 | validation | 1 | P | 1.85e-05 | 10308 | 1 | usable | null | 3 | 0.45045 |
| 20261104T005845Z-c1-validation-adamw-1.45e-05-s10309-b08584e2-a1 | validation | 1 | A | 1.45e-05 | 10309 | 1 | usable | null | 1.4 | 0 |
| 20261104T010423Z-c1-validation-adamw-2.11e-05-s10309-b0c5f20c-a1 | validation | 1 | A | 2.11e-05 | 10309 | 1 | usable | null | 4.2 | 0.45045 |
| 20261104T011002Z-c1-validation-adam_frozen-1.-s10309-dae5b28b-a1 | validation | 1 | F | 1.3e-05 | 10309 | 1 | usable | null | 2.4 | 0.45045 |
| 20261104T011638Z-c1-validation-adam_frozen-1.-s10309-92ce6217-a1 | validation | 1 | F | 1.83e-05 | 10309 | 1 | usable | null | 5.6 | 0.45045 |
| 20261104T012311Z-c1-validation-adamw_prewarme-s10309-1fa24097-a1 | validation | 1 | P | 1.3e-05 | 10309 | 1 | usable | null | 2.4 | 0.45045 |
| 20261104T012952Z-c1-validation-adamw_prewarme-s10309-94b96e63-a1 | validation | 1 | P | 1.85e-05 | 10309 | 1 | usable | null | 5.6 | 0.45045 |
| 20261104T013633Z-c1-validation-adamw-1.45e-05-s10310-186853ff-a1 | validation | 1 | A | 1.45e-05 | 10310 | 1 | usable | null | 2 | 0.45045 |
| 20261104T014212Z-c1-validation-adamw-2.11e-05-s10310-8d7b3c47-a1 | validation | 1 | A | 2.11e-05 | 10310 | 1 | usable | null | 1.6 | 0.900901 |
| 20261104T014752Z-c1-validation-adam_frozen-1.-s10310-7edb726e-a1 | validation | 1 | F | 1.3e-05 | 10310 | 1 | usable | null | 2.4 | 0.900901 |
| 20261104T015427Z-c1-validation-adam_frozen-1.-s10310-0be09ebb-a1 | validation | 1 | F | 1.83e-05 | 10310 | 1 | usable | null | 2.6 | 1.35135 |
| 20261104T020104Z-c1-validation-adamw_prewarme-s10310-86f3c37e-a1 | validation | 1 | P | 1.3e-05 | 10310 | 1 | usable | null | 2.2 | 0.900901 |
| 20261104T020746Z-c1-validation-adamw_prewarme-s10310-16ab5624-a1 | validation | 1 | P | 1.85e-05 | 10310 | 1 | usable | null | 2 | 1.35135 |
| 20261104T021429Z-c1-validation-adamw-1.45e-05-s10311-8ce29cf5-a1 | validation | 1 | A | 1.45e-05 | 10311 | 1 | usable | null | 4.8 | 0.45045 |
| 20261104T022010Z-c1-validation-adamw-2.11e-05-s10311-b3d42905-a1 | validation | 1 | A | 2.11e-05 | 10311 | 1 | usable | null | 5.6 | 0 |
| 20261104T022547Z-c1-validation-adam_frozen-1.-s10311-860a2233-a1 | validation | 1 | F | 1.3e-05 | 10311 | 1 | usable | null | 3.2 | 0.900901 |
| 20261104T023224Z-c1-validation-adam_frozen-1.-s10311-36abf2fd-a1 | validation | 1 | F | 1.83e-05 | 10311 | 1 | usable | null | 5 | 0.45045 |
| 20261104T023858Z-c1-validation-adamw_prewarme-s10311-53bea0cf-a1 | validation | 1 | P | 1.3e-05 | 10311 | 1 | usable | null | 3.4 | 0.900901 |
| 20261104T024541Z-c1-validation-adamw_prewarme-s10311-0caae3f1-a1 | validation | 1 | P | 1.85e-05 | 10311 | 1 | usable | null | 5.2 | 0.45045 |
| 20261104T025224Z-c1-validation-adamw-1.45e-05-s10312-551cfabe-a1 | validation | 1 | A | 1.45e-05 | 10312 | 1 | usable | null | 3.2 | 0.45045 |
| 20261104T025734Z-c1-validation-adamw-2.11e-05-s10312-1e303d2a-a1 | validation | 1 | A | 2.11e-05 | 10312 | 1 | usable | null | 2.8 | 1.35135 |
| 20261104T030245Z-c1-validation-adam_frozen-1.-s10312-a4b1adfe-a1 | validation | 1 | F | 1.3e-05 | 10312 | 1 | usable | null | 2.4 | 0 |
| 20261104T030854Z-c1-validation-adam_frozen-1.-s10312-3b1925d5-a1 | validation | 1 | F | 1.83e-05 | 10312 | 1 | usable | null | 2.4 | 0.45045 |
| 20261104T031501Z-c1-validation-adamw_prewarme-s10312-713738c8-a1 | validation | 1 | P | 1.3e-05 | 10312 | 1 | usable | null | 2.4 | 0 |
| 20261104T032117Z-c1-validation-adamw_prewarme-s10312-c8756eaf-a1 | validation | 1 | P | 1.85e-05 | 10312 | 1 | usable | null | 2.6 | 0.45045 |
| 20261104T032726Z-c1-validation-adamw-1.45e-05-s10313-b8493810-a1 | validation | 1 | A | 1.45e-05 | 10313 | 1 | usable | null | 1.4 | 0 |
| 20261104T033308Z-c1-validation-adamw-2.11e-05-s10313-7d202842-a1 | validation | 1 | A | 2.11e-05 | 10313 | 1 | usable | null | 3.6 | 0.900901 |
| 20261104T033848Z-c1-validation-adam_frozen-1.-s10313-75c1d3e1-a1 | validation | 1 | F | 1.3e-05 | 10313 | 1 | usable | null | 2.2 | 0.45045 |
| 20261104T034524Z-c1-validation-adam_frozen-1.-s10313-a32858ab-a1 | validation | 1 | F | 1.83e-05 | 10313 | 1 | usable | null | 1.8 | 0.45045 |
| 20261104T035159Z-c1-validation-adamw_prewarme-s10313-07795de0-a1 | validation | 1 | P | 1.3e-05 | 10313 | 1 | usable | null | 2 | 0.45045 |
| 20261104T035841Z-c1-validation-adamw_prewarme-s10313-3c7fd1ab-a1 | validation | 1 | P | 1.85e-05 | 10313 | 1 | usable | null | 2 | 0.45045 |
| 20261104T040526Z-c1-validation-adamw-1.45e-05-s10314-fcf3c51e-a1 | validation | 1 | A | 1.45e-05 | 10314 | 1 | usable | null | 4 | 0 |
| 20261104T041021Z-c1-validation-adamw-2.11e-05-s10314-7b64e1de-a1 | validation | 1 | A | 2.11e-05 | 10314 | 1 | usable | null | 5 | 0 |
| 20261104T041545Z-c1-validation-adam_frozen-1.-s10314-9418e837-a1 | validation | 1 | F | 1.3e-05 | 10314 | 1 | usable | null | 2.6 | 0 |
| 20261104T042134Z-c1-validation-adam_frozen-1.-s10314-32be69f1-a1 | validation | 1 | F | 1.83e-05 | 10314 | 1 | usable | null | 3 | 0.45045 |
| 20261104T042732Z-c1-validation-adamw_prewarme-s10314-c2c0b3dc-a1 | validation | 1 | P | 1.3e-05 | 10314 | 1 | usable | null | 2.6 | 0 |
| 20261104T043328Z-c1-validation-adamw_prewarme-s10314-659e61f2-a1 | validation | 1 | P | 1.85e-05 | 10314 | 1 | usable | null | 3 | 0.45045 |
| 20261104T043934Z-c1-validation-adamw-1.45e-05-s10315-d2b05d12-a1 | validation | 1 | A | 1.45e-05 | 10315 | 1 | usable | null | 1.6 | 0.900901 |
| 20261104T044447Z-c1-validation-adamw-2.11e-05-s10315-696d2bac-a1 | validation | 1 | A | 2.11e-05 | 10315 | 1 | usable | null | 0.2 | 0.900901 |
| 20261104T045021Z-c1-validation-adam_frozen-1.-s10315-0f0d15d4-a1 | validation | 1 | F | 1.3e-05 | 10315 | 1 | usable | null | 1.2 | 0.900901 |
| 20261104T045632Z-c1-validation-adam_frozen-1.-s10315-2397296e-a1 | validation | 1 | F | 1.83e-05 | 10315 | 1 | usable | null | 5.2 | 0.45045 |
| 20261104T050309Z-c1-validation-adamw_prewarme-s10315-77fe40a8-a1 | validation | 1 | P | 1.3e-05 | 10315 | 1 | usable | null | 1.6 | 0.900901 |
| 20261104T050931Z-c1-validation-adamw_prewarme-s10315-c1b9acac-a1 | validation | 1 | P | 1.85e-05 | 10315 | 1 | usable | null | 5.2 | 0.45045 |
| 20261104T051616Z-c1-validation-adamw-1.45e-05-s10316-1eeb4c0e-a1 | validation | 1 | A | 1.45e-05 | 10316 | 1 | usable | null | 1.6 | 0 |
| 20261104T052145Z-c1-validation-adamw-2.11e-05-s10316-cdc750d2-a1 | validation | 1 | A | 2.11e-05 | 10316 | 1 | usable | null | 2.8 | 0 |
| 20261104T052721Z-c1-validation-adam_frozen-1.-s10316-fa54cf3d-a1 | validation | 1 | F | 1.3e-05 | 10316 | 1 | usable | null | 2.2 | 0 |
| 20261104T053345Z-c1-validation-adam_frozen-1.-s10316-f04261b3-a1 | validation | 1 | F | 1.83e-05 | 10316 | 1 | usable | null | 3 | 0.45045 |
| 20261104T054017Z-c1-validation-adamw_prewarme-s10316-1340bc80-a1 | validation | 1 | P | 1.3e-05 | 10316 | 1 | usable | null | 2.2 | 0 |
| 20261104T054648Z-c1-validation-adamw_prewarme-s10316-81ee5f0c-a1 | validation | 1 | P | 1.85e-05 | 10316 | 1 | usable | null | 3 | 0.45045 |
| 20261104T055328Z-c1-validation-adamw-1.45e-05-s10317-85b4de6d-a1 | validation | 1 | A | 1.45e-05 | 10317 | 1 | usable | null | 1.8 | 0 |
| 20261104T055855Z-c1-validation-adamw-2.11e-05-s10317-f0cafc6e-a1 | validation | 1 | A | 2.11e-05 | 10317 | 1 | usable | null | 1.8 | 0.45045 |
| 20261104T060425Z-c1-validation-adam_frozen-1.-s10317-19fa5f26-a1 | validation | 1 | F | 1.3e-05 | 10317 | 1 | usable | null | 2 | 0 |
| 20261104T061103Z-c1-validation-adam_frozen-1.-s10317-b1767e3d-a1 | validation | 1 | F | 1.83e-05 | 10317 | 1 | usable | null | 2.2 | 0.900901 |
| 20261104T061739Z-c1-validation-adamw_prewarme-s10317-7105fbbf-a1 | validation | 1 | P | 1.3e-05 | 10317 | 1 | usable | null | 3.6 | 0 |
| 20261104T062421Z-c1-validation-adamw_prewarme-s10317-e4bc9fb7-a1 | validation | 1 | P | 1.85e-05 | 10317 | 1 | usable | null | 2.4 | 0.900901 |
| 20261104T063105Z-c1-validation-adamw-1.45e-05-s10318-6c52d013-a1 | validation | 1 | A | 1.45e-05 | 10318 | 1 | usable | null | 3.4 | 0.45045 |
| 20261104T063646Z-c1-validation-adamw-2.11e-05-s10318-c079bd5c-a1 | validation | 1 | A | 2.11e-05 | 10318 | 1 | usable | null | 4.2 | 0.900901 |
| 20261104T064228Z-c1-validation-adam_frozen-1.-s10318-7bc13378-a1 | validation | 1 | F | 1.3e-05 | 10318 | 1 | usable | null | 3.4 | 0.45045 |
| 20261104T064905Z-c1-validation-adam_frozen-1.-s10318-b6be3a96-a1 | validation | 1 | F | 1.83e-05 | 10318 | 1 | usable | null | 4.2 | 1.35135 |
| 20261104T065540Z-c1-validation-adamw_prewarme-s10318-e08546cd-a1 | validation | 1 | P | 1.3e-05 | 10318 | 1 | usable | null | 3.6 | 0.45045 |
| 20261104T070222Z-c1-validation-adamw_prewarme-s10318-17b3f3de-a1 | validation | 1 | P | 1.85e-05 | 10318 | 1 | usable | null | 4.2 | 0.900901 |
| 20261104T070904Z-c1-validation-adamw-1.45e-05-s10319-832f44ae-a1 | validation | 1 | A | 1.45e-05 | 10319 | 1 | usable | null | 2.6 | 0 |
| 20261104T071439Z-c1-validation-adamw-2.11e-05-s10319-e0622e91-a1 | validation | 1 | A | 2.11e-05 | 10319 | 1 | usable | null | 2 | 0.45045 |
| 20261104T072019Z-c1-validation-adam_frozen-1.-s10319-bc8f9262-a1 | validation | 1 | F | 1.3e-05 | 10319 | 1 | usable | null | 2.8 | 0 |
| 20261104T072651Z-c1-validation-adam_frozen-1.-s10319-ccdadbcb-a1 | validation | 1 | F | 1.83e-05 | 10319 | 1 | usable | null | 0.4 | 0 |
| 20261104T073326Z-c1-validation-adamw_prewarme-s10319-4f349277-a1 | validation | 1 | P | 1.3e-05 | 10319 | 1 | usable | null | 2.6 | 0 |
| 20261104T074004Z-c1-validation-adamw_prewarme-s10319-85c772b0-a1 | validation | 1 | P | 1.85e-05 | 10319 | 1 | usable | null | 0.4 | 0 |
| 20261104T074648Z-c1-validation-adamw-1.45e-05-s10320-b9686efe-a1 | validation | 1 | A | 1.45e-05 | 10320 | 1 | usable | null | 2.8 | 0.45045 |
| 20261104T075226Z-c1-validation-adamw-2.11e-05-s10320-b65b8eab-a1 | validation | 1 | A | 2.11e-05 | 10320 | 1 | usable | null | 2.8 | 0 |
| 20261104T075807Z-c1-validation-adam_frozen-1.-s10320-648e99d2-a1 | validation | 1 | F | 1.3e-05 | 10320 | 1 | usable | null | 2.6 | 0.45045 |
| 20261104T080437Z-c1-validation-adam_frozen-1.-s10320-eebaaea4-a1 | validation | 1 | F | 1.83e-05 | 10320 | 1 | usable | null | 2.2 | 0.45045 |
| 20261104T081115Z-c1-validation-adamw_prewarme-s10320-0addc088-a1 | validation | 1 | P | 1.3e-05 | 10320 | 1 | usable | null | 2.6 | 0.45045 |
| 20261104T081756Z-c1-validation-adamw_prewarme-s10320-81b4ecdf-a1 | validation | 1 | P | 1.85e-05 | 10320 | 1 | usable | null | 2.2 | 0.45045 |
| 20261104T082440Z-c1-validation-adamw-1.45e-05-s10321-492d650d-a1 | validation | 1 | A | 1.45e-05 | 10321 | 1 | usable | null | 3 | 0.45045 |
| 20261104T083009Z-c1-validation-adamw-2.11e-05-s10321-2486a7c3-a1 | validation | 1 | A | 2.11e-05 | 10321 | 1 | usable | null | 2.4 | 0.900901 |
| 20261104T083543Z-c1-validation-adam_frozen-1.-s10321-c808d05e-a1 | validation | 1 | F | 1.3e-05 | 10321 | 1 | usable | null | 3.2 | 0.45045 |
| 20261104T084209Z-c1-validation-adam_frozen-1.-s10321-3c8c3090-a1 | validation | 1 | F | 1.83e-05 | 10321 | 1 | usable | null | 3.6 | 0.900901 |
| 20261104T084838Z-c1-validation-adamw_prewarme-s10321-d5df55a2-a1 | validation | 1 | P | 1.3e-05 | 10321 | 1 | usable | null | 3.2 | 0.45045 |
| 20261104T085513Z-c1-validation-adamw_prewarme-s10321-08e0a230-a1 | validation | 1 | P | 1.85e-05 | 10321 | 1 | usable | null | 3.4 | 0.900901 |
| 20261104T090151Z-c1-validation-adamw-1.45e-05-s10322-13dedfbc-a1 | validation | 1 | A | 1.45e-05 | 10322 | 1 | usable | null | 1.6 | 0 |
| 20261104T090705Z-c1-validation-adamw-2.11e-05-s10322-40f8751f-a1 | validation | 1 | A | 2.11e-05 | 10322 | 1 | usable | null | 2.8 | 0.45045 |
| 20261104T091230Z-c1-validation-adam_frozen-1.-s10322-35727de1-a1 | validation | 1 | F | 1.3e-05 | 10322 | 1 | usable | null | 1.2 | 0.45045 |
| 20261104T091832Z-c1-validation-adam_frozen-1.-s10322-f489bdb9-a1 | validation | 1 | F | 1.83e-05 | 10322 | 1 | usable | null | 4 | 1.8018 |
| 20261104T092508Z-c1-validation-adamw_prewarme-s10322-cdc79294-a1 | validation | 1 | P | 1.3e-05 | 10322 | 1 | usable | null | 1 | 0.45045 |
| 20261104T093120Z-c1-validation-adamw_prewarme-s10322-8ff04316-a1 | validation | 1 | P | 1.85e-05 | 10322 | 1 | usable | null | 4.2 | 0.45045 |
| 20261104T093805Z-c1-validation-adamw-1.45e-05-s10323-58ec6b5b-a1 | validation | 1 | A | 1.45e-05 | 10323 | 1 | usable | null | 2.6 | 0 |
| 20261104T094313Z-c1-validation-adamw-2.11e-05-s10323-698f0494-a1 | validation | 1 | A | 2.11e-05 | 10323 | 1 | usable | null | 4 | 0.45045 |
| 20261104T094841Z-c1-validation-adam_frozen-1.-s10323-71a4f6e2-a1 | validation | 1 | F | 1.3e-05 | 10323 | 1 | usable | null | 3 | 0 |
| 20261104T095447Z-c1-validation-adam_frozen-1.-s10323-2934b145-a1 | validation | 1 | F | 1.83e-05 | 10323 | 1 | usable | null | 3.8 | 0.45045 |
| 20261104T100105Z-c1-validation-adamw_prewarme-s10323-616e0235-a1 | validation | 1 | P | 1.3e-05 | 10323 | 1 | usable | null | 3.2 | 0 |
| 20261104T100716Z-c1-validation-adamw_prewarme-s10323-dadac9af-a1 | validation | 1 | P | 1.85e-05 | 10323 | 1 | usable | null | 4 | 0.45045 |
| 20261104T101345Z-c1-validation-adamw-1.45e-05-s10324-cf5a7289-a1 | validation | 1 | A | 1.45e-05 | 10324 | 1 | usable | null | 0.8 | 0 |
| 20261104T101918Z-c1-validation-adamw-2.11e-05-s10324-e565458c-a1 | validation | 1 | A | 2.11e-05 | 10324 | 1 | usable | null | -0.4 | 0.900901 |
| 20261104T102500Z-c1-validation-adam_frozen-1.-s10324-1236b68f-a1 | validation | 1 | F | 1.3e-05 | 10324 | 1 | usable | null | 0.8 | 0 |
| 20261104T103125Z-c1-validation-adam_frozen-1.-s10324-1dfe4288-a1 | validation | 1 | F | 1.83e-05 | 10324 | 1 | usable | null | 0.6 | 0.900901 |
| 20261104T103759Z-c1-validation-adamw_prewarme-s10324-81b0a66b-a1 | validation | 1 | P | 1.3e-05 | 10324 | 1 | usable | null | 0.8 | 0 |
| 20261104T104431Z-c1-validation-adamw_prewarme-s10324-89d4ada2-a1 | validation | 1 | P | 1.85e-05 | 10324 | 1 | usable | null | 0.2 | 0.900901 |
| 20261104T105115Z-c1-validation-adamw-1.45e-05-s10325-1affe626-a1 | validation | 1 | A | 1.45e-05 | 10325 | 1 | usable | null | 3.2 | 0 |
| 20261104T105651Z-c1-validation-adamw-2.11e-05-s10325-8dde29c4-a1 | validation | 1 | A | 2.11e-05 | 10325 | 1 | usable | null | 4.2 | 1.8018 |
| 20261104T110224Z-c1-validation-adam_frozen-1.-s10325-1c5f63fc-a1 | validation | 1 | F | 1.3e-05 | 10325 | 1 | usable | null | 3.8 | 0 |
| 20261104T110837Z-c1-validation-adam_frozen-1.-s10325-b4cca10f-a1 | validation | 1 | F | 1.83e-05 | 10325 | 1 | usable | null | 6.6 | 1.35135 |
| 20261104T111511Z-c1-validation-adamw_prewarme-s10325-9b48af55-a1 | validation | 1 | P | 1.3e-05 | 10325 | 1 | usable | null | 3.4 | 0 |
| 20261104T112129Z-c1-validation-adamw_prewarme-s10325-f55f7270-a1 | validation | 1 | P | 1.85e-05 | 10325 | 1 | usable | null | 7.6 | 1.35135 |
| 20261104T112812Z-c1-validation-adamw-1.45e-05-s10326-30bbf194-a1 | validation | 1 | A | 1.45e-05 | 10326 | 1 | usable | null | 2.2 | 0 |
| 20261104T113328Z-c1-validation-adamw-2.11e-05-s10326-10fd4e63-a1 | validation | 1 | A | 2.11e-05 | 10326 | 1 | usable | null | 2.2 | 0 |
| 20261104T113859Z-c1-validation-adam_frozen-1.-s10326-1121fa74-a1 | validation | 1 | F | 1.3e-05 | 10326 | 1 | usable | null | 2.8 | 0 |
| 20261104T114510Z-c1-validation-adam_frozen-1.-s10326-c809491f-a1 | validation | 1 | F | 1.83e-05 | 10326 | 1 | usable | null | 3.6 | 0 |
| 20261104T115133Z-c1-validation-adamw_prewarme-s10326-d9a266ec-a1 | validation | 1 | P | 1.3e-05 | 10326 | 1 | usable | null | 2.8 | 0 |
| 20261104T115757Z-c1-validation-adamw_prewarme-s10326-30f6d96c-a1 | validation | 1 | P | 1.85e-05 | 10326 | 1 | usable | null | 3.8 | 0 |
| 20261104T120437Z-c1-validation-adamw-1.45e-05-s10327-824331e7-a1 | validation | 1 | A | 1.45e-05 | 10327 | 1 | usable | null | 3 | 0 |
| 20261104T121016Z-c1-validation-adamw-2.11e-05-s10327-51a720b0-a1 | validation | 1 | A | 2.11e-05 | 10327 | 1 | usable | null | 4 | 1.35135 |
| 20261104T121556Z-c1-validation-adam_frozen-1.-s10327-1a3b4641-a1 | validation | 1 | F | 1.3e-05 | 10327 | 1 | usable | null | 2.6 | 0 |
| 20261104T122231Z-c1-validation-adam_frozen-1.-s10327-098b6e0d-a1 | validation | 1 | F | 1.83e-05 | 10327 | 1 | usable | null | 3.4 | 1.35135 |
| 20261104T122906Z-c1-validation-adamw_prewarme-s10327-55167954-a1 | validation | 1 | P | 1.3e-05 | 10327 | 1 | usable | null | 3.2 | 0 |
| 20261104T123547Z-c1-validation-adamw_prewarme-s10327-d438cd5e-a1 | validation | 1 | P | 1.85e-05 | 10327 | 1 | usable | null | 3.6 | 1.35135 |
| 20261105T050251Z-c1-confirmation-adamw-1.45e--s10400-93b612e3-a1 | confirmation | 1 | A | 1.45e-05 | 10400 | 1 | usable | null | 1.2 | 0 |
| 20261105T050847Z-c1-confirmation-adamw-2.11e--s10400-2f0c4d7a-a1 | confirmation | 1 | A | 2.11e-05 | 10400 | 1 | usable | null | 1 | 0.45045 |
| 20261105T051443Z-c1-confirmation-adam_frozen--s10400-981a4aa8-a1 | confirmation | 1 | F | 1.3e-05 | 10400 | 1 | usable | null | 2 | 0 |
| 20261105T052138Z-c1-confirmation-adam_frozen--s10400-fed71d21-a1 | confirmation | 1 | F | 1.83e-05 | 10400 | 1 | usable | null | 5.6 | 0.900901 |
| 20261105T052837Z-c1-confirmation-adamw_prewar-s10400-3eba5aee-a1 | confirmation | 1 | P | 1.3e-05 | 10400 | 1 | usable | null | 1.6 | 0 |
| 20261105T053542Z-c1-confirmation-adamw_prewar-s10400-7afff485-a1 | confirmation | 1 | P | 1.85e-05 | 10400 | 1 | usable | null | 5 | 0.900901 |
| 20261105T054332Z-c1-confirmation-adamw-1.45e--s10401-b70eac9a-a1 | confirmation | 1 | A | 1.45e-05 | 10401 | 1 | usable | null | 3.6 | 0 |
| 20261105T054946Z-c1-confirmation-adamw-2.11e--s10401-2235bf7a-a1 | confirmation | 1 | A | 2.11e-05 | 10401 | 1 | usable | null | 4.2 | 0.45045 |
| 20261105T055612Z-c1-confirmation-adam_frozen--s10401-c3b639c6-a1 | confirmation | 1 | F | 1.3e-05 | 10401 | 1 | usable | null | 2.6 | 0 |
| 20261105T060318Z-c1-confirmation-adam_frozen--s10401-e5f1f0c8-a1 | confirmation | 1 | F | 1.83e-05 | 10401 | 1 | usable | null | 3 | 0 |
| 20261105T061024Z-c1-confirmation-adamw_prewar-s10401-9ca929f7-a1 | confirmation | 1 | P | 1.3e-05 | 10401 | 1 | usable | null | 2.4 | 0 |
| 20261105T061745Z-c1-confirmation-adamw_prewar-s10401-45eeaee4-a1 | confirmation | 1 | P | 1.85e-05 | 10401 | 1 | usable | null | 3.2 | 0 |
| 20261105T062552Z-c1-confirmation-adamw-1.45e--s10402-1bec2e29-a1 | confirmation | 1 | A | 1.45e-05 | 10402 | 1 | usable | null | 2.2 | 0 |
| 20261105T063208Z-c1-confirmation-adamw-2.11e--s10402-79b8512b-a1 | confirmation | 1 | A | 2.11e-05 | 10402 | 1 | usable | null | 2.8 | 0.900901 |
| 20261105T063826Z-c1-confirmation-adam_frozen--s10402-5bfa9f14-a1 | confirmation | 1 | F | 1.3e-05 | 10402 | 1 | usable | null | 1.8 | 0.45045 |
| 20261105T064536Z-c1-confirmation-adam_frozen--s10402-f81bb06d-a1 | confirmation | 1 | F | 1.83e-05 | 10402 | 1 | usable | null | 4 | 0.900901 |
| 20261105T065245Z-c1-confirmation-adamw_prewar-s10402-f1a5d5ee-a1 | confirmation | 1 | P | 1.3e-05 | 10402 | 1 | usable | null | 2.4 | 0.45045 |
| 20261105T070011Z-c1-confirmation-adamw_prewar-s10402-b740b6aa-a1 | confirmation | 1 | P | 1.85e-05 | 10402 | 1 | usable | null | 2.6 | 1.35135 |
| 20261105T070816Z-c1-confirmation-adamw-1.45e--s10403-ae69aa40-a1 | confirmation | 1 | A | 1.45e-05 | 10403 | 1 | usable | null | 3.2 | 0 |
| 20261105T071354Z-c1-confirmation-adamw-2.11e--s10403-b0680528-a1 | confirmation | 1 | A | 2.11e-05 | 10403 | 1 | usable | null | 4.4 | 0.45045 |
| 20261105T072010Z-c1-confirmation-adam_frozen--s10403-d1ce7a92-a1 | confirmation | 1 | F | 1.3e-05 | 10403 | 1 | usable | null | 2.8 | 0 |
| 20261105T072713Z-c1-confirmation-adam_frozen--s10403-3b4601aa-a1 | confirmation | 1 | F | 1.83e-05 | 10403 | 1 | usable | null | 3.6 | 0.45045 |
| 20261105T073428Z-c1-confirmation-adamw_prewar-s10403-e9e910a8-a1 | confirmation | 1 | P | 1.3e-05 | 10403 | 1 | usable | null | 2.8 | 0 |
| 20261105T074151Z-c1-confirmation-adamw_prewar-s10403-a6cc2188-a1 | confirmation | 1 | P | 1.85e-05 | 10403 | 1 | usable | null | 4.2 | 0.45045 |
| 20261105T075001Z-c1-confirmation-adamw-1.45e--s10404-b6c46fae-a1 | confirmation | 1 | A | 1.45e-05 | 10404 | 1 | usable | null | 1.8 | 0 |
| 20261105T075614Z-c1-confirmation-adamw-2.11e--s10404-bb5e2a10-a1 | confirmation | 1 | A | 2.11e-05 | 10404 | 1 | usable | null | 2.2 | -0.45045 |
| 20261105T080218Z-c1-confirmation-adam_frozen--s10404-79e87542-a1 | confirmation | 1 | F | 1.3e-05 | 10404 | 1 | usable | null | 1.8 | 0 |
| 20261105T080908Z-c1-confirmation-adam_frozen--s10404-12d33e5f-a1 | confirmation | 1 | F | 1.83e-05 | 10404 | 1 | usable | null | 0.2 | -0.45045 |
| 20261105T081611Z-c1-confirmation-adamw_prewar-s10404-7592f4d6-a1 | confirmation | 1 | P | 1.3e-05 | 10404 | 1 | usable | null | 2.2 | 0 |
| 20261105T082319Z-c1-confirmation-adamw_prewar-s10404-a34cbf14-a1 | confirmation | 1 | P | 1.85e-05 | 10404 | 1 | usable | null | 0.6 | -0.45045 |
| 20261105T083113Z-c1-confirmation-adamw-1.45e--s10405-f579fb0f-a1 | confirmation | 1 | A | 1.45e-05 | 10405 | 1 | usable | null | 1.8 | 0.45045 |
| 20261105T083654Z-c1-confirmation-adamw-2.11e--s10405-aed243f3-a1 | confirmation | 1 | A | 2.11e-05 | 10405 | 1 | usable | null | 2.6 | 0.45045 |
| 20261105T084245Z-c1-confirmation-adam_frozen--s10405-1eb3a012-a1 | confirmation | 1 | F | 1.3e-05 | 10405 | 1 | usable | null | 2 | 0.45045 |
| 20261105T084928Z-c1-confirmation-adam_frozen--s10405-edb4027e-a1 | confirmation | 1 | F | 1.83e-05 | 10405 | 1 | usable | null | 1.6 | 0.45045 |
| 20261105T085610Z-c1-confirmation-adamw_prewar-s10405-ec2ea5a6-a1 | confirmation | 1 | P | 1.3e-05 | 10405 | 1 | usable | null | 2 | 0.45045 |
| 20261105T090258Z-c1-confirmation-adamw_prewar-s10405-7d12bf18-a1 | confirmation | 1 | P | 1.85e-05 | 10405 | 1 | usable | null | 2.4 | 0.45045 |
| 20261105T091036Z-c1-confirmation-adamw-1.45e--s10406-e1ce045d-a1 | confirmation | 1 | A | 1.45e-05 | 10406 | 1 | usable | null | 3.4 | 0.45045 |
| 20261105T091629Z-c1-confirmation-adamw-2.11e--s10406-22f3ba19-a1 | confirmation | 1 | A | 2.11e-05 | 10406 | 1 | usable | null | 6 | 0.45045 |
| 20261105T092224Z-c1-confirmation-adam_frozen--s10406-d8218472-a1 | confirmation | 1 | F | 1.3e-05 | 10406 | 1 | usable | null | 3.8 | 0.45045 |
| 20261105T092917Z-c1-confirmation-adam_frozen--s10406-9e30cbb9-a1 | confirmation | 1 | F | 1.83e-05 | 10406 | 1 | usable | null | 6.8 | 0.45045 |
| 20261105T093616Z-c1-confirmation-adamw_prewar-s10406-aca3b584-a1 | confirmation | 1 | P | 1.3e-05 | 10406 | 1 | usable | null | 4 | 0.45045 |
| 20261105T094311Z-c1-confirmation-adamw_prewar-s10406-ea55fbdf-a1 | confirmation | 1 | P | 1.85e-05 | 10406 | 1 | usable | null | 6.8 | 0.45045 |
| 20261105T095059Z-c1-confirmation-adamw-1.45e--s10407-3c639471-a1 | confirmation | 1 | A | 1.45e-05 | 10407 | 1 | usable | null | 0.6 | 0.45045 |
| 20261105T095646Z-c1-confirmation-adamw-2.11e--s10407-62fa123e-a1 | confirmation | 1 | A | 2.11e-05 | 10407 | 1 | usable | null | 1.4 | 0.900901 |
| 20261105T100243Z-c1-confirmation-adam_frozen--s10407-34a3b501-a1 | confirmation | 1 | F | 1.3e-05 | 10407 | 1 | usable | null | 0.6 | 0.45045 |
| 20261105T100932Z-c1-confirmation-adam_frozen--s10407-605fa885-a1 | confirmation | 1 | F | 1.83e-05 | 10407 | 1 | usable | null | 1.8 | 0.45045 |
| 20261105T101627Z-c1-confirmation-adamw_prewar-s10407-78c68b4f-a1 | confirmation | 1 | P | 1.3e-05 | 10407 | 1 | usable | null | 0.8 | 0.45045 |
| 20261105T102324Z-c1-confirmation-adamw_prewar-s10407-67735e48-a1 | confirmation | 1 | P | 1.85e-05 | 10407 | 1 | usable | null | 1.8 | 0.45045 |
| 20261105T103121Z-c1-confirmation-adamw-1.45e--s10408-dbcb16ee-a1 | confirmation | 1 | A | 1.45e-05 | 10408 | 1 | usable | null | 2.2 | 0 |
| 20261105T103647Z-c1-confirmation-adamw-2.11e--s10408-3c9d06a5-a1 | confirmation | 1 | A | 2.11e-05 | 10408 | 1 | usable | null | 1.8 | 0.45045 |
| 20261105T104230Z-c1-confirmation-adam_frozen--s10408-96e3b5d7-a1 | confirmation | 1 | F | 1.3e-05 | 10408 | 1 | usable | null | 2.4 | 0.45045 |
| 20261105T104901Z-c1-confirmation-adam_frozen--s10408-a04acaca-a1 | confirmation | 1 | F | 1.83e-05 | 10408 | 1 | usable | null | 5.6 | 0 |
| 20261105T105543Z-c1-confirmation-adamw_prewar-s10408-9bb29205-a1 | confirmation | 1 | P | 1.3e-05 | 10408 | 1 | usable | null | 2.6 | 0.45045 |
| 20261105T110219Z-c1-confirmation-adamw_prewar-s10408-f7fe1d66-a1 | confirmation | 1 | P | 1.85e-05 | 10408 | 1 | usable | null | 6.6 | 0 |
| 20261105T110951Z-c1-confirmation-adamw-1.45e--s10409-58afcffb-a1 | confirmation | 1 | A | 1.45e-05 | 10409 | 1 | usable | null | 1.6 | 0 |
| 20261105T111507Z-c1-confirmation-adamw-2.11e--s10409-3a580d44-a1 | confirmation | 1 | A | 2.11e-05 | 10409 | 1 | usable | null | 9.6 | -0.45045 |
| 20261105T112034Z-c1-confirmation-adam_frozen--s10409-e3ebd510-a1 | confirmation | 1 | F | 1.3e-05 | 10409 | 1 | usable | null | 1.2 | 0 |
| 20261105T112653Z-c1-confirmation-adam_frozen--s10409-54f3996b-a1 | confirmation | 1 | F | 1.83e-05 | 10409 | 1 | usable | null | 6 | 0 |
| 20261105T113327Z-c1-confirmation-adamw_prewar-s10409-ab36d016-a1 | confirmation | 1 | P | 1.3e-05 | 10409 | 1 | usable | null | 1.2 | 0 |
| 20261105T114001Z-c1-confirmation-adamw_prewar-s10409-fcd0e1e4-a1 | confirmation | 1 | P | 1.85e-05 | 10409 | 1 | usable | null | 6.4 | 0 |
| 20261105T114720Z-c1-confirmation-adamw-1.45e--s10410-7192fddb-a1 | confirmation | 1 | A | 1.45e-05 | 10410 | 1 | usable | null | 1.6 | 0.45045 |
| 20261105T115319Z-c1-confirmation-adamw-2.11e--s10410-1d3fdf0e-a1 | confirmation | 1 | A | 2.11e-05 | 10410 | 1 | usable | null | 1.4 | 0.45045 |
| 20261105T115919Z-c1-confirmation-adam_frozen--s10410-6b3c5534-a1 | confirmation | 1 | F | 1.3e-05 | 10410 | 1 | usable | null | 1.6 | 0.45045 |
| 20261105T120619Z-c1-confirmation-adam_frozen--s10410-577760aa-a1 | confirmation | 1 | F | 1.83e-05 | 10410 | 1 | usable | null | 0.4 | 0.900901 |
| 20261105T121318Z-c1-confirmation-adamw_prewar-s10410-ce3e0c4b-a1 | confirmation | 1 | P | 1.3e-05 | 10410 | 1 | usable | null | 1 | 0.45045 |
| 20261105T122024Z-c1-confirmation-adamw_prewar-s10410-dfe8b72c-a1 | confirmation | 1 | P | 1.85e-05 | 10410 | 1 | usable | null | 0.6 | 0.900901 |
| 20261105T122819Z-c1-confirmation-adamw-1.45e--s10411-117a41c8-a1 | confirmation | 1 | A | 1.45e-05 | 10411 | 1 | usable | null | 1.8 | 0 |
| 20261105T123417Z-c1-confirmation-adamw-2.11e--s10411-c9153ae3-a1 | confirmation | 1 | A | 2.11e-05 | 10411 | 1 | usable | null | 2.4 | 0.45045 |
| 20261105T124026Z-c1-confirmation-adam_frozen--s10411-78eb2e7f-a1 | confirmation | 1 | F | 1.3e-05 | 10411 | 1 | usable | null | 1.6 | 0.45045 |
| 20261105T124824Z-c1-confirmation-adam_frozen--s10411-15239db3-a1 | confirmation | 1 | F | 1.83e-05 | 10411 | 1 | usable | null | 1.8 | 0.45045 |
| 20261105T125605Z-c1-confirmation-adamw_prewar-s10411-e10c9c16-a1 | confirmation | 1 | P | 1.3e-05 | 10411 | 1 | usable | null | 1.4 | 0.45045 |
| 20261105T130317Z-c1-confirmation-adamw_prewar-s10411-f4c3dc43-a1 | confirmation | 1 | P | 1.85e-05 | 10411 | 1 | usable | null | 1.6 | 0.45045 |
| 20261105T131109Z-c1-confirmation-adamw-1.45e--s10412-40161921-a1 | confirmation | 1 | A | 1.45e-05 | 10412 | 1 | usable | null | 3 | 0 |
| 20261105T131726Z-c1-confirmation-adamw-2.11e--s10412-4c0d653e-a1 | confirmation | 1 | A | 2.11e-05 | 10412 | 1 | usable | null | 2.8 | 0.45045 |
| 20261105T132342Z-c1-confirmation-adam_frozen--s10412-3d189a9e-a1 | confirmation | 1 | F | 1.3e-05 | 10412 | 1 | usable | null | 3.4 | 0 |
| 20261105T133122Z-c1-confirmation-adam_frozen--s10412-ddb42053-a1 | confirmation | 1 | F | 1.83e-05 | 10412 | 1 | usable | null | 2.2 | 0.900901 |
| 20261105T133902Z-c1-confirmation-adamw_prewar-s10412-8e6e43cc-a1 | confirmation | 1 | P | 1.3e-05 | 10412 | 1 | usable | null | 3.4 | 0 |
| 20261105T134621Z-c1-confirmation-adamw_prewar-s10412-89a4dd42-a1 | confirmation | 1 | P | 1.85e-05 | 10412 | 1 | usable | null | 2 | 0.900901 |
| 20261105T135418Z-c1-confirmation-adamw-1.45e--s10413-f5a99bbf-a1 | confirmation | 1 | A | 1.45e-05 | 10413 | 1 | usable | null | 3 | 0 |
| 20261105T135959Z-c1-confirmation-adamw-2.11e--s10413-a73231ba-a1 | confirmation | 1 | A | 2.11e-05 | 10413 | 1 | usable | null | 3.8 | 0 |
| 20261105T140555Z-c1-confirmation-adam_frozen--s10413-58af40c6-a1 | confirmation | 1 | F | 1.3e-05 | 10413 | 1 | usable | null | 2 | 0 |
| 20261105T141236Z-c1-confirmation-adam_frozen--s10413-4d6c07d7-a1 | confirmation | 1 | F | 1.83e-05 | 10413 | 1 | usable | null | 3.8 | 0 |
| 20261105T141930Z-c1-confirmation-adamw_prewar-s10413-da232883-a1 | confirmation | 1 | P | 1.3e-05 | 10413 | 1 | usable | null | 2 | 0 |
| 20261105T142620Z-c1-confirmation-adamw_prewar-s10413-849dc5d0-a1 | confirmation | 1 | P | 1.85e-05 | 10413 | 1 | usable | null | 3.8 | 0 |
| 20261105T143412Z-c1-confirmation-adamw-1.45e--s10414-c9bd9a7b-a1 | confirmation | 1 | A | 1.45e-05 | 10414 | 1 | usable | null | 2.2 | 0.45045 |
| 20261105T143957Z-c1-confirmation-adamw-2.11e--s10414-1e74ceb1-a1 | confirmation | 1 | A | 2.11e-05 | 10414 | 1 | usable | null | 3.8 | 0 |
| 20261105T144547Z-c1-confirmation-adam_frozen--s10414-e392cc59-a1 | confirmation | 1 | F | 1.3e-05 | 10414 | 1 | usable | null | 3 | 0.45045 |
| 20261105T145244Z-c1-confirmation-adam_frozen--s10414-f8fc56df-a1 | confirmation | 1 | F | 1.83e-05 | 10414 | 1 | usable | null | 2.6 | 0.900901 |
| 20261105T145940Z-c1-confirmation-adamw_prewar-s10414-f7981dff-a1 | confirmation | 1 | P | 1.3e-05 | 10414 | 1 | usable | null | 3.4 | 0.45045 |
| 20261105T150646Z-c1-confirmation-adamw_prewar-s10414-766f8fe4-a1 | confirmation | 1 | P | 1.85e-05 | 10414 | 1 | scientific | null | null | null |
| 20261105T151054Z-c1-confirmation-adamw-1.45e--s10415-045f4b83-a1 | confirmation | 1 | A | 1.45e-05 | 10415 | 1 | usable | null | 1.4 | 0 |
| 20261105T151642Z-c1-confirmation-adamw-2.11e--s10415-2ea0cd4c-a1 | confirmation | 1 | A | 2.11e-05 | 10415 | 1 | usable | null | 2.2 | 0.45045 |
| 20261105T152240Z-c1-confirmation-adam_frozen--s10415-67ae11c0-a1 | confirmation | 1 | F | 1.3e-05 | 10415 | 1 | usable | null | 1.8 | 0.45045 |
| 20261105T152929Z-c1-confirmation-adam_frozen--s10415-43e4aacd-a1 | confirmation | 1 | F | 1.83e-05 | 10415 | 1 | usable | null | 0.8 | 0 |
| 20261105T153627Z-c1-confirmation-adamw_prewar-s10415-b36ba660-a1 | confirmation | 1 | P | 1.3e-05 | 10415 | 1 | usable | null | 1.8 | 0 |
| 20261105T154334Z-c1-confirmation-adamw_prewar-s10415-a8a33d9a-a1 | confirmation | 1 | P | 1.85e-05 | 10415 | 1 | usable | null | 1 | 0 |
| 20261105T155129Z-c1-confirmation-adamw-1.45e--s10416-ea1c0f98-a1 | confirmation | 1 | A | 1.45e-05 | 10416 | 1 | usable | null | 0.8 | 0 |
| 20261105T155707Z-c1-confirmation-adamw-2.11e--s10416-4b86d833-a1 | confirmation | 1 | A | 2.11e-05 | 10416 | 1 | usable | null | 3.6 | 0.45045 |
| 20261105T160304Z-c1-confirmation-adam_frozen--s10416-0ff06754-a1 | confirmation | 1 | F | 1.3e-05 | 10416 | 1 | usable | null | 0.8 | 0 |
| 20261105T160943Z-c1-confirmation-adam_frozen--s10416-6822f765-a1 | confirmation | 1 | F | 1.83e-05 | 10416 | 1 | usable | null | 3.2 | 0 |
| 20261105T161635Z-c1-confirmation-adamw_prewar-s10416-cf82cc26-a1 | confirmation | 1 | P | 1.3e-05 | 10416 | 1 | usable | null | 0.8 | 0 |
| 20261105T162321Z-c1-confirmation-adamw_prewar-s10416-6651afa0-a1 | confirmation | 1 | P | 1.85e-05 | 10416 | 1 | usable | null | 3 | 0 |
| 20261105T163113Z-c1-confirmation-adamw-1.45e--s10417-910700b0-a1 | confirmation | 1 | A | 1.45e-05 | 10417 | 1 | usable | null | 3 | 0.45045 |
| 20261105T163707Z-c1-confirmation-adamw-2.11e--s10417-01f939fd-a1 | confirmation | 1 | A | 2.11e-05 | 10417 | 1 | usable | null | 3.2 | 0.45045 |
| 20261105T164311Z-c1-confirmation-adam_frozen--s10417-d9628054-a1 | confirmation | 1 | F | 1.3e-05 | 10417 | 1 | usable | null | 3.2 | 0.45045 |
| 20261105T165023Z-c1-confirmation-adam_frozen--s10417-46f283fb-a1 | confirmation | 1 | F | 1.83e-05 | 10417 | 1 | usable | null | 2.6 | 0.45045 |
| 20261105T165728Z-c1-confirmation-adamw_prewar-s10417-1cf3412c-a1 | confirmation | 1 | P | 1.3e-05 | 10417 | 1 | usable | null | 3.2 | 0.45045 |
| 20261105T170448Z-c1-confirmation-adamw_prewar-s10417-58003cb7-a1 | confirmation | 1 | P | 1.85e-05 | 10417 | 1 | usable | null | 2.2 | 0.45045 |
| 20261105T171540Z-c1-confirmation-adamw-1.45e--s10418-93e70590-a1 | confirmation | 1 | A | 1.45e-05 | 10418 | 1 | usable | null | 2.4 | 0 |
| 20261105T172132Z-c1-confirmation-adamw-2.11e--s10418-9c83a44e-a1 | confirmation | 1 | A | 2.11e-05 | 10418 | 1 | usable | null | 3 | 0.45045 |
| 20261105T172729Z-c1-confirmation-adam_frozen--s10418-273760e2-a1 | confirmation | 1 | F | 1.3e-05 | 10418 | 1 | usable | null | 3.2 | 0 |
| 20261105T173410Z-c1-confirmation-adam_frozen--s10418-e3156ed5-a1 | confirmation | 1 | F | 1.83e-05 | 10418 | 1 | usable | null | 7.4 | 0.45045 |
| 20261105T174106Z-c1-confirmation-adamw_prewar-s10418-6baae8cf-a1 | confirmation | 1 | P | 1.3e-05 | 10418 | 1 | usable | null | 3.2 | 0 |
| 20261105T174804Z-c1-confirmation-adamw_prewar-s10418-d3f728ab-a1 | confirmation | 1 | P | 1.85e-05 | 10418 | 1 | usable | null | 6.4 | 0.45045 |
| 20261105T175559Z-c1-confirmation-adamw-1.45e--s10419-19d3cd1e-a1 | confirmation | 1 | A | 1.45e-05 | 10419 | 1 | usable | null | 1.4 | 0 |
| 20261105T180156Z-c1-confirmation-adamw-2.11e--s10419-f26c1619-a1 | confirmation | 1 | A | 2.11e-05 | 10419 | 1 | usable | null | 1.8 | 0.45045 |
| 20261105T180807Z-c1-confirmation-adam_frozen--s10419-0539efc0-a1 | confirmation | 1 | F | 1.3e-05 | 10419 | 1 | usable | null | 1 | 0 |
| 20261105T181511Z-c1-confirmation-adam_frozen--s10419-ecaab3c3-a1 | confirmation | 1 | F | 1.83e-05 | 10419 | 1 | usable | null | 1.6 | 0.45045 |
| 20261105T182222Z-c1-confirmation-adamw_prewar-s10419-184dacbf-a1 | confirmation | 1 | P | 1.3e-05 | 10419 | 1 | usable | null | 1 | 0 |
| 20261105T182929Z-c1-confirmation-adamw_prewar-s10419-9e233b70-a1 | confirmation | 1 | P | 1.85e-05 | 10419 | 1 | usable | null | 1.8 | 0.45045 |
| 20261105T183729Z-c1-confirmation-adamw-1.45e--s10420-bc45682e-a1 | confirmation | 1 | A | 1.45e-05 | 10420 | 1 | usable | null | 0.8 | 0 |
| 20261105T184314Z-c1-confirmation-adamw-2.11e--s10420-1b0afb84-a1 | confirmation | 1 | A | 2.11e-05 | 10420 | 1 | usable | null | 1.4 | 0.45045 |
| 20261105T184915Z-c1-confirmation-adam_frozen--s10420-d2b3957d-a1 | confirmation | 1 | F | 1.3e-05 | 10420 | 1 | usable | null | 1.2 | 0 |
| 20261105T185613Z-c1-confirmation-adam_frozen--s10420-cd7b2ba4-a1 | confirmation | 1 | F | 1.83e-05 | 10420 | 1 | usable | null | 1.4 | 0.45045 |
| 20261105T190328Z-c1-confirmation-adamw_prewar-s10420-7c4e8ee4-a1 | confirmation | 1 | P | 1.3e-05 | 10420 | 1 | usable | null | 1.2 | 0 |
| 20261105T191041Z-c1-confirmation-adamw_prewar-s10420-cb97827a-a1 | confirmation | 1 | P | 1.85e-05 | 10420 | 1 | usable | null | 1.8 | 0 |
| 20261105T191856Z-c1-confirmation-adamw-1.45e--s10421-8a5b6b7b-a1 | confirmation | 1 | A | 1.45e-05 | 10421 | 1 | usable | null | 1.8 | 0 |
| 20261105T192446Z-c1-confirmation-adamw-2.11e--s10421-f58dcf90-a1 | confirmation | 1 | A | 2.11e-05 | 10421 | 1 | usable | null | 2 | 0.45045 |
| 20261105T193034Z-c1-confirmation-adam_frozen--s10421-4a5bd5a2-a1 | confirmation | 1 | F | 1.3e-05 | 10421 | 1 | usable | null | 1.8 | 0 |
| 20261105T193725Z-c1-confirmation-adam_frozen--s10421-b93a4ad1-a1 | confirmation | 1 | F | 1.83e-05 | 10421 | 1 | usable | null | 1.2 | -0.45045 |
| 20261105T194419Z-c1-confirmation-adamw_prewar-s10421-61004c38-a1 | confirmation | 1 | P | 1.3e-05 | 10421 | 1 | usable | null | 2 | 0 |
| 20261105T195117Z-c1-confirmation-adamw_prewar-s10421-b87ffa29-a1 | confirmation | 1 | P | 1.85e-05 | 10421 | 1 | usable | null | 1.2 | -0.45045 |
| 20261105T195913Z-c1-confirmation-adamw-1.45e--s10422-edacce41-a1 | confirmation | 1 | A | 1.45e-05 | 10422 | 1 | usable | null | 2.6 | 0 |
| 20261105T200457Z-c1-confirmation-adamw-2.11e--s10422-d04e6282-a1 | confirmation | 1 | A | 2.11e-05 | 10422 | 1 | usable | null | 3.6 | 0 |
| 20261105T201036Z-c1-confirmation-adam_frozen--s10422-c5555632-a1 | confirmation | 1 | F | 1.3e-05 | 10422 | 1 | usable | null | 2.6 | 0 |
| 20261105T201704Z-c1-confirmation-adam_frozen--s10422-f5200e5a-a1 | confirmation | 1 | F | 1.83e-05 | 10422 | 1 | usable | null | 3 | 0 |
| 20261105T202344Z-c1-confirmation-adamw_prewar-s10422-4b1c2dc5-a1 | confirmation | 1 | P | 1.3e-05 | 10422 | 1 | usable | null | 2.6 | 0 |
| 20261105T203021Z-c1-confirmation-adamw_prewar-s10422-264a4a74-a1 | confirmation | 1 | P | 1.85e-05 | 10422 | 1 | usable | null | 3 | 0 |
| 20261105T203755Z-c1-confirmation-adamw-1.45e--s10423-86e6f4ce-a1 | confirmation | 1 | A | 1.45e-05 | 10423 | 1 | usable | null | 1.6 | 0 |
| 20261105T204319Z-c1-confirmation-adamw-2.11e--s10423-545c9cd0-a1 | confirmation | 1 | A | 2.11e-05 | 10423 | 1 | usable | null | 2.4 | 0 |
| 20261105T204916Z-c1-confirmation-adam_frozen--s10423-60e64f7e-a1 | confirmation | 1 | F | 1.3e-05 | 10423 | 1 | usable | null | 1.6 | -0.45045 |
| 20261105T205543Z-c1-confirmation-adam_frozen--s10423-e35ce75e-a1 | confirmation | 1 | F | 1.83e-05 | 10423 | 1 | usable | null | 1.4 | 0 |
| 20261105T210234Z-c1-confirmation-adamw_prewar-s10423-40144c78-a1 | confirmation | 1 | P | 1.3e-05 | 10423 | 1 | usable | null | 1.6 | -0.45045 |
| 20261105T210911Z-c1-confirmation-adamw_prewar-s10423-efbc1e8a-a1 | confirmation | 1 | P | 1.85e-05 | 10423 | 1 | usable | null | 1 | 0 |
| 20261105T211702Z-c1-confirmation-adamw-1.45e--s10424-78e20f40-a1 | confirmation | 1 | A | 1.45e-05 | 10424 | 1 | usable | null | 1.6 | 0 |
| 20261105T212302Z-c1-confirmation-adamw-2.11e--s10424-b7fe2c54-a1 | confirmation | 1 | A | 2.11e-05 | 10424 | 1 | usable | null | 1.6 | -0.45045 |
| 20261105T212904Z-c1-confirmation-adam_frozen--s10424-69efaa65-a1 | confirmation | 1 | F | 1.3e-05 | 10424 | 1 | usable | null | 1.8 | 0 |
| 20261105T213605Z-c1-confirmation-adam_frozen--s10424-969193cc-a1 | confirmation | 1 | F | 1.83e-05 | 10424 | 1 | usable | null | 3.2 | -0.45045 |
| 20261105T214305Z-c1-confirmation-adamw_prewar-s10424-86604e80-a1 | confirmation | 1 | P | 1.3e-05 | 10424 | 1 | usable | null | 1.8 | 0 |
| 20261105T215014Z-c1-confirmation-adamw_prewar-s10424-a135fb2c-a1 | confirmation | 1 | P | 1.85e-05 | 10424 | 1 | usable | null | 3.2 | -0.45045 |
| 20261105T215814Z-c1-confirmation-adamw-1.45e--s10425-39799eed-a1 | confirmation | 1 | A | 1.45e-05 | 10425 | 1 | usable | null | 4 | 0 |
| 20261105T220404Z-c1-confirmation-adamw-2.11e--s10425-f78a70e9-a1 | confirmation | 1 | A | 2.11e-05 | 10425 | 1 | usable | null | 3.4 | 0.900901 |
| 20261105T221007Z-c1-confirmation-adam_frozen--s10425-01809d0e-a1 | confirmation | 1 | F | 1.3e-05 | 10425 | 1 | usable | null | 3 | 0.45045 |
| 20261105T221706Z-c1-confirmation-adam_frozen--s10425-b1b6d153-a1 | confirmation | 1 | F | 1.83e-05 | 10425 | 1 | usable | null | 4.2 | 1.35135 |
| 20261105T222357Z-c1-confirmation-adamw_prewar-s10425-c7361330-a1 | confirmation | 1 | P | 1.3e-05 | 10425 | 1 | usable | null | 2.6 | 0.45045 |
| 20261105T223104Z-c1-confirmation-adamw_prewar-s10425-d509f01e-a1 | confirmation | 1 | P | 1.85e-05 | 10425 | 1 | usable | null | 4 | 1.35135 |
| 20261105T223851Z-c1-confirmation-adamw-1.45e--s10426-35c26c9f-a1 | confirmation | 1 | A | 1.45e-05 | 10426 | 1 | usable | null | 2.4 | -0.45045 |
| 20261105T224438Z-c1-confirmation-adamw-2.11e--s10426-75057582-a1 | confirmation | 1 | A | 2.11e-05 | 10426 | 1 | usable | null | 2.2 | 0.45045 |
| 20261105T225037Z-c1-confirmation-adam_frozen--s10426-dc5c9c0f-a1 | confirmation | 1 | F | 1.3e-05 | 10426 | 1 | usable | null | 1.6 | -0.45045 |
| 20261105T225714Z-c1-confirmation-adam_frozen--s10426-82bf1eff-a1 | confirmation | 1 | F | 1.83e-05 | 10426 | 1 | usable | null | 3.4 | 0 |
| 20261105T230411Z-c1-confirmation-adamw_prewar-s10426-a7125f6f-a1 | confirmation | 1 | P | 1.3e-05 | 10426 | 1 | usable | null | 1.6 | -0.45045 |
| 20261105T231056Z-c1-confirmation-adamw_prewar-s10426-0df82a95-a1 | confirmation | 1 | P | 1.85e-05 | 10426 | 1 | usable | null | 3 | 0.45045 |
| 20261105T231856Z-c1-confirmation-adamw-1.45e--s10427-4e1cbdc0-a1 | confirmation | 1 | A | 1.45e-05 | 10427 | 1 | usable | null | 1.6 | 0 |
| 20261105T232456Z-c1-confirmation-adamw-2.11e--s10427-3eb2b6f7-a1 | confirmation | 1 | A | 2.11e-05 | 10427 | 1 | usable | null | 2 | 0.45045 |
| 20261105T233056Z-c1-confirmation-adam_frozen--s10427-1773b720-a1 | confirmation | 1 | F | 1.3e-05 | 10427 | 1 | usable | null | 2.6 | 0 |
| 20261105T233757Z-c1-confirmation-adam_frozen--s10427-2869adc9-a1 | confirmation | 1 | F | 1.83e-05 | 10427 | 1 | usable | null | 2 | 0.45045 |
| 20261105T234458Z-c1-confirmation-adamw_prewar-s10427-9755cacd-a1 | confirmation | 1 | P | 1.3e-05 | 10427 | 1 | usable | null | 2.6 | 0 |
| 20261105T235203Z-c1-confirmation-adamw_prewar-s10427-37755855-a1 | confirmation | 1 | P | 1.85e-05 | 10427 | 1 | usable | null | 2.2 | 0.45045 |
| 20261106T000001Z-c1-confirmation-adamw-1.45e--s10428-c9c6c35b-a1 | confirmation | 1 | A | 1.45e-05 | 10428 | 1 | usable | null | -0.2 | 0 |
| 20261106T000556Z-c1-confirmation-adamw-2.11e--s10428-a608fd9d-a1 | confirmation | 1 | A | 2.11e-05 | 10428 | 1 | usable | null | 1.4 | 0.45045 |
| 20261106T001151Z-c1-confirmation-adam_frozen--s10428-583d2b2b-a1 | confirmation | 1 | F | 1.3e-05 | 10428 | 1 | usable | null | 0.8 | 0 |
| 20261106T001830Z-c1-confirmation-adam_frozen--s10428-a65b39c9-a1 | confirmation | 1 | F | 1.83e-05 | 10428 | 1 | usable | null | 4.4 | 1.35135 |
| 20261106T002538Z-c1-confirmation-adamw_prewar-s10428-5f320a89-a1 | confirmation | 1 | P | 1.3e-05 | 10428 | 1 | usable | null | 0.4 | 0.45045 |
| 20261106T003232Z-c1-confirmation-adamw_prewar-s10428-41b523b0-a1 | confirmation | 1 | P | 1.85e-05 | 10428 | 1 | usable | null | 0.8 | 0.900901 |
| 20261106T004027Z-c1-confirmation-adamw-1.45e--s10429-81292270-a1 | confirmation | 1 | A | 1.45e-05 | 10429 | 1 | usable | null | 2.4 | 0.45045 |
| 20261106T004615Z-c1-confirmation-adamw-2.11e--s10429-b0963d48-a1 | confirmation | 1 | A | 2.11e-05 | 10429 | 1 | usable | null | 5 | 0.45045 |
| 20261106T005204Z-c1-confirmation-adam_frozen--s10429-e65aaa16-a1 | confirmation | 1 | F | 1.3e-05 | 10429 | 1 | usable | null | 2.4 | 0.45045 |
| 20261106T005842Z-c1-confirmation-adam_frozen--s10429-3ba8c3e0-a1 | confirmation | 1 | F | 1.83e-05 | 10429 | 1 | usable | null | 3.2 | 0.45045 |
| 20261106T010520Z-c1-confirmation-adamw_prewar-s10429-67e85e45-a1 | confirmation | 1 | P | 1.3e-05 | 10429 | 1 | usable | null | 2.4 | 0.45045 |
| 20261106T011204Z-c1-confirmation-adamw_prewar-s10429-97d1507e-a1 | confirmation | 1 | P | 1.85e-05 | 10429 | 1 | usable | null | 2.8 | 0.45045 |
| 20261106T011948Z-c1-confirmation-adamw-1.45e--s10430-7ee8c181-a1 | confirmation | 1 | A | 1.45e-05 | 10430 | 1 | usable | null | 2 | 0 |
| 20261106T012537Z-c1-confirmation-adamw-2.11e--s10430-e7248105-a1 | confirmation | 1 | A | 2.11e-05 | 10430 | 1 | usable | null | 3.4 | 0.45045 |
| 20261106T013134Z-c1-confirmation-adam_frozen--s10430-bd969c64-a1 | confirmation | 1 | F | 1.3e-05 | 10430 | 1 | usable | null | 2.8 | 0 |
| 20261106T013828Z-c1-confirmation-adam_frozen--s10430-07fefc41-a1 | confirmation | 1 | F | 1.83e-05 | 10430 | 1 | usable | null | 3 | 0.45045 |
| 20261106T014543Z-c1-confirmation-adamw_prewar-s10430-766850f4-a1 | confirmation | 1 | P | 1.3e-05 | 10430 | 1 | usable | null | 2.6 | 0 |
| 20261106T015253Z-c1-confirmation-adamw_prewar-s10430-f7b13dc1-a1 | confirmation | 1 | P | 1.85e-05 | 10430 | 1 | usable | null | 3.2 | 0.45045 |
| 20261106T020119Z-c1-confirmation-adamw-1.45e--s10431-b450976f-a1 | confirmation | 1 | A | 1.45e-05 | 10431 | 1 | usable | null | 3.6 | 0.45045 |
| 20261106T020708Z-c1-confirmation-adamw-2.11e--s10431-7601c848-a1 | confirmation | 1 | A | 2.11e-05 | 10431 | 1 | usable | null | 4.6 | 0.900901 |
| 20261106T021321Z-c1-confirmation-adam_frozen--s10431-6126cfc1-a1 | confirmation | 1 | F | 1.3e-05 | 10431 | 1 | usable | null | 4 | 0.45045 |
| 20261106T022019Z-c1-confirmation-adam_frozen--s10431-64a012cc-a1 | confirmation | 1 | F | 1.83e-05 | 10431 | 1 | usable | null | 7.2 | 0.45045 |
| 20261106T022725Z-c1-confirmation-adamw_prewar-s10431-4a51db0c-a1 | confirmation | 1 | P | 1.3e-05 | 10431 | 1 | usable | null | 4 | 0.45045 |
| 20261106T023435Z-c1-confirmation-adamw_prewar-s10431-7f2ed9b0-a1 | confirmation | 1 | P | 1.85e-05 | 10431 | 1 | usable | null | 7.4 | 0.45045 |
| 20261106T024247Z-c1-confirmation-adamw-1.45e--s10432-6119d069-a1 | confirmation | 1 | A | 1.45e-05 | 10432 | 1 | usable | null | 3 | 0 |
| 20261106T024900Z-c1-confirmation-adamw-2.11e--s10432-bf49c43b-a1 | confirmation | 1 | A | 2.11e-05 | 10432 | 1 | usable | null | 3 | 0 |
| 20261106T025516Z-c1-confirmation-adam_frozen--s10432-20f01b57-a1 | confirmation | 1 | F | 1.3e-05 | 10432 | 1 | usable | null | 2.4 | 0 |
| 20261106T030231Z-c1-confirmation-adam_frozen--s10432-557823ed-a1 | confirmation | 1 | F | 1.83e-05 | 10432 | 1 | usable | null | 3 | 0 |
| 20261106T030947Z-c1-confirmation-adamw_prewar-s10432-b8ec7883-a1 | confirmation | 1 | P | 1.3e-05 | 10432 | 1 | usable | null | 2.4 | 0 |
| 20261106T031708Z-c1-confirmation-adamw_prewar-s10432-3d3ce87b-a1 | confirmation | 1 | P | 1.85e-05 | 10432 | 1 | usable | null | 3.4 | 0 |
| 20261106T032522Z-c1-confirmation-adamw-1.45e--s10433-a1830825-a1 | confirmation | 1 | A | 1.45e-05 | 10433 | 1 | usable | null | 1.6 | 0 |
| 20261106T033111Z-c1-confirmation-adamw-2.11e--s10433-a173166c-a1 | confirmation | 1 | A | 2.11e-05 | 10433 | 1 | usable | null | 1.4 | 0.45045 |
| 20261106T033708Z-c1-confirmation-adam_frozen--s10433-3df6abc9-a1 | confirmation | 1 | F | 1.3e-05 | 10433 | 1 | usable | null | 0.8 | 0.45045 |
| 20261106T034404Z-c1-confirmation-adam_frozen--s10433-dc9eb320-a1 | confirmation | 1 | F | 1.83e-05 | 10433 | 1 | usable | null | -0.6 | 1.35135 |
| 20261106T035100Z-c1-confirmation-adamw_prewar-s10433-c165bb9c-a1 | confirmation | 1 | P | 1.3e-05 | 10433 | 1 | usable | null | 1 | 0.45045 |
| 20261106T035803Z-c1-confirmation-adamw_prewar-s10433-e7762d3e-a1 | confirmation | 1 | P | 1.85e-05 | 10433 | 1 | usable | null | -1.2 | 0.900901 |
| 20261106T040601Z-c1-confirmation-adamw-1.45e--s10434-df15065c-a1 | confirmation | 1 | A | 1.45e-05 | 10434 | 1 | usable | null | 1 | 0 |
| 20261106T041157Z-c1-confirmation-adamw-2.11e--s10434-7ef76fd9-a1 | confirmation | 1 | A | 2.11e-05 | 10434 | 1 | usable | null | 3.2 | 0.900901 |
| 20261106T041753Z-c1-confirmation-adam_frozen--s10434-fadbe7f8-a1 | confirmation | 1 | F | 1.3e-05 | 10434 | 1 | usable | null | 2.6 | 0 |
| 20261106T042449Z-c1-confirmation-adam_frozen--s10434-119d457d-a1 | confirmation | 1 | F | 1.83e-05 | 10434 | 1 | usable | null | 5 | 0 |
| 20261106T043144Z-c1-confirmation-adamw_prewar-s10434-c4405c25-a1 | confirmation | 1 | P | 1.3e-05 | 10434 | 1 | usable | null | 2.6 | 0 |
| 20261106T043847Z-c1-confirmation-adamw_prewar-s10434-488fcac6-a1 | confirmation | 1 | P | 1.85e-05 | 10434 | 1 | usable | null | 4.8 | 0 |
| 20261106T044645Z-c1-confirmation-adamw-1.45e--s10435-526b6742-a1 | confirmation | 1 | A | 1.45e-05 | 10435 | 1 | usable | null | 1.4 | 0.45045 |
| 20261106T045242Z-c1-confirmation-adamw-2.11e--s10435-0ee2e522-a1 | confirmation | 1 | A | 2.11e-05 | 10435 | 1 | usable | null | 1.8 | 0.45045 |
| 20261106T045838Z-c1-confirmation-adam_frozen--s10435-ee601b33-a1 | confirmation | 1 | F | 1.3e-05 | 10435 | 1 | usable | null | 1.8 | 0.45045 |
| 20261106T050533Z-c1-confirmation-adam_frozen--s10435-10a49311-a1 | confirmation | 1 | F | 1.83e-05 | 10435 | 1 | usable | null | 2.4 | 0.45045 |
| 20261106T051228Z-c1-confirmation-adamw_prewar-s10435-0a606037-a1 | confirmation | 1 | P | 1.3e-05 | 10435 | 1 | usable | null | 1.8 | 0.45045 |
| 20261106T051931Z-c1-confirmation-adamw_prewar-s10435-6ba72121-a1 | confirmation | 1 | P | 1.85e-05 | 10435 | 1 | usable | null | 2.4 | 0.45045 |
| 20261106T052730Z-c1-confirmation-adamw-1.45e--s10436-8f2ed530-a1 | confirmation | 1 | A | 1.45e-05 | 10436 | 1 | usable | null | 3 | 0 |
| 20261106T053346Z-c1-confirmation-adamw-2.11e--s10436-09e9d7d0-a1 | confirmation | 1 | A | 2.11e-05 | 10436 | 1 | usable | null | 3 | 0 |
| 20261106T054006Z-c1-confirmation-adam_frozen--s10436-8a52448f-a1 | confirmation | 1 | F | 1.3e-05 | 10436 | 1 | usable | null | 2.8 | 0 |
| 20261106T054724Z-c1-confirmation-adam_frozen--s10436-62a7f4d9-a1 | confirmation | 1 | F | 1.83e-05 | 10436 | 1 | usable | null | 3.4 | 0 |
| 20261106T055440Z-c1-confirmation-adamw_prewar-s10436-9c1bf40b-a1 | confirmation | 1 | P | 1.3e-05 | 10436 | 1 | usable | null | 2.8 | 0 |
| 20261106T060200Z-c1-confirmation-adamw_prewar-s10436-afe92eef-a1 | confirmation | 1 | P | 1.85e-05 | 10436 | 1 | usable | null | 3 | 0 |
| 20261106T061019Z-c1-confirmation-adamw-1.45e--s10437-2d02cfa4-a1 | confirmation | 1 | A | 1.45e-05 | 10437 | 1 | usable | null | 1.8 | 0.45045 |
| 20261106T061619Z-c1-confirmation-adamw-2.11e--s10437-475d699e-a1 | confirmation | 1 | A | 2.11e-05 | 10437 | 1 | usable | null | 3.8 | 0.900901 |
| 20261106T062217Z-c1-confirmation-adam_frozen--s10437-ffc8c12f-a1 | confirmation | 1 | F | 1.3e-05 | 10437 | 1 | usable | null | 2.4 | 0.45045 |
| 20261106T062917Z-c1-confirmation-adam_frozen--s10437-84e4ed22-a1 | confirmation | 1 | F | 1.83e-05 | 10437 | 1 | usable | null | 4.6 | 1.35135 |
| 20261106T063616Z-c1-confirmation-adamw_prewar-s10437-a2724948-a1 | confirmation | 1 | P | 1.3e-05 | 10437 | 1 | usable | null | 2.4 | 0.45045 |
| 20261106T064323Z-c1-confirmation-adamw_prewar-s10437-b0e2d329-a1 | confirmation | 1 | P | 1.85e-05 | 10437 | 1 | usable | null | 3.6 | 1.35135 |
| 20261106T065130Z-c1-confirmation-adamw-1.45e--s10438-505849a9-a1 | confirmation | 1 | A | 1.45e-05 | 10438 | 1 | usable | null | 3.6 | 0.45045 |
| 20261106T065729Z-c1-confirmation-adamw-2.11e--s10438-d8caab8f-a1 | confirmation | 1 | A | 2.11e-05 | 10438 | 1 | usable | null | 3.4 | 0.900901 |
| 20261106T070330Z-c1-confirmation-adam_frozen--s10438-2abf2415-a1 | confirmation | 1 | F | 1.3e-05 | 10438 | 1 | usable | null | 3.2 | 0.45045 |
| 20261106T071032Z-c1-confirmation-adam_frozen--s10438-8b0a1461-a1 | confirmation | 1 | F | 1.83e-05 | 10438 | 1 | usable | null | 5.8 | 0.900901 |
| 20261106T071736Z-c1-confirmation-adamw_prewar-s10438-99fff2aa-a1 | confirmation | 1 | P | 1.3e-05 | 10438 | 1 | usable | null | 3.6 | 0.45045 |
| 20261106T072442Z-c1-confirmation-adamw_prewar-s10438-5d922af5-a1 | confirmation | 1 | P | 1.85e-05 | 10438 | 1 | usable | null | 6.8 | 1.35135 |
| 20261106T073245Z-c1-confirmation-adamw-1.45e--s10439-876c48e5-a1 | confirmation | 1 | A | 1.45e-05 | 10439 | 1 | usable | null | 1.8 | 0 |
| 20261106T073902Z-c1-confirmation-adamw-2.11e--s10439-193cf5c6-a1 | confirmation | 1 | A | 2.11e-05 | 10439 | 1 | usable | null | 2 | 0.45045 |
| 20261106T074514Z-c1-confirmation-adam_frozen--s10439-5badb301-a1 | confirmation | 1 | F | 1.3e-05 | 10439 | 1 | usable | null | 1.2 | 0 |
| 20261106T075225Z-c1-confirmation-adam_frozen--s10439-86712ce0-a1 | confirmation | 1 | F | 1.83e-05 | 10439 | 1 | usable | null | 1.8 | 1.35135 |
| 20261106T075930Z-c1-confirmation-adamw_prewar-s10439-58d812ea-a1 | confirmation | 1 | P | 1.3e-05 | 10439 | 1 | usable | null | 0.4 | 0 |
| 20261106T080650Z-c1-confirmation-adamw_prewar-s10439-83b15d3c-a1 | confirmation | 1 | P | 1.85e-05 | 10439 | 1 | usable | null | 0.6 | 0.900901 |
| 20261106T081508Z-c1-confirmation-adamw-1.45e--s10440-4e2322ff-a1 | confirmation | 1 | A | 1.45e-05 | 10440 | 1 | usable | null | 2.2 | 0.45045 |
| 20261106T082122Z-c1-confirmation-adamw-2.11e--s10440-bdce69ce-a1 | confirmation | 1 | A | 2.11e-05 | 10440 | 1 | usable | null | 2.2 | 0.45045 |
| 20261106T082738Z-c1-confirmation-adam_frozen--s10440-eb1dfe56-a1 | confirmation | 1 | F | 1.3e-05 | 10440 | 1 | usable | null | 3 | 0.45045 |
| 20261106T083446Z-c1-confirmation-adam_frozen--s10440-7c105018-a1 | confirmation | 1 | F | 1.83e-05 | 10440 | 1 | usable | null | 4.4 | 0 |
| 20261106T084201Z-c1-confirmation-adamw_prewar-s10440-c2a1c9bd-a1 | confirmation | 1 | P | 1.3e-05 | 10440 | 1 | usable | null | 3.2 | 0.45045 |
| 20261106T084919Z-c1-confirmation-adamw_prewar-s10440-7abf1dca-a1 | confirmation | 1 | P | 1.85e-05 | 10440 | 1 | usable | null | 4 | 0.45045 |
| 20261106T085738Z-c1-confirmation-adamw-1.45e--s10441-ee07330c-a1 | confirmation | 1 | A | 1.45e-05 | 10441 | 1 | usable | null | 0.8 | 0 |
| 20261106T090334Z-c1-confirmation-adamw-2.11e--s10441-9064d65b-a1 | confirmation | 1 | A | 2.11e-05 | 10441 | 1 | usable | null | 2.2 | 0 |
| 20261106T090929Z-c1-confirmation-adam_frozen--s10441-7e4ba88a-a1 | confirmation | 1 | F | 1.3e-05 | 10441 | 1 | usable | null | 1.8 | 0 |
| 20261106T091633Z-c1-confirmation-adam_frozen--s10441-afaab4eb-a1 | confirmation | 1 | F | 1.83e-05 | 10441 | 1 | usable | null | 9.2 | 0 |
| 20261106T092347Z-c1-confirmation-adamw_prewar-s10441-bc879582-a1 | confirmation | 1 | P | 1.3e-05 | 10441 | 1 | usable | null | 2 | 0 |
| 20261106T093102Z-c1-confirmation-adamw_prewar-s10441-e72c79e7-a1 | confirmation | 1 | P | 1.85e-05 | 10441 | 1 | usable | null | 12.2 | 0 |
| 20261106T093913Z-c1-confirmation-adamw-1.45e--s10442-d546b246-a1 | confirmation | 1 | A | 1.45e-05 | 10442 | 1 | usable | null | 2.6 | 0 |
| 20261106T094508Z-c1-confirmation-adamw-2.11e--s10442-020f6463-a1 | confirmation | 1 | A | 2.11e-05 | 10442 | 1 | usable | null | 3 | 0.45045 |
| 20261106T095109Z-c1-confirmation-adam_frozen--s10442-0b388ee6-a1 | confirmation | 1 | F | 1.3e-05 | 10442 | 1 | usable | null | 2.6 | 0 |
| 20261106T095759Z-c1-confirmation-adam_frozen--s10442-55f08873-a1 | confirmation | 1 | F | 1.83e-05 | 10442 | 1 | usable | null | 3.4 | 0 |
| 20261106T100459Z-c1-confirmation-adamw_prewar-s10442-150e42ad-a1 | confirmation | 1 | P | 1.3e-05 | 10442 | 1 | usable | null | 2.6 | 0 |
| 20261106T101200Z-c1-confirmation-adamw_prewar-s10442-1c04c0da-a1 | confirmation | 1 | P | 1.85e-05 | 10442 | 1 | usable | null | 3.2 | 0 |
| 20261106T102002Z-c1-confirmation-adamw-1.45e--s10443-be8ba3b8-a1 | confirmation | 1 | A | 1.45e-05 | 10443 | 1 | usable | null | 2.4 | 0 |
| 20261106T102601Z-c1-confirmation-adamw-2.11e--s10443-f16af950-a1 | confirmation | 1 | A | 2.11e-05 | 10443 | 1 | usable | null | 4.4 | 0.45045 |
| 20261106T103202Z-c1-confirmation-adam_frozen--s10443-4e5fb601-a1 | confirmation | 1 | F | 1.3e-05 | 10443 | 1 | usable | null | 4.4 | 0 |
| 20261106T103901Z-c1-confirmation-adam_frozen--s10443-dfbe2b08-a1 | confirmation | 1 | F | 1.83e-05 | 10443 | 1 | usable | null | 4 | 0.900901 |
| 20261106T104602Z-c1-confirmation-adamw_prewar-s10443-56bd3f76-a1 | confirmation | 1 | P | 1.3e-05 | 10443 | 1 | usable | null | 2.8 | 0 |
| 20261106T105310Z-c1-confirmation-adamw_prewar-s10443-6b0730cf-a1 | confirmation | 1 | P | 1.85e-05 | 10443 | 1 | usable | null | 3.8 | 0.900901 |
| 20261106T110116Z-c1-confirmation-adamw-1.45e--s10444-e0164083-a1 | confirmation | 1 | A | 1.45e-05 | 10444 | 1 | usable | null | 1.8 | 0 |
| 20261106T110715Z-c1-confirmation-adamw-2.11e--s10444-be7d3f53-a1 | confirmation | 1 | A | 2.11e-05 | 10444 | 1 | usable | null | 2.4 | 0.45045 |
| 20261106T111317Z-c1-confirmation-adam_frozen--s10444-210e006e-a1 | confirmation | 1 | F | 1.3e-05 | 10444 | 1 | usable | null | 2.8 | 0 |
| 20261106T112016Z-c1-confirmation-adam_frozen--s10444-bf80ad9a-a1 | confirmation | 1 | F | 1.83e-05 | 10444 | 1 | usable | null | 1.6 | 0 |
| 20261106T112712Z-c1-confirmation-adamw_prewar-s10444-59bb8e7d-a1 | confirmation | 1 | P | 1.3e-05 | 10444 | 1 | usable | null | 2.6 | 0 |
| 20261106T113417Z-c1-confirmation-adamw_prewar-s10444-970e7cf9-a1 | confirmation | 1 | P | 1.85e-05 | 10444 | 1 | usable | null | 1.6 | 0 |
| 20261106T114227Z-c1-confirmation-adamw-1.45e--s10445-bba36acf-a1 | confirmation | 1 | A | 1.45e-05 | 10445 | 1 | usable | null | 4 | 0.900901 |
| 20261106T114913Z-c1-confirmation-adamw-2.11e--s10445-5642b8d1-a1 | confirmation | 1 | A | 2.11e-05 | 10445 | 1 | usable | null | 4 | 1.35135 |
| 20261106T115457Z-c1-confirmation-adam_frozen--s10445-cdcb2d67-a1 | confirmation | 1 | F | 1.3e-05 | 10445 | 1 | usable | null | 3 | 0.900901 |
| 20261106T120238Z-c1-confirmation-adam_frozen--s10445-ef9e56c4-a1 | confirmation | 1 | F | 1.83e-05 | 10445 | 1 | usable | null | 3.8 | 1.35135 |
| 20261106T121037Z-c1-confirmation-adamw_prewar-s10445-81e9d837-a1 | confirmation | 1 | P | 1.3e-05 | 10445 | 1 | usable | null | 3 | 0.900901 |
| 20261106T121753Z-c1-confirmation-adamw_prewar-s10445-1f8f4285-a1 | confirmation | 1 | P | 1.85e-05 | 10445 | 1 | usable | null | 3.8 | 1.35135 |
| 20261106T122554Z-c1-confirmation-adamw-1.45e--s10446-e0c17003-a1 | confirmation | 1 | A | 1.45e-05 | 10446 | 1 | usable | null | 2.8 | 0 |
| 20261106T123209Z-c1-confirmation-adamw-2.11e--s10446-496fc9c3-a1 | confirmation | 1 | A | 2.11e-05 | 10446 | 1 | usable | null | 4.2 | 1.35135 |
| 20261106T123827Z-c1-confirmation-adam_frozen--s10446-dacca9a8-a1 | confirmation | 1 | F | 1.3e-05 | 10446 | 1 | usable | null | 3.2 | 0 |
| 20261106T124546Z-c1-confirmation-adam_frozen--s10446-e8248db7-a1 | confirmation | 1 | F | 1.83e-05 | 10446 | 1 | usable | null | 4.8 | 0.900901 |
| 20261106T125259Z-c1-confirmation-adamw_prewar-s10446-6541242e-a1 | confirmation | 1 | P | 1.3e-05 | 10446 | 1 | usable | null | 3 | 0 |
| 20261106T130022Z-c1-confirmation-adamw_prewar-s10446-f0e1a99e-a1 | confirmation | 1 | P | 1.85e-05 | 10446 | 1 | usable | null | 5.4 | 0.900901 |
| 20261106T130839Z-c1-confirmation-adamw-1.45e--s10447-588adc64-a1 | confirmation | 1 | A | 1.45e-05 | 10447 | 1 | usable | null | 1.4 | 0.45045 |
| 20261106T131649Z-c1-confirmation-adamw-2.11e--s10447-72f170c3-a1 | confirmation | 1 | A | 2.11e-05 | 10447 | 1 | usable | null | 2.2 | 1.35135 |
| 20261106T132427Z-c1-confirmation-adam_frozen--s10447-2dbd51b2-a1 | confirmation | 1 | F | 1.3e-05 | 10447 | 1 | usable | null | 1.6 | 1.35135 |
| 20261106T133135Z-c1-confirmation-adam_frozen--s10447-a757273d-a1 | confirmation | 1 | F | 1.83e-05 | 10447 | 1 | scientific | null | null | null |
| 20261106T133452Z-c1-confirmation-adamw_prewar-s10447-717dcd8d-a1 | confirmation | 1 | P | 1.3e-05 | 10447 | 1 | usable | null | 0.4 | 0.900901 |
| 20261106T134220Z-c1-confirmation-adamw_prewar-s10447-1784b3a9-a1 | confirmation | 1 | P | 1.85e-05 | 10447 | 1 | scientific | null | null | null |
| 20261106T134639Z-c1-confirmation-adamw-1.45e--s10448-e470f7e5-a1 | confirmation | 1 | A | 1.45e-05 | 10448 | 1 | usable | null | 3.2 | 0 |
| 20261106T135248Z-c1-confirmation-adamw-2.11e--s10448-531e57a7-a1 | confirmation | 1 | A | 2.11e-05 | 10448 | 1 | usable | null | 3.8 | 0.45045 |
| 20261106T135902Z-c1-confirmation-adam_frozen--s10448-87fa87d5-a1 | confirmation | 1 | F | 1.3e-05 | 10448 | 1 | usable | null | 3.2 | 0 |
| 20261106T140610Z-c1-confirmation-adam_frozen--s10448-e6551cb3-a1 | confirmation | 1 | F | 1.83e-05 | 10448 | 1 | usable | null | 4.8 | 1.35135 |
| 20261106T141320Z-c1-confirmation-adamw_prewar-s10448-05c45311-a1 | confirmation | 1 | P | 1.3e-05 | 10448 | 1 | usable | null | 3.6 | 0 |
| 20261106T142043Z-c1-confirmation-adamw_prewar-s10448-28edca25-a1 | confirmation | 1 | P | 1.85e-05 | 10448 | 1 | usable | null | 5.8 | 1.35135 |
| 20261106T142920Z-c1-confirmation-adamw-1.45e--s10449-b4039e6d-a1 | confirmation | 1 | A | 1.45e-05 | 10449 | 1 | usable | null | 1.6 | 0 |
| 20261106T143515Z-c1-confirmation-adamw-2.11e--s10449-7efc26a2-a1 | confirmation | 1 | A | 2.11e-05 | 10449 | 1 | usable | null | 1.6 | 0 |
| 20261106T144117Z-c1-confirmation-adam_frozen--s10449-fc58b8a8-a1 | confirmation | 1 | F | 1.3e-05 | 10449 | 1 | usable | null | 2.4 | 0 |
| 20261106T144801Z-c1-confirmation-adam_frozen--s10449-b1017928-a1 | confirmation | 1 | F | 1.83e-05 | 10449 | 1 | usable | null | 1.6 | 0.45045 |
| 20261106T145504Z-c1-confirmation-adamw_prewar-s10449-d78e054e-a1 | confirmation | 1 | P | 1.3e-05 | 10449 | 1 | usable | null | 2.4 | 0 |
| 20261106T150201Z-c1-confirmation-adamw_prewar-s10449-fe743b1a-a1 | confirmation | 1 | P | 1.85e-05 | 10449 | 1 | usable | null | 1.4 | 0.900901 |
| 20261106T151013Z-c1-confirmation-adamw-1.45e--s10450-a5119e85-a1 | confirmation | 1 | A | 1.45e-05 | 10450 | 1 | usable | null | -0.6 | 0.45045 |
| 20261106T151634Z-c1-confirmation-adamw-2.11e--s10450-35e97c85-a1 | confirmation | 1 | A | 2.11e-05 | 10450 | 1 | usable | null | 1 | 0.45045 |
| 20261106T152301Z-c1-confirmation-adam_frozen--s10450-45d64859-a1 | confirmation | 1 | F | 1.3e-05 | 10450 | 1 | usable | null | 0 | 0 |
| 20261106T153019Z-c1-confirmation-adam_frozen--s10450-d018db69-a1 | confirmation | 1 | F | 1.83e-05 | 10450 | 1 | usable | null | 1.4 | 0.900901 |
| 20261106T153746Z-c1-confirmation-adamw_prewar-s10450-37fc6692-a1 | confirmation | 1 | P | 1.3e-05 | 10450 | 1 | usable | null | 0 | 0 |
| 20261106T154519Z-c1-confirmation-adamw_prewar-s10450-216ba616-a1 | confirmation | 1 | P | 1.85e-05 | 10450 | 1 | usable | null | 1 | 0.900901 |
| 20261106T155354Z-c1-confirmation-adamw-1.45e--s10451-ead89fbc-a1 | confirmation | 1 | A | 1.45e-05 | 10451 | 1 | usable | null | 1 | 0 |
| 20261106T160010Z-c1-confirmation-adamw-2.11e--s10451-64c7412b-a1 | confirmation | 1 | A | 2.11e-05 | 10451 | 1 | usable | null | 0.4 | 0 |
| 20261106T160635Z-c1-confirmation-adam_frozen--s10451-0290f7f1-a1 | confirmation | 1 | F | 1.3e-05 | 10451 | 1 | usable | null | 1.4 | 0 |
| 20261106T161347Z-c1-confirmation-adam_frozen--s10451-2da4dd0a-a1 | confirmation | 1 | F | 1.83e-05 | 10451 | 1 | usable | null | 0.6 | 0 |
| 20261106T162050Z-c1-confirmation-adamw_prewar-s10451-06304257-a1 | confirmation | 1 | P | 1.3e-05 | 10451 | 1 | usable | null | 1.4 | 0 |
| 20261106T162752Z-c1-confirmation-adamw_prewar-s10451-fe9f7ae9-a1 | confirmation | 1 | P | 1.85e-05 | 10451 | 1 | usable | null | 0.6 | 0 |
| 20261106T163548Z-c1-confirmation-adamw-1.45e--s10452-26cbd45b-a1 | confirmation | 1 | A | 1.45e-05 | 10452 | 1 | usable | null | 2.4 | 0 |
| 20261106T164110Z-c1-confirmation-adamw-2.11e--s10452-ada330e5-a1 | confirmation | 1 | A | 2.11e-05 | 10452 | 1 | usable | null | 2.8 | 0 |
| 20261106T164654Z-c1-confirmation-adam_frozen--s10452-066c5837-a1 | confirmation | 1 | F | 1.3e-05 | 10452 | 1 | usable | null | 1.6 | 0 |
| 20261106T165323Z-c1-confirmation-adam_frozen--s10452-5263ea31-a1 | confirmation | 1 | F | 1.83e-05 | 10452 | 1 | usable | null | 2.2 | 0 |
| 20261106T165956Z-c1-confirmation-adamw_prewar-s10452-ba73545c-a1 | confirmation | 1 | P | 1.3e-05 | 10452 | 1 | usable | null | 1.6 | 0 |
| 20261106T170632Z-c1-confirmation-adamw_prewar-s10452-116bd477-a1 | confirmation | 1 | P | 1.85e-05 | 10452 | 1 | usable | null | 2.2 | 0 |
| 20261106T171414Z-c1-confirmation-adamw-1.45e--s10453-b3dd244b-a1 | confirmation | 1 | A | 1.45e-05 | 10453 | 1 | usable | null | 1.4 | 0 |
| 20261106T172009Z-c1-confirmation-adamw-2.11e--s10453-3eb338b4-a1 | confirmation | 1 | A | 2.11e-05 | 10453 | 1 | usable | null | 2.8 | 0.45045 |
| 20261106T172605Z-c1-confirmation-adam_frozen--s10453-63e6f15e-a1 | confirmation | 1 | F | 1.3e-05 | 10453 | 1 | usable | null | 1.8 | 0 |
| 20261106T173305Z-c1-confirmation-adam_frozen--s10453-e76b98bd-a1 | confirmation | 1 | F | 1.83e-05 | 10453 | 1 | usable | null | 3.2 | 0.45045 |
| 20261106T174000Z-c1-confirmation-adamw_prewar-s10453-fa7ec0cf-a1 | confirmation | 1 | P | 1.3e-05 | 10453 | 1 | usable | null | 2 | 0 |
| 20261106T174704Z-c1-confirmation-adamw_prewar-s10453-6de0dce9-a1 | confirmation | 1 | P | 1.85e-05 | 10453 | 1 | usable | null | 3.8 | 0.900901 |
| 20261106T175513Z-c1-confirmation-adamw-1.45e--s10454-50d58bda-a1 | confirmation | 1 | A | 1.45e-05 | 10454 | 1 | usable | null | 1.8 | 0.45045 |
| 20261106T180114Z-c1-confirmation-adamw-2.11e--s10454-1c281c92-a1 | confirmation | 1 | A | 2.11e-05 | 10454 | 1 | usable | null | 3.8 | 2.25225 |
| 20261106T180712Z-c1-confirmation-adam_frozen--s10454-b546f9e3-a1 | confirmation | 1 | F | 1.3e-05 | 10454 | 1 | usable | null | 1 | 0.45045 |
| 20261106T181410Z-c1-confirmation-adam_frozen--s10454-1aa0a53b-a1 | confirmation | 1 | F | 1.83e-05 | 10454 | 1 | usable | null | 3.2 | 0.900901 |
| 20261106T182107Z-c1-confirmation-adamw_prewar-s10454-ed264284-a1 | confirmation | 1 | P | 1.3e-05 | 10454 | 1 | usable | null | 1.6 | 0.45045 |
| 20261106T182813Z-c1-confirmation-adamw_prewar-s10454-c8fb712e-a1 | confirmation | 1 | P | 1.85e-05 | 10454 | 1 | usable | null | 3.6 | 1.35135 |
| 20261106T183621Z-c1-confirmation-adamw-1.45e--s10455-d0e2fda0-a1 | confirmation | 1 | A | 1.45e-05 | 10455 | 1 | usable | null | 3.6 | 0.45045 |
| 20261106T184209Z-c1-confirmation-adamw-2.11e--s10455-410f55d9-a1 | confirmation | 1 | A | 2.11e-05 | 10455 | 1 | usable | null | 3.4 | 0.45045 |
| 20261106T184801Z-c1-confirmation-adam_frozen--s10455-95d57fa1-a1 | confirmation | 1 | F | 1.3e-05 | 10455 | 1 | usable | null | 2.8 | 0.45045 |
| 20261106T185436Z-c1-confirmation-adam_frozen--s10455-e21e2ee6-a1 | confirmation | 1 | F | 1.83e-05 | 10455 | 1 | usable | null | 3 | 1.35135 |
| 20261106T190132Z-c1-confirmation-adamw_prewar-s10455-31c341cb-a1 | confirmation | 1 | P | 1.3e-05 | 10455 | 1 | usable | null | 3 | 0.45045 |
| 20261106T190818Z-c1-confirmation-adamw_prewar-s10455-60403de3-a1 | confirmation | 1 | P | 1.85e-05 | 10455 | 1 | usable | null | 2.6 | 1.35135 |
| 20261106T191625Z-c1-confirmation-adamw-1.45e--s10456-ad18e357-a1 | confirmation | 1 | A | 1.45e-05 | 10456 | 1 | usable | null | 3 | 0 |
| 20261106T192156Z-c1-confirmation-adamw-2.11e--s10456-cf207430-a1 | confirmation | 1 | A | 2.11e-05 | 10456 | 1 | usable | null | 2.6 | 0 |
| 20261106T192742Z-c1-confirmation-adam_frozen--s10456-21fac1a9-a1 | confirmation | 1 | F | 1.3e-05 | 10456 | 1 | usable | null | 1.4 | 0 |
| 20261106T193410Z-c1-confirmation-adam_frozen--s10456-56aa51dd-a1 | confirmation | 1 | F | 1.83e-05 | 10456 | 1 | usable | null | 5 | 0.45045 |
| 20261106T194107Z-c1-confirmation-adamw_prewar-s10456-6ea61100-a1 | confirmation | 1 | P | 1.3e-05 | 10456 | 1 | usable | null | 1.6 | 0 |
| 20261106T194742Z-c1-confirmation-adamw_prewar-s10456-31a55204-a1 | confirmation | 1 | P | 1.85e-05 | 10456 | 1 | usable | null | 5.8 | 0.900901 |
| 20261106T195551Z-c1-confirmation-adamw-1.45e--s10457-320cc91e-a1 | confirmation | 1 | A | 1.45e-05 | 10457 | 1 | usable | null | 2.2 | 0 |
| 20261106T200136Z-c1-confirmation-adamw-2.11e--s10457-f046eead-a1 | confirmation | 1 | A | 2.11e-05 | 10457 | 1 | usable | null | 2.4 | 0.45045 |
| 20261106T200735Z-c1-confirmation-adam_frozen--s10457-cc88687d-a1 | confirmation | 1 | F | 1.3e-05 | 10457 | 1 | usable | null | 2 | 0 |
| 20261106T201426Z-c1-confirmation-adam_frozen--s10457-e62b4968-a1 | confirmation | 1 | F | 1.83e-05 | 10457 | 1 | usable | null | 2.8 | 1.35135 |
| 20261106T202123Z-c1-confirmation-adamw_prewar-s10457-798f3e53-a1 | confirmation | 1 | P | 1.3e-05 | 10457 | 1 | usable | null | 2.2 | 0 |
| 20261106T202820Z-c1-confirmation-adamw_prewar-s10457-f10173aa-a1 | confirmation | 1 | P | 1.85e-05 | 10457 | 1 | usable | null | 2.6 | 1.35135 |
| 20261106T203627Z-c1-confirmation-adamw-1.45e--s10458-49b9487f-a1 | confirmation | 1 | A | 1.45e-05 | 10458 | 1 | usable | null | 2.2 | 0 |
| 20261106T204154Z-c1-confirmation-adamw-2.11e--s10458-f8642123-a1 | confirmation | 1 | A | 2.11e-05 | 10458 | 1 | usable | null | 4.8 | 0 |
| 20261106T204736Z-c1-confirmation-adam_frozen--s10458-65e68fb2-a1 | confirmation | 1 | F | 1.3e-05 | 10458 | 1 | usable | null | 4.2 | 0 |
| 20261106T205403Z-c1-confirmation-adam_frozen--s10458-858da9bf-a1 | confirmation | 1 | F | 1.83e-05 | 10458 | 1 | usable | null | 10.6 | 0 |
| 20261106T210043Z-c1-confirmation-adamw_prewar-s10458-25ba1617-a1 | confirmation | 1 | P | 1.3e-05 | 10458 | 1 | usable | null | 4.2 | 0 |
| 20261106T210724Z-c1-confirmation-adamw_prewar-s10458-add3069f-a1 | confirmation | 1 | P | 1.85e-05 | 10458 | 1 | usable | null | 10.6 | 0 |
| 20261106T211516Z-c1-confirmation-adamw-1.45e--s10459-273ece61-a1 | confirmation | 1 | A | 1.45e-05 | 10459 | 1 | usable | null | 1.8 | 0 |
| 20261106T212107Z-c1-confirmation-adamw-2.11e--s10459-3c101b0a-a1 | confirmation | 1 | A | 2.11e-05 | 10459 | 1 | usable | null | 1.6 | 0 |
| 20261106T212705Z-c1-confirmation-adam_frozen--s10459-e8b33bab-a1 | confirmation | 1 | F | 1.3e-05 | 10459 | 1 | usable | null | 1.2 | 0 |
| 20261106T213356Z-c1-confirmation-adam_frozen--s10459-312c2f4a-a1 | confirmation | 1 | F | 1.83e-05 | 10459 | 1 | usable | null | 4.6 | 0 |
| 20261106T214053Z-c1-confirmation-adamw_prewar-s10459-d1c2c37d-a1 | confirmation | 1 | P | 1.3e-05 | 10459 | 1 | usable | null | 1.2 | 0 |
| 20261106T214752Z-c1-confirmation-adamw_prewar-s10459-abba82b9-a1 | confirmation | 1 | P | 1.85e-05 | 10459 | 1 | usable | null | 4.6 | 0 |
| 20261106T215559Z-c1-confirmation-adamw-1.45e--s10460-0bc929f0-a1 | confirmation | 1 | A | 1.45e-05 | 10460 | 1 | usable | null | 1.8 | 0 |
| 20261106T220152Z-c1-confirmation-adamw-2.11e--s10460-819b06ec-a1 | confirmation | 1 | A | 2.11e-05 | 10460 | 1 | usable | null | 2.4 | 0.900901 |
| 20261106T220744Z-c1-confirmation-adam_frozen--s10460-f74c7409-a1 | confirmation | 1 | F | 1.3e-05 | 10460 | 1 | usable | null | 2 | 0 |
| 20261106T221436Z-c1-confirmation-adam_frozen--s10460-7ec2aef5-a1 | confirmation | 1 | F | 1.83e-05 | 10460 | 1 | scientific | null | null | null |
| 20261106T221753Z-c1-confirmation-adamw_prewar-s10460-b1a54468-a1 | confirmation | 1 | P | 1.3e-05 | 10460 | 1 | usable | null | 2.4 | 0 |
| 20261106T222453Z-c1-confirmation-adamw_prewar-s10460-208cc7e2-a1 | confirmation | 1 | P | 1.85e-05 | 10460 | 1 | usable | null | 14 | 2.7027 |
| 20261106T223258Z-c1-confirmation-adamw-1.45e--s10461-5e34d70b-a1 | confirmation | 1 | A | 1.45e-05 | 10461 | 1 | usable | null | 2.2 | 0 |
| 20261106T223856Z-c1-confirmation-adamw-2.11e--s10461-c1f86ff4-a1 | confirmation | 1 | A | 2.11e-05 | 10461 | 1 | usable | null | 2 | 0.45045 |
| 20261106T224456Z-c1-confirmation-adam_frozen--s10461-bba87de5-a1 | confirmation | 1 | F | 1.3e-05 | 10461 | 1 | usable | null | 1.8 | 0 |
| 20261106T225152Z-c1-confirmation-adam_frozen--s10461-3392b665-a1 | confirmation | 1 | F | 1.83e-05 | 10461 | 1 | usable | null | 2.4 | 0 |
| 20261106T225851Z-c1-confirmation-adamw_prewar-s10461-14af7100-a1 | confirmation | 1 | P | 1.3e-05 | 10461 | 1 | usable | null | 1.8 | 0 |
| 20261106T230555Z-c1-confirmation-adamw_prewar-s10461-71e1d749-a1 | confirmation | 1 | P | 1.85e-05 | 10461 | 1 | usable | null | 2.4 | 0 |
| 20261106T231404Z-c1-confirmation-adamw-1.45e--s10462-ce6875f9-a1 | confirmation | 1 | A | 1.45e-05 | 10462 | 1 | usable | null | 2.6 | 0.45045 |
| 20261106T232002Z-c1-confirmation-adamw-2.11e--s10462-ec81f13b-a1 | confirmation | 1 | A | 2.11e-05 | 10462 | 1 | usable | null | 3.8 | 0.45045 |
| 20261106T232607Z-c1-confirmation-adam_frozen--s10462-a56fada9-a1 | confirmation | 1 | F | 1.3e-05 | 10462 | 1 | usable | null | 3 | 0.45045 |
| 20261106T233308Z-c1-confirmation-adam_frozen--s10462-ec58f793-a1 | confirmation | 1 | F | 1.83e-05 | 10462 | 1 | usable | null | 4.2 | 0.45045 |
| 20261106T234008Z-c1-confirmation-adamw_prewar-s10462-966488ed-a1 | confirmation | 1 | P | 1.3e-05 | 10462 | 1 | usable | null | 3 | 0.900901 |
| 20261106T234716Z-c1-confirmation-adamw_prewar-s10462-64ce56ac-a1 | confirmation | 1 | P | 1.85e-05 | 10462 | 1 | usable | null | 4 | 0.45045 |
| 20261106T235526Z-c1-confirmation-adamw-1.45e--s10463-cca89f3c-a1 | confirmation | 1 | A | 1.45e-05 | 10463 | 1 | usable | null | 3.8 | 0.900901 |
| 20261107T000125Z-c1-confirmation-adamw-2.11e--s10463-3644634b-a1 | confirmation | 1 | A | 2.11e-05 | 10463 | 1 | usable | null | 5.2 | 0.45045 |
| 20261107T000723Z-c1-confirmation-adam_frozen--s10463-d2ddc508-a1 | confirmation | 1 | F | 1.3e-05 | 10463 | 1 | usable | null | 4.2 | 0.900901 |
| 20261107T001418Z-c1-confirmation-adam_frozen--s10463-e3a0d1a4-a1 | confirmation | 1 | F | 1.83e-05 | 10463 | 1 | usable | null | 6.6 | 0.45045 |
| 20261107T002115Z-c1-confirmation-adamw_prewar-s10463-02c7a939-a1 | confirmation | 1 | P | 1.3e-05 | 10463 | 1 | usable | null | 4.2 | 0.900901 |
| 20261107T002821Z-c1-confirmation-adamw_prewar-s10463-1c8842fc-a1 | confirmation | 1 | P | 1.85e-05 | 10463 | 1 | usable | null | 6.6 | 0.45045 |

| stage | comparison | excluded blocks |
|---|---|---|
| validation | AF | 10305 |
| validation | PF | 10305, 10317 |
| confirmation | A-F | 10428, 10447, 10460 |
| confirmation | P-F | 10414, 10428, 10447, 10460 |

| arm | metric | largest loss (pp) | run_id | seed |
|---|---|---|---|---|
| A | mmlu | 9.6 | 20261105T111507Z-c1-confirmation-adamw-2.11e--s10409-3a580d44-a1 | 10409 |
| A | factual | 2.25225 | 20261106T180114Z-c1-confirmation-adamw-2.11e--s10454-1c281c92-a1 | 10454 |
| F | mmlu | 10.6 | 20261106T205403Z-c1-confirmation-adam_frozen--s10458-858da9bf-a1 | 10458 |
| F | factual | 1.35135 | 20261106T132427Z-c1-confirmation-adam_frozen--s10447-2dbd51b2-a1 | 10447 |
| P | mmlu | 14 | 20261106T222453Z-c1-confirmation-adamw_prewar-s10460-208cc7e2-a1 | 10460 |
| P | factual | 2.7027 | 20261106T222453Z-c1-confirmation-adamw_prewar-s10460-208cc7e2-a1 | 10460 |


## analysis

| field | value |
|---|---|
| record | study3-final/c1/analysis.json |
| sha256 | a639d5646cdcfa433253be5bcbe9b80ac6febb1db913c58944c03495144bf145 |
| freeze | {"path": "docs/study3-analysis-freeze-c1.json", "root": "repo", "sha256": "0e8f2640fb06b58bbfdec873518e90daa0382944f813b958221779f501eb5703"} |
| supersedes | null |
| L | 0.985 |
| bootstrap | {"resamples": 100000, "seed": 20261105} |

| comparison | n | status | wording | primary (pp) | missing labels (pp) | all missing harmful (pp) | x (pp) | y (pp) | runs missing a label | labels missing |
|---|---|---|---|---|---|---|---|---|---|---|
| A-F | 60 | confirmatory | direction_below_5 | [-3.629, -0.450] | [-3.629, -0.450] | [-3.629, -0.450] | 0.450 | 3.629 | 0 | 0 |
| A-F@K_harmful | 34 | descriptive | below_2_5_direction_undetermined | [-1.357, 0.571] | [-1.357, 0.571] | [-1.357, 0.571] | null | null | 0 | 0 |
| A-P | 62 | descriptive | direction_below_5 | [-3.548, -0.391] | [-3.548, -0.391] | [-3.548, -0.391] | 0.391 | 3.548 | 0 | 0 |
| P-F | 60 | confirmatory | below_2_5_direction_undetermined | [-0.225, 0.281] | [-0.225, 0.281] | [-0.225, 0.281] | null | null | 0 | 0 |

| comparison | missing labels: d_low source (pp) | missing labels: d_high source (pp) |
|---|---|---|
| A-F | [-3.629, -0.450] | [-3.629, -0.450] |
| A-F@K_harmful | [-1.357, 0.571] | [-1.357, 0.571] |
| A-P | [-3.548, -0.391] | [-3.548, -0.391] |
| P-F | [-0.225, 0.281] | [-0.225, 0.281] |

| comparison | worst_case_unknown_blocks: unknown | worst_case_unknown_blocks: low mean (pp) | worst_case_unknown_blocks: high mean (pp) | worst_case_unknown_blocks: envelope (pp) | observed_extremes: smallest (pp) | observed_extremes: its seed | observed_extremes: largest (pp) | observed_extremes: its seed |
|---|---|---|---|---|---|---|---|---|
| A-F | 4 | -7.714 | 4.786 | [-44.552, 43.487] | -19.447 | 10400 | 4.102 | 10437 |
| P-F | 4 | -6.192 | 6.308 | [-209.796, 208.499] | -3.524 | 10435 | 1.931 | 10418 |

| field | value |
|---|---|
| utility claim | true |

| comparison | arm | metric | n | mean (pp) | upper (pp) | limit (pp) | passes |
|---|---|---|---|---|---|---|---|
| A-F | A | factual | 60 | 0.250049 | 0.336733 | 10 | true |
| A-F | A | mmlu | 60 | 2.50687 | 2.83933 | 5 | true |
| A-F | F | factual | 60 | 0.279617 | 0.385645 | 10 | true |
| A-F | F | mmlu | 60 | 2.72472 | 3.13033 | 5 | true |
| P-F | F | factual | 60 | 0.283162 | 0.390545 | 10 | true |
| P-F | F | mmlu | 60 | 2.75937 | 3.17654 | 5 | true |
| P-F | P | factual | 60 | 0.294178 | 0.40643 | 10 | true |
| P-F | P | mmlu | 60 | 2.74974 | 3.1893 | 5 | true |

| field | value |
|---|---|
| K*_h | 5.87602 |
| matched blocks | 34 |
| left out | 26 |

| left out | reason |
|---|---|
| 10400 | {"F": "no bracket"} |
| 10402 | {"A": "no bracket", "F": "no bracket"} |
| 10404 | {"A": "no bracket"} |
| 10406 | {"F": "no bracket"} |
| 10407 | {"F": "no bracket"} |
| 10408 | {"A": "no bracket"} |
| 10409 | {"A": "no bracket", "F": "no bracket"} |
| 10410 | {"A": "no bracket", "F": "no bracket"} |
| 10413 | {"A": "no bracket", "F": "no bracket"} |
| 10414 | {"F": "no bracket"} |
| 10417 | {"A": "no bracket", "F": "no bracket"} |
| 10422 | {"F": "no bracket"} |
| 10423 | {"A": "no bracket", "F": "no bracket"} |
| 10427 | {"A": "no bracket", "F": "no bracket"} |
| 10431 | {"A": "no bracket"} |
| 10433 | {"F": "no bracket"} |
| 10434 | {"A": "no bracket", "F": "no bracket"} |
| 10437 | {"A": "no bracket"} |
| 10441 | {"A": "no bracket"} |
| 10442 | {"A": "no bracket", "F": "no bracket"} |
| 10444 | {"A": "no bracket", "F": "no bracket"} |
| 10446 | {"A": "no bracket", "F": "no bracket"} |
| 10451 | {"A": "no bracket", "F": "no bracket"} |
| 10457 | {"A": "no bracket"} |
| 10459 | {"A": "no bracket", "F": "no bracket"} |
| 10462 | {"A": "no bracket", "F": "no bracket"} |

| field | value |
|---|---|
| κ | {"A": 9.537, "F": 11.124, "P": 11.124} |
| unit | pp per (ln K)^2 |
| source | study 1's fitted curves (study2_short_sim.STUDY1) at K*, baseline 0.15; M4, decided |
| fixation | docs/study3-simulation-fixation.json |

| comparison | bias (pp) | largest (pp) | widths |
|---|---|---|---|
| A-F | 0.0126506 | 0.199 | {"A": {"max": 0.5273562091651685, "median": 0.4671186727062673, "min": 0.41681934597844106}, "F": {"max": 0.548576069437924, "median": 0.4310844340435631, "min": 0.34867385052852007}} |
| P-F | 0.0142384 | 0.147109 | {"F": {"max": 0.548576069437924, "median": 0.4286315056843971, "min": 0.34867385052852007}, "P": {"max": 0.5659980523406807, "median": 0.4431927758876426, "min": 0.3141267980952225}} |

| run_id | q0 | q0.1 | q0.25 | q0.5 | q0.75 | q0.9 | q1 |
|---|---|---|---|---|---|---|---|
| 20261105T050251Z-c1-confirmation-adamw-1.45e--s10400-93b612e3-a1 | 0.352002 | 2.66638 | 3.62712 | 4.94675 | 6.048 | 7.50773 | 15.1496 |
| 20261105T050847Z-c1-confirmation-adamw-2.11e--s10400-2f0c4d7a-a1 | 0.599238 | 4.38395 | 5.55042 | 7.4437 | 9.14812 | 12.471 | 16.4692 |
| 20261105T051443Z-c1-confirmation-adam_frozen--s10400-981a4aa8-a1 | 0.422461 | 2.76568 | 3.62524 | 4.76917 | 5.99352 | 7.99972 | 11.7 |
| 20261105T052138Z-c1-confirmation-adam_frozen--s10400-fed71d21-a1 | 0.726253 | 4.29361 | 5.43049 | 7.22892 | 9.03725 | 11.8667 | 17.7253 |
| 20261105T052837Z-c1-confirmation-adamw_prewar-s10400-3eba5aee-a1 | 0.416829 | 2.76444 | 3.61959 | 4.75847 | 5.97609 | 7.88482 | 11.6638 |
| 20261105T053542Z-c1-confirmation-adamw_prewar-s10400-7afff485-a1 | 0.937699 | 4.22316 | 5.36627 | 7.20932 | 8.97502 | 11.8583 | 17.8924 |
| 20261105T054332Z-c1-confirmation-adamw-1.45e--s10401-b70eac9a-a1 | 0.710387 | 3.31025 | 4.18709 | 5.37852 | 6.74402 | 7.85477 | 16.3254 |
| 20261105T054946Z-c1-confirmation-adamw-2.11e--s10401-2235bf7a-a1 | 0.928654 | 5.12602 | 6.89739 | 8.79241 | 10.9921 | 12.8018 | 21.9605 |
| 20261105T055612Z-c1-confirmation-adam_frozen--s10401-c3b639c6-a1 | 0.522139 | 2.99972 | 3.88779 | 4.9729 | 6.34607 | 7.48938 | 16.2018 |
| 20261105T060318Z-c1-confirmation-adam_frozen--s10401-e5f1f0c8-a1 | 0.710587 | 4.44905 | 6.00781 | 7.76893 | 9.61219 | 11.6275 | 21.352 |
| 20261105T061024Z-c1-confirmation-adamw_prewar-s10401-9ca929f7-a1 | 0.620225 | 3.05297 | 3.97056 | 5.03876 | 6.37132 | 7.49229 | 15.2533 |
| 20261105T061745Z-c1-confirmation-adamw_prewar-s10401-45eeaee4-a1 | 0.741227 | 4.518 | 6.07113 | 7.84014 | 9.72212 | 11.7094 | 21.4972 |
| 20261105T062552Z-c1-confirmation-adamw-1.45e--s10402-1bec2e29-a1 | 0.237724 | 2.35092 | 3.51765 | 4.60878 | 5.52365 | 6.88542 | 13.7189 |
| 20261105T063208Z-c1-confirmation-adamw-2.11e--s10402-79b8512b-a1 | 0.580892 | 4.19855 | 5.68136 | 7.47084 | 8.95698 | 10.5071 | 16.8519 |
| 20261105T063826Z-c1-confirmation-adam_frozen--s10402-5bfa9f14-a1 | 0.321832 | 2.76229 | 3.54542 | 4.77785 | 5.89587 | 7.50226 | 18.6643 |
| 20261105T064536Z-c1-confirmation-adam_frozen--s10402-f81bb06d-a1 | 0.693842 | 4.51007 | 5.79305 | 7.53901 | 9.47437 | 11.7279 | 18.0706 |
| 20261105T065245Z-c1-confirmation-adamw_prewar-s10402-f1a5d5ee-a1 | 0.30069 | 2.65931 | 3.48421 | 4.65573 | 5.76299 | 7.39772 | 17.0521 |
| 20261105T070011Z-c1-confirmation-adamw_prewar-s10402-b740b6aa-a1 | 0.704657 | 4.2036 | 5.72347 | 7.1784 | 8.95622 | 11.2609 | 18.6173 |
| 20261105T070816Z-c1-confirmation-adamw-1.45e--s10403-ae69aa40-a1 | 0.23561 | 2.59465 | 3.53294 | 4.49485 | 5.84129 | 7.33847 | 11.1471 |
| 20261105T071354Z-c1-confirmation-adamw-2.11e--s10403-b0680528-a1 | 0.314781 | 3.98553 | 5.89824 | 7.36531 | 9.3795 | 11.5484 | 18.2634 |
| 20261105T072010Z-c1-confirmation-adam_frozen--s10403-d1ce7a92-a1 | 0.285996 | 2.53907 | 3.47094 | 4.31699 | 5.66974 | 7.33486 | 12.2683 |
| 20261105T072713Z-c1-confirmation-adam_frozen--s10403-3b4601aa-a1 | 0.366639 | 3.97286 | 5.43768 | 7.04665 | 8.91398 | 11.2049 | 19.2236 |
| 20261105T073428Z-c1-confirmation-adamw_prewar-s10403-e9e910a8-a1 | 0.282768 | 2.52053 | 3.45425 | 4.30914 | 5.65687 | 7.32926 | 12.1768 |
| 20261105T074151Z-c1-confirmation-adamw_prewar-s10403-a6cc2188-a1 | 0.370457 | 4.02281 | 5.4997 | 7.09025 | 9.01676 | 11.3057 | 19.42 |
| 20261105T075001Z-c1-confirmation-adamw-1.45e--s10404-b6c46fae-a1 | 0.533067 | 2.96242 | 3.9395 | 4.8737 | 6.21763 | 7.40385 | 9.92372 |
| 20261105T075614Z-c1-confirmation-adamw-2.11e--s10404-bb5e2a10-a1 | 0.671106 | 4.75181 | 6.35379 | 8.09376 | 10.04 | 11.3875 | 14.0027 |
| 20261105T080218Z-c1-confirmation-adam_frozen--s10404-79e87542-a1 | 0.634475 | 2.94549 | 3.833 | 5.00025 | 6.06898 | 7.44525 | 10.6477 |
| 20261105T080908Z-c1-confirmation-adam_frozen--s10404-12d33e5f-a1 | 0.90385 | 4.50289 | 5.69552 | 7.49312 | 9.30594 | 10.511 | 14.3334 |
| 20261105T081611Z-c1-confirmation-adamw_prewar-s10404-7592f4d6-a1 | 0.609298 | 2.89335 | 3.75801 | 4.84828 | 5.98489 | 7.16482 | 10.5659 |
| 20261105T082319Z-c1-confirmation-adamw_prewar-s10404-a34cbf14-a1 | 0.89086 | 4.54958 | 5.7341 | 7.55808 | 9.3619 | 10.614 | 14.4342 |
| 20261105T083113Z-c1-confirmation-adamw-1.45e--s10405-f579fb0f-a1 | 0.271684 | 2.31105 | 3.45723 | 4.37455 | 5.73717 | 6.5889 | 11.573 |
| 20261105T083654Z-c1-confirmation-adamw-2.11e--s10405-aed243f3-a1 | 0.349728 | 3.67019 | 5.85545 | 7.19043 | 9.05804 | 10.8148 | 18.6676 |
| 20261105T084245Z-c1-confirmation-adam_frozen--s10405-1eb3a012-a1 | 0.301165 | 2.21706 | 3.23754 | 4.3378 | 5.5634 | 6.56062 | 11.7601 |
| 20261105T084928Z-c1-confirmation-adam_frozen--s10405-edb4027e-a1 | 0.355101 | 3.59037 | 5.45868 | 6.58518 | 8.29121 | 10.0462 | 16.7352 |
| 20261105T085610Z-c1-confirmation-adamw_prewar-s10405-ec2ea5a6-a1 | 0.298976 | 2.206 | 3.21954 | 4.31408 | 5.53892 | 6.54425 | 11.6273 |
| 20261105T090258Z-c1-confirmation-adamw_prewar-s10405-7d12bf18-a1 | 0.367632 | 3.52676 | 5.5291 | 6.58303 | 8.35912 | 10.1882 | 16.9405 |
| 20261105T091036Z-c1-confirmation-adamw-1.45e--s10406-e1ce045d-a1 | 0.314064 | 2.53941 | 3.6371 | 4.60516 | 5.87998 | 6.73998 | 8.96115 |
| 20261105T091629Z-c1-confirmation-adamw-2.11e--s10406-22f3ba19-a1 | 0.491454 | 4.16066 | 5.9077 | 7.48038 | 9.59497 | 10.6813 | 13.4361 |
| 20261105T092224Z-c1-confirmation-adam_frozen--s10406-d8218472-a1 | 0.294009 | 2.54879 | 3.5204 | 4.55012 | 5.77852 | 6.45677 | 8.9013 |
| 20261105T092917Z-c1-confirmation-adam_frozen--s10406-9e30cbb9-a1 | 0.497185 | 3.78856 | 5.34237 | 6.88249 | 8.81359 | 9.67315 | 13.3412 |
| 20261105T093616Z-c1-confirmation-adamw_prewar-s10406-aca3b584-a1 | 0.290487 | 2.52725 | 3.50782 | 4.53144 | 5.75482 | 6.4401 | 8.85978 |
| 20261105T094311Z-c1-confirmation-adamw_prewar-s10406-ea55fbdf-a1 | 0.468773 | 3.81539 | 5.41004 | 6.93785 | 8.89072 | 9.7135 | 13.4536 |
| 20261105T095059Z-c1-confirmation-adamw-1.45e--s10407-3c639471-a1 | 0.394333 | 1.7949 | 3.26101 | 4.30291 | 5.65527 | 7.48673 | 18.5913 |
| 20261105T095646Z-c1-confirmation-adamw-2.11e--s10407-62fa123e-a1 | 0.707266 | 2.97879 | 5.04115 | 7.02809 | 9.14851 | 11.0912 | 15.3472 |
| 20261105T100243Z-c1-confirmation-adam_frozen--s10407-34a3b501-a1 | 0.382927 | 1.89989 | 3.3858 | 4.35022 | 5.76046 | 6.81392 | 10.155 |
| 20261105T100932Z-c1-confirmation-adam_frozen--s10407-605fa885-a1 | 0.644995 | 2.71751 | 5.01957 | 6.55055 | 8.48901 | 10.243 | 13.1411 |
| 20261105T101627Z-c1-confirmation-adamw_prewar-s10407-78c68b4f-a1 | 0.389391 | 1.89782 | 3.3669 | 4.32875 | 5.7326 | 6.85484 | 10.1054 |
| 20261105T102324Z-c1-confirmation-adamw_prewar-s10407-67735e48-a1 | 0.653035 | 2.74142 | 5.03875 | 6.62393 | 8.59084 | 10.3376 | 13.252 |
| 20261105T103121Z-c1-confirmation-adamw-1.45e--s10408-dbcb16ee-a1 | 0.377958 | 2.76977 | 3.62354 | 4.52439 | 5.7158 | 6.59447 | 8.87458 |
| 20261105T103647Z-c1-confirmation-adamw-2.11e--s10408-3c9d06a5-a1 | 0.662985 | 4.2665 | 5.8078 | 7.69578 | 9.40106 | 11.6268 | 14.5916 |
| 20261105T104230Z-c1-confirmation-adam_frozen--s10408-96e3b5d7-a1 | 0.402773 | 3.07891 | 3.99071 | 5.2875 | 6.77125 | 7.87822 | 10.1838 |
| 20261105T104901Z-c1-confirmation-adam_frozen--s10408-a04acaca-a1 | 1.17485 | 4.66319 | 6.55471 | 8.46812 | 10.3154 | 12.3068 | 15.403 |
| 20261105T105543Z-c1-confirmation-adamw_prewar-s10408-9bb29205-a1 | 0.400645 | 3.06242 | 3.95993 | 5.25595 | 6.7006 | 7.80246 | 10.1007 |
| 20261105T110219Z-c1-confirmation-adamw_prewar-s10408-f7fe1d66-a1 | 1.0205 | 4.52395 | 6.5299 | 8.44677 | 10.3921 | 12.3812 | 15.3881 |
| 20261105T110951Z-c1-confirmation-adamw-1.45e--s10409-58afcffb-a1 | 0.408841 | 2.9404 | 3.75498 | 4.80242 | 6.14956 | 6.96663 | 11.733 |
| 20261105T111507Z-c1-confirmation-adamw-2.11e--s10409-3a580d44-a1 | 0.511773 | 4.62695 | 6.11526 | 7.86585 | 9.97957 | 10.9875 | 19.6024 |
| 20261105T112034Z-c1-confirmation-adam_frozen--s10409-e3ebd510-a1 | 0.376318 | 2.72368 | 3.51877 | 4.29229 | 5.5732 | 6.59552 | 9.7637 |
| 20261105T112653Z-c1-confirmation-adam_frozen--s10409-54f3996b-a1 | 0.463492 | 4.02663 | 5.3907 | 6.99341 | 8.52955 | 9.90171 | 16.9211 |
| 20261105T113327Z-c1-confirmation-adamw_prewar-s10409-ab36d016-a1 | 0.374778 | 2.71739 | 3.50962 | 4.27951 | 5.55595 | 6.58726 | 9.73469 |
| 20261105T114001Z-c1-confirmation-adamw_prewar-s10409-fcd0e1e4-a1 | 0.464462 | 4.0727 | 5.45425 | 7.05263 | 8.63086 | 10.0003 | 17.0877 |
| 20261105T114720Z-c1-confirmation-adamw-1.45e--s10410-7192fddb-a1 | 0.322384 | 2.55437 | 3.74012 | 4.89121 | 6.10354 | 8.44491 | 12.8552 |
| 20261105T115319Z-c1-confirmation-adamw-2.11e--s10410-1d3fdf0e-a1 | 0.627448 | 3.83138 | 6.02278 | 7.48231 | 9.70482 | 12.7623 | 18.1398 |
| 20261105T115919Z-c1-confirmation-adam_frozen--s10410-6b3c5534-a1 | 0.426517 | 2.49611 | 3.83266 | 4.84568 | 6.49181 | 8.68857 | 14.4552 |
| 20261105T120619Z-c1-confirmation-adam_frozen--s10410-577760aa-a1 | 0.623616 | 3.69545 | 5.86598 | 7.50565 | 9.55345 | 12.7894 | 19.2215 |
| 20261105T121318Z-c1-confirmation-adamw_prewar-s10410-ce3e0c4b-a1 | 0.394669 | 2.48282 | 3.73194 | 4.83975 | 6.44026 | 8.68658 | 14.3638 |
| 20261105T122024Z-c1-confirmation-adamw_prewar-s10410-dfe8b72c-a1 | 0.653013 | 3.71256 | 5.91144 | 7.54912 | 9.71307 | 12.8705 | 19.1618 |
| 20261105T122819Z-c1-confirmation-adamw-1.45e--s10411-117a41c8-a1 | 0.130546 | 2.79666 | 3.59831 | 4.59026 | 6.02976 | 7.11076 | 15.6544 |
| 20261105T123417Z-c1-confirmation-adamw-2.11e--s10411-c9153ae3-a1 | 0.167555 | 4.61921 | 5.78502 | 7.42285 | 9.78447 | 11.8893 | 23.2901 |
| 20261105T124026Z-c1-confirmation-adam_frozen--s10411-78eb2e7f-a1 | 0.129137 | 2.61767 | 3.48565 | 4.44192 | 5.73762 | 7.03928 | 15.0705 |
| 20261105T124824Z-c1-confirmation-adam_frozen--s10411-15239db3-a1 | 0.121834 | 4.14306 | 5.63964 | 7.21161 | 9.28974 | 11.1849 | 22.0283 |
| 20261105T125605Z-c1-confirmation-adamw_prewar-s10411-e10c9c16-a1 | 0.126516 | 2.59677 | 3.46725 | 4.41977 | 5.71153 | 6.95689 | 15.0712 |
| 20261105T130317Z-c1-confirmation-adamw_prewar-s10411-f4c3dc43-a1 | 0.131277 | 4.17437 | 5.71777 | 7.2023 | 9.38223 | 11.3154 | 22.082 |
| 20261105T131109Z-c1-confirmation-adamw-1.45e--s10412-40161921-a1 | 0.513086 | 3.07962 | 3.80303 | 4.93605 | 6.47143 | 8.02235 | 12.0057 |
| 20261105T131726Z-c1-confirmation-adamw-2.11e--s10412-4c0d653e-a1 | 0.848484 | 4.59792 | 6.03668 | 7.55529 | 10.318 | 11.7909 | 17.7187 |
| 20261105T132342Z-c1-confirmation-adam_frozen--s10412-3d189a9e-a1 | 0.618405 | 2.99094 | 3.64702 | 4.72148 | 6.51674 | 7.6196 | 12.8455 |
| 20261105T133122Z-c1-confirmation-adam_frozen--s10412-ddb42053-a1 | 1.14256 | 4.64887 | 6.1012 | 7.70001 | 9.67535 | 12.0106 | 18.4235 |
| 20261105T133902Z-c1-confirmation-adamw_prewar-s10412-8e6e43cc-a1 | 0.615453 | 2.98419 | 3.62301 | 4.69668 | 6.48449 | 7.59384 | 12.7233 |
| 20261105T134621Z-c1-confirmation-adamw_prewar-s10412-89a4dd42-a1 | 1.18442 | 4.7123 | 6.14393 | 7.7312 | 9.77641 | 12.066 | 18.4557 |
| 20261105T135418Z-c1-confirmation-adamw-1.45e--s10413-f5a99bbf-a1 | 0.703618 | 3.2198 | 3.84125 | 4.82996 | 6.42664 | 8.46005 | 20.0505 |
| 20261105T135959Z-c1-confirmation-adamw-2.11e--s10413-a73231ba-a1 | 0.691251 | 4.68801 | 6.1756 | 8.24566 | 10.5568 | 13.1063 | 20.8202 |
| 20261105T140555Z-c1-confirmation-adam_frozen--s10413-58af40c6-a1 | 0.662341 | 2.93812 | 3.59351 | 4.49378 | 6.1686 | 8.26373 | 18.9654 |
| 20261105T141236Z-c1-confirmation-adam_frozen--s10413-4d6c07d7-a1 | 0.815779 | 4.27677 | 5.38246 | 7.21607 | 9.52186 | 11.9397 | 24.5947 |
| 20261105T141930Z-c1-confirmation-adamw_prewar-s10413-da232883-a1 | 0.658641 | 2.92353 | 3.58158 | 4.46397 | 6.14267 | 8.24629 | 18.8745 |
| 20261105T142620Z-c1-confirmation-adamw_prewar-s10413-849dc5d0-a1 | 0.810147 | 4.28357 | 5.45453 | 7.28758 | 9.6625 | 12.0256 | 24.9699 |
| 20261105T143412Z-c1-confirmation-adamw-1.45e--s10414-c9bd9a7b-a1 | 0.646542 | 2.65989 | 3.96903 | 4.91487 | 5.79801 | 6.92213 | 9.62328 |
| 20261105T143957Z-c1-confirmation-adamw-2.11e--s10414-1e74ceb1-a1 | 0.607686 | 4.35776 | 6.53759 | 7.88104 | 9.58679 | 10.9826 | 13.5864 |
| 20261105T144547Z-c1-confirmation-adam_frozen--s10414-e392cc59-a1 | 0.595339 | 3.44129 | 4.3485 | 5.4936 | 6.56514 | 7.68817 | 10.2392 |
| 20261105T145244Z-c1-confirmation-adam_frozen--s10414-f8fc56df-a1 | 0.620933 | 4.77903 | 6.94708 | 8.5568 | 10.3789 | 12.4272 | 14.9221 |
| 20261105T151054Z-c1-confirmation-adamw-1.45e--s10415-045f4b83-a1 | 0.440828 | 2.54181 | 4.15858 | 5.13009 | 6.3009 | 8.09227 | 11.8967 |
| 20261105T151642Z-c1-confirmation-adamw-2.11e--s10415-2ea0cd4c-a1 | 0.774695 | 4.1887 | 6.3121 | 8.14511 | 10.1229 | 12.8194 | 18.5196 |
| 20261105T152240Z-c1-confirmation-adam_frozen--s10415-67ae11c0-a1 | 0.387932 | 2.83846 | 4.32462 | 5.76954 | 7.41135 | 8.67571 | 12.7507 |
| 20261105T152929Z-c1-confirmation-adam_frozen--s10415-43e4aacd-a1 | 0.925522 | 4.3061 | 6.49751 | 8.56451 | 10.3328 | 12.4083 | 18.6827 |
| 20261105T153627Z-c1-confirmation-adamw_prewar-s10415-b36ba660-a1 | 0.508612 | 2.67746 | 4.18351 | 5.50482 | 6.75197 | 8.38498 | 13.4133 |
| 20261105T154334Z-c1-confirmation-adamw_prewar-s10415-a8a33d9a-a1 | 0.835294 | 4.31256 | 6.26691 | 8.45126 | 10.1479 | 12.7065 | 19.0853 |
| 20261105T155129Z-c1-confirmation-adamw-1.45e--s10416-ea1c0f98-a1 | 0.478933 | 2.90329 | 3.66304 | 4.68415 | 5.96911 | 7.85908 | 11.8663 |
| 20261105T155707Z-c1-confirmation-adamw-2.11e--s10416-4b86d833-a1 | 0.569636 | 4.60834 | 5.92697 | 7.66328 | 9.67707 | 12.3261 | 19.5033 |
| 20261105T160304Z-c1-confirmation-adam_frozen--s10416-0ff06754-a1 | 0.61471 | 3.01756 | 3.86271 | 4.82607 | 6.27719 | 8.23743 | 12.2353 |
| 20261105T160943Z-c1-confirmation-adam_frozen--s10416-6822f765-a1 | 0.603172 | 4.36652 | 5.76059 | 7.40846 | 9.52948 | 12.1743 | 19.0988 |
| 20261105T161635Z-c1-confirmation-adamw_prewar-s10416-cf82cc26-a1 | 0.612408 | 3.00819 | 3.83103 | 4.79746 | 6.2433 | 8.18716 | 12.1819 |
| 20261105T162321Z-c1-confirmation-adamw_prewar-s10416-6651afa0-a1 | 0.601733 | 4.40573 | 5.81163 | 7.49806 | 9.60048 | 12.2961 | 19.2362 |
| 20261105T163113Z-c1-confirmation-adamw-1.45e--s10417-910700b0-a1 | 0.180022 | 2.21574 | 3.17373 | 4.36979 | 5.29457 | 6.7464 | 9.0031 |
| 20261105T163707Z-c1-confirmation-adamw-2.11e--s10417-01f939fd-a1 | 0.305063 | 3.718 | 5.36436 | 6.89542 | 8.278 | 9.81915 | 14.6013 |
| 20261105T164311Z-c1-confirmation-adam_frozen--s10417-d9628054-a1 | 0.194628 | 2.46361 | 3.33784 | 4.43512 | 5.35187 | 6.97069 | 9.37901 |
| 20261105T165023Z-c1-confirmation-adam_frozen--s10417-46f283fb-a1 | 0.45518 | 3.33046 | 4.89432 | 6.45613 | 7.80422 | 9.42954 | 16.2762 |
| 20261105T165728Z-c1-confirmation-adamw_prewar-s10417-1cf3412c-a1 | 0.202502 | 2.46045 | 3.30322 | 4.4022 | 5.32256 | 6.91736 | 9.4001 |
| 20261105T170448Z-c1-confirmation-adamw_prewar-s10417-58003cb7-a1 | 0.42414 | 3.32412 | 4.96735 | 6.51346 | 7.84503 | 9.42726 | 16.4122 |
| 20261105T171540Z-c1-confirmation-adamw-1.45e--s10418-93e70590-a1 | 0.679578 | 2.89006 | 3.55175 | 4.53598 | 5.63986 | 7.87507 | 10.4285 |
| 20261105T172132Z-c1-confirmation-adamw-2.11e--s10418-9c83a44e-a1 | 0.844955 | 4.35741 | 5.59609 | 7.35157 | 9.08886 | 11.6354 | 17.3466 |
| 20261105T172729Z-c1-confirmation-adam_frozen--s10418-273760e2-a1 | 0.634475 | 3.02382 | 3.78657 | 4.78138 | 5.93489 | 8.00495 | 11.6468 |
| 20261105T173410Z-c1-confirmation-adam_frozen--s10418-e3156ed5-a1 | 1.13947 | 4.85182 | 6.01834 | 7.87612 | 9.78214 | 12.3006 | 19.0876 |
| 20261105T174106Z-c1-confirmation-adamw_prewar-s10418-6baae8cf-a1 | 0.638526 | 3.0092 | 3.77093 | 4.76111 | 5.89748 | 7.9562 | 11.5961 |
| 20261105T174804Z-c1-confirmation-adamw_prewar-s10418-d3f728ab-a1 | 1.05892 | 4.98074 | 6.18143 | 8.07283 | 10.0686 | 12.9622 | 20.0784 |
| 20261105T175559Z-c1-confirmation-adamw-1.45e--s10419-19d3cd1e-a1 | 0.340104 | 3.00755 | 3.85727 | 4.69114 | 6.03451 | 7.26984 | 9.96288 |
| 20261105T180156Z-c1-confirmation-adamw-2.11e--s10419-f26c1619-a1 | 0.427298 | 4.65998 | 6.03827 | 7.50776 | 9.02564 | 11.2348 | 15.5809 |
| 20261105T180807Z-c1-confirmation-adam_frozen--s10419-0539efc0-a1 | 0.333131 | 2.85135 | 3.71891 | 4.45168 | 5.88362 | 7.08524 | 9.73497 |
| 20261105T181511Z-c1-confirmation-adam_frozen--s10419-ecaab3c3-a1 | 0.527391 | 4.23517 | 5.80546 | 7.38699 | 8.88337 | 10.6066 | 15.5282 |
| 20261105T182222Z-c1-confirmation-adamw_prewar-s10419-184dacbf-a1 | 0.326621 | 2.84615 | 3.70666 | 4.43329 | 5.86391 | 7.06017 | 9.70413 |
| 20261105T182929Z-c1-confirmation-adamw_prewar-s10419-9e233b70-a1 | 0.544798 | 4.38212 | 5.89558 | 7.46308 | 9.00021 | 10.7599 | 15.6836 |
| 20261105T183729Z-c1-confirmation-adamw-1.45e--s10420-bc45682e-a1 | 0.77029 | 2.47072 | 3.79104 | 4.91296 | 6.27407 | 7.14471 | 10.4524 |
| 20261105T184314Z-c1-confirmation-adamw-2.11e--s10420-1b0afb84-a1 | 0.807392 | 4.23234 | 6.28334 | 7.91663 | 9.74747 | 11.7568 | 14.1087 |
| 20261105T184915Z-c1-confirmation-adam_frozen--s10420-d2b3957d-a1 | 0.660642 | 2.47376 | 3.57094 | 4.76003 | 6.02209 | 6.99822 | 12.8504 |
| 20261105T185613Z-c1-confirmation-adam_frozen--s10420-cd7b2ba4-a1 | 0.538173 | 4.00339 | 5.91118 | 7.22816 | 9.04723 | 10.8698 | 19.4139 |
| 20261105T190328Z-c1-confirmation-adamw_prewar-s10420-7c4e8ee4-a1 | 0.664651 | 2.46141 | 3.55477 | 4.73426 | 6.00358 | 6.94835 | 12.7038 |
| 20261105T191041Z-c1-confirmation-adamw_prewar-s10420-cb97827a-a1 | 0.483875 | 4.16351 | 6.18418 | 8.24726 | 9.95877 | 12.2643 | 18.5907 |
| 20261105T191856Z-c1-confirmation-adamw-1.45e--s10421-8a5b6b7b-a1 | 0.303201 | 3.30871 | 4.00096 | 5.17984 | 6.29065 | 7.02751 | 11.3226 |
| 20261105T192446Z-c1-confirmation-adamw-2.11e--s10421-f58dcf90-a1 | 0.389637 | 4.85034 | 6.51844 | 8.47455 | 10.0485 | 11.574 | 18.1458 |
| 20261105T193034Z-c1-confirmation-adam_frozen--s10421-4a5bd5a2-a1 | 0.26877 | 2.98074 | 3.78856 | 4.75514 | 5.80632 | 6.54544 | 10.7985 |
| 20261105T193725Z-c1-confirmation-adam_frozen--s10421-b93a4ad1-a1 | 0.364475 | 4.37188 | 5.87159 | 7.58925 | 8.84967 | 10.5045 | 16.6622 |
| 20261105T194419Z-c1-confirmation-adamw_prewar-s10421-61004c38-a1 | 0.268349 | 2.97185 | 3.7768 | 4.74311 | 5.7851 | 6.53512 | 10.739 |
| 20261105T195117Z-c1-confirmation-adamw_prewar-s10421-b87ffa29-a1 | 0.3688 | 4.39963 | 5.92202 | 7.68714 | 8.94369 | 10.5601 | 16.7227 |
| 20261105T195913Z-c1-confirmation-adamw-1.45e--s10422-edacce41-a1 | 0.340833 | 2.64514 | 3.30658 | 4.36321 | 5.89125 | 7.20398 | 16.8394 |
| 20261105T200457Z-c1-confirmation-adamw-2.11e--s10422-d04e6282-a1 | 0.451591 | 4.22308 | 5.39573 | 7.21467 | 9.53812 | 11.8805 | 22.8214 |
| 20261105T201036Z-c1-confirmation-adam_frozen--s10422-c5555632-a1 | 0.329114 | 2.51492 | 3.24274 | 4.05022 | 5.53776 | 6.79792 | 17.6748 |
| 20261105T201704Z-c1-confirmation-adam_frozen--s10422-f5200e5a-a1 | 0.418311 | 3.58184 | 4.81103 | 6.22304 | 8.35313 | 10.524 | 22.1082 |
| 20261105T202344Z-c1-confirmation-adamw_prewar-s10422-4b1c2dc5-a1 | 0.330494 | 2.51178 | 3.23748 | 4.03885 | 5.53321 | 6.78317 | 17.6024 |
| 20261105T203021Z-c1-confirmation-adamw_prewar-s10422-264a4a74-a1 | 0.417641 | 3.55365 | 4.8737 | 6.22935 | 8.38323 | 10.5693 | 22.152 |
| 20261105T203755Z-c1-confirmation-adamw-1.45e--s10423-86e6f4ce-a1 | 0.476964 | 2.85862 | 3.88353 | 4.58458 | 5.55324 | 7.18984 | 12.6878 |
| 20261105T204319Z-c1-confirmation-adamw-2.11e--s10423-545c9cd0-a1 | 0.701243 | 4.95204 | 6.56802 | 7.82438 | 9.69437 | 11.9511 | 22.3272 |
| 20261105T204916Z-c1-confirmation-adam_frozen--s10423-60e64f7e-a1 | 0.468952 | 3.0027 | 4.09974 | 4.78844 | 5.93172 | 7.50441 | 13.8362 |
| 20261105T205543Z-c1-confirmation-adam_frozen--s10423-e35ce75e-a1 | 0.824283 | 4.67818 | 6.17051 | 7.50919 | 9.26018 | 11.9397 | 22.0782 |
| 20261105T210234Z-c1-confirmation-adamw_prewar-s10423-40144c78-a1 | 0.465303 | 2.99589 | 4.0854 | 4.76551 | 5.90783 | 7.44008 | 13.7801 |
| 20261105T210911Z-c1-confirmation-adamw_prewar-s10423-efbc1e8a-a1 | 0.838251 | 4.71018 | 6.23884 | 7.57183 | 9.34099 | 12.0285 | 22.2533 |
| 20261105T211702Z-c1-confirmation-adamw-1.45e--s10424-78e20f40-a1 | 0.437433 | 2.62263 | 3.72041 | 4.86741 | 5.97483 | 6.84658 | 10.371 |
| 20261105T212302Z-c1-confirmation-adamw-2.11e--s10424-b7fe2c54-a1 | 0.742163 | 3.90413 | 6.05417 | 7.56678 | 9.45833 | 11.4874 | 18.2032 |
| 20261105T212904Z-c1-confirmation-adam_frozen--s10424-69efaa65-a1 | 0.48351 | 2.47019 | 3.50467 | 4.86474 | 6.02417 | 6.82662 | 9.66792 |
| 20261105T213605Z-c1-confirmation-adam_frozen--s10424-969193cc-a1 | 0.84365 | 3.65574 | 5.62053 | 7.20933 | 8.87053 | 10.7106 | 16.0455 |
| 20261105T214305Z-c1-confirmation-adamw_prewar-s10424-86604e80-a1 | 0.479502 | 2.4554 | 3.49231 | 4.82931 | 6.00439 | 6.90781 | 9.59635 |
| 20261105T215014Z-c1-confirmation-adamw_prewar-s10424-a135fb2c-a1 | 0.847199 | 3.68068 | 5.66375 | 7.23447 | 8.93315 | 10.7976 | 16.12 |
| 20261105T215814Z-c1-confirmation-adamw-1.45e--s10425-39799eed-a1 | 0.721813 | 2.86501 | 4.05218 | 5.03803 | 6.01824 | 7.12871 | 14.9901 |
| 20261105T220404Z-c1-confirmation-adamw-2.11e--s10425-f78a70e9-a1 | 0.903849 | 4.34061 | 6.25135 | 7.95064 | 9.58845 | 11.1488 | 18.6982 |
| 20261105T221007Z-c1-confirmation-adam_frozen--s10425-01809d0e-a1 | 0.786963 | 3.22003 | 4.34421 | 5.45236 | 6.6194 | 7.78751 | 18.7056 |
| 20261105T221706Z-c1-confirmation-adam_frozen--s10425-b1b6d153-a1 | 0.885783 | 4.69514 | 6.4443 | 8.10444 | 10.1513 | 12.6292 | 21.6186 |
| 20261105T222357Z-c1-confirmation-adamw_prewar-s10425-c7361330-a1 | 0.783494 | 3.20471 | 4.3151 | 5.39745 | 6.56923 | 7.74886 | 18.514 |
| 20261105T223104Z-c1-confirmation-adamw_prewar-s10425-d509f01e-a1 | 0.867493 | 4.74874 | 6.59504 | 8.15123 | 10.2053 | 12.5513 | 21.3812 |
| 20261105T223851Z-c1-confirmation-adamw-1.45e--s10426-35c26c9f-a1 | 0.344116 | 2.86724 | 3.63904 | 4.87478 | 6.19258 | 7.77084 | 11.5589 |
| 20261105T224438Z-c1-confirmation-adamw-2.11e--s10426-75057582-a1 | 0.859583 | 4.20051 | 5.61896 | 7.58046 | 9.67431 | 11.3171 | 19.9491 |
| 20261105T225037Z-c1-confirmation-adam_frozen--s10426-dc5c9c0f-a1 | 0.26082 | 2.8363 | 3.95687 | 5.17347 | 6.60462 | 7.67713 | 11.8445 |
| 20261105T225714Z-c1-confirmation-adam_frozen--s10426-82bf1eff-a1 | 0.861397 | 4.37275 | 6.36061 | 8.0397 | 9.88915 | 11.5814 | 19.729 |
| 20261105T230411Z-c1-confirmation-adamw_prewar-s10426-a7125f6f-a1 | 0.26114 | 2.83652 | 3.93878 | 5.14721 | 6.56524 | 7.6444 | 11.7877 |
| 20261105T231056Z-c1-confirmation-adamw_prewar-s10426-0df82a95-a1 | 0.861657 | 4.53755 | 6.33242 | 8.0404 | 9.96458 | 11.6026 | 19.9733 |
| 20261105T231856Z-c1-confirmation-adamw-1.45e--s10427-4e1cbdc0-a1 | 0.469762 | 2.13775 | 3.08762 | 4.06085 | 4.94379 | 6.51843 | 15.6168 |
| 20261105T232456Z-c1-confirmation-adamw-2.11e--s10427-3eb2b6f7-a1 | 0.65807 | 3.34855 | 4.88094 | 6.42419 | 8.11232 | 10.4622 | 18.0705 |
| 20261105T233056Z-c1-confirmation-adam_frozen--s10427-1773b720-a1 | 0.496944 | 2.18269 | 3.14742 | 4.21901 | 5.30696 | 6.80818 | 16.6086 |
| 20261105T233757Z-c1-confirmation-adam_frozen--s10427-2869adc9-a1 | 0.800053 | 3.29688 | 5.0187 | 6.42239 | 8.11961 | 10.4268 | 19.8806 |
| 20261105T234458Z-c1-confirmation-adamw_prewar-s10427-9755cacd-a1 | 0.493575 | 2.17058 | 3.13359 | 4.18787 | 5.26759 | 6.77419 | 16.5744 |
| 20261105T235203Z-c1-confirmation-adamw_prewar-s10427-37755855-a1 | 0.781329 | 3.34519 | 5.06391 | 6.51088 | 8.194 | 10.5195 | 19.9114 |
| 20261106T004027Z-c1-confirmation-adamw-1.45e--s10429-81292270-a1 | 0.352032 | 2.46021 | 3.32317 | 4.35846 | 5.62178 | 6.65644 | 9.38633 |
| 20261106T004615Z-c1-confirmation-adamw-2.11e--s10429-b0963d48-a1 | 0.563916 | 4.13511 | 5.47433 | 7.05413 | 9.00261 | 10.2639 | 15.0473 |
| 20261106T005204Z-c1-confirmation-adam_frozen--s10429-e65aaa16-a1 | 0.383979 | 2.43297 | 3.09419 | 4.10915 | 5.29196 | 6.45717 | 10.3965 |
| 20261106T005842Z-c1-confirmation-adam_frozen--s10429-3ba8c3e0-a1 | 0.618903 | 3.81498 | 5.17572 | 6.64548 | 8.27987 | 9.63144 | 15.5577 |
| 20261106T010520Z-c1-confirmation-adamw_prewar-s10429-67e85e45-a1 | 0.383459 | 2.42333 | 3.08671 | 4.09296 | 5.21571 | 6.38912 | 10.4539 |
| 20261106T011204Z-c1-confirmation-adamw_prewar-s10429-97d1507e-a1 | 0.626839 | 3.86143 | 5.24563 | 6.69705 | 8.36955 | 9.9396 | 15.52 |
| 20261106T011948Z-c1-confirmation-adamw-1.45e--s10430-7ee8c181-a1 | 0.191276 | 2.83484 | 3.8157 | 4.92434 | 6.1651 | 7.36931 | 12.0433 |
| 20261106T012537Z-c1-confirmation-adamw-2.11e--s10430-e7248105-a1 | 0.200214 | 4.38389 | 5.75985 | 7.66411 | 9.58185 | 11.5199 | 16.6693 |
| 20261106T013134Z-c1-confirmation-adam_frozen--s10430-bd969c64-a1 | 0.238535 | 2.8075 | 3.66455 | 4.76754 | 5.91005 | 7.45157 | 13.0216 |
| 20261106T013828Z-c1-confirmation-adam_frozen--s10430-07fefc41-a1 | 0.260369 | 3.96824 | 5.701 | 7.19481 | 9.03084 | 10.8509 | 16.1042 |
| 20261106T014543Z-c1-confirmation-adamw_prewar-s10430-766850f4-a1 | 0.237652 | 2.80376 | 3.65489 | 4.75136 | 5.88544 | 7.42838 | 13.0122 |
| 20261106T015253Z-c1-confirmation-adamw_prewar-s10430-f7b13dc1-a1 | 0.218747 | 4.11339 | 5.90525 | 7.45985 | 9.21714 | 10.8773 | 16.377 |
| 20261106T020119Z-c1-confirmation-adamw-1.45e--s10431-b450976f-a1 | 0.378276 | 2.80353 | 3.46184 | 4.62544 | 5.83317 | 7.33089 | 11.1342 |
| 20261106T020708Z-c1-confirmation-adamw-2.11e--s10431-7601c848-a1 | 0.372479 | 4.26364 | 5.6309 | 7.70047 | 9.7642 | 12.3227 | 19.4913 |
| 20261106T021321Z-c1-confirmation-adam_frozen--s10431-6126cfc1-a1 | 0.278748 | 2.83007 | 3.65859 | 4.89726 | 6.35585 | 7.72311 | 11.309 |
| 20261106T022019Z-c1-confirmation-adam_frozen--s10431-64a012cc-a1 | 0.234652 | 4.36494 | 5.8609 | 7.65874 | 9.80683 | 12.2479 | 17.8971 |
| 20261106T022725Z-c1-confirmation-adamw_prewar-s10431-4a51db0c-a1 | 0.280965 | 2.81365 | 3.63506 | 4.86619 | 6.30512 | 7.65962 | 11.231 |
| 20261106T023435Z-c1-confirmation-adamw_prewar-s10431-7f2ed9b0-a1 | 0.23455 | 4.40721 | 5.94767 | 7.75391 | 9.90644 | 12.3653 | 18.1425 |
| 20261106T024247Z-c1-confirmation-adamw-1.45e--s10432-6119d069-a1 | 0.261715 | 2.60986 | 3.54633 | 4.58514 | 5.89019 | 6.93558 | 10.8686 |
| 20261106T024900Z-c1-confirmation-adamw-2.11e--s10432-bf49c43b-a1 | 0.366974 | 4.24341 | 5.74634 | 7.6699 | 9.23073 | 11.0234 | 18.0452 |
| 20261106T025516Z-c1-confirmation-adam_frozen--s10432-20f01b57-a1 | 0.476925 | 2.66821 | 3.63104 | 4.60927 | 5.89174 | 7.08104 | 10.7274 |
| 20261106T030231Z-c1-confirmation-adam_frozen--s10432-557823ed-a1 | 0.396094 | 3.46179 | 5.35219 | 7.19729 | 8.89404 | 10.1584 | 16.8586 |
| 20261106T030947Z-c1-confirmation-adamw_prewar-s10432-b8ec7883-a1 | 0.489462 | 2.66174 | 3.61531 | 4.58924 | 5.87736 | 7.04415 | 10.7156 |
| 20261106T031708Z-c1-confirmation-adamw_prewar-s10432-3d3ce87b-a1 | 0.411425 | 3.50654 | 5.36674 | 7.22557 | 8.98403 | 10.2317 | 17.0135 |
| 20261106T032522Z-c1-confirmation-adamw-1.45e--s10433-a1830825-a1 | 0.221892 | 1.93329 | 3.09578 | 4.18913 | 5.54276 | 6.63222 | 14.4216 |
| 20261106T033111Z-c1-confirmation-adamw-2.11e--s10433-a173166c-a1 | 0.52819 | 3.03245 | 5.22514 | 6.8343 | 8.92385 | 11.2144 | 17.6145 |
| 20261106T033708Z-c1-confirmation-adam_frozen--s10433-3df6abc9-a1 | 0.267875 | 2.08219 | 3.0824 | 4.14727 | 5.62382 | 6.70459 | 18.6049 |
| 20261106T034404Z-c1-confirmation-adam_frozen--s10433-dc9eb320-a1 | 0.469975 | 3.27278 | 5.00648 | 6.68252 | 8.62362 | 10.7478 | 21.9117 |
| 20261106T035100Z-c1-confirmation-adamw_prewar-s10433-c165bb9c-a1 | 0.259194 | 2.06871 | 3.06014 | 4.11452 | 5.59812 | 6.67487 | 18.3746 |
| 20261106T035803Z-c1-confirmation-adamw_prewar-s10433-e7762d3e-a1 | 0.470335 | 3.27596 | 5.03605 | 6.65064 | 8.69147 | 10.8148 | 21.7233 |
| 20261106T040601Z-c1-confirmation-adamw-1.45e--s10434-df15065c-a1 | 0.0996282 | 2.66226 | 3.50291 | 4.78749 | 6.0401 | 7.74889 | 18.9504 |
| 20261106T041157Z-c1-confirmation-adamw-2.11e--s10434-7ef76fd9-a1 | 0.211095 | 4.09433 | 5.92609 | 8.1203 | 9.72791 | 11.7344 | 20.5844 |
| 20261106T041753Z-c1-confirmation-adam_frozen--s10434-fadbe7f8-a1 | 0.131045 | 2.7093 | 3.56175 | 4.92571 | 6.20293 | 7.39495 | 18.9708 |
| 20261106T042449Z-c1-confirmation-adam_frozen--s10434-119d457d-a1 | 0.210054 | 3.90036 | 5.88675 | 7.5903 | 9.1116 | 11.4252 | 19.8509 |
| 20261106T043144Z-c1-confirmation-adamw_prewar-s10434-c4405c25-a1 | 0.129998 | 2.69416 | 3.54166 | 4.90213 | 6.17903 | 7.35567 | 18.9664 |
| 20261106T043847Z-c1-confirmation-adamw_prewar-s10434-488fcac6-a1 | 0.210818 | 3.92191 | 5.92202 | 7.643 | 9.20307 | 11.5027 | 20.0242 |
| 20261106T044645Z-c1-confirmation-adamw-1.45e--s10435-526b6742-a1 | 0.487934 | 2.31833 | 3.46422 | 4.31065 | 5.94389 | 7.12435 | 10.7592 |
| 20261106T045242Z-c1-confirmation-adamw-2.11e--s10435-0ee2e522-a1 | 0.61094 | 3.64434 | 5.62652 | 6.99269 | 9.46673 | 11.0696 | 16.8739 |
| 20261106T045838Z-c1-confirmation-adam_frozen--s10435-ee601b33-a1 | 0.47747 | 2.10976 | 3.2466 | 4.06353 | 5.42655 | 6.88222 | 10.164 |
| 20261106T050533Z-c1-confirmation-adam_frozen--s10435-10a49311-a1 | 0.629729 | 3.36009 | 5.03042 | 6.25765 | 8.53703 | 9.97027 | 16.103 |
| 20261106T051228Z-c1-confirmation-adamw_prewar-s10435-0a606037-a1 | 0.430528 | 2.40948 | 3.60226 | 4.64198 | 6.15543 | 7.66725 | 11.1803 |
| 20261106T051931Z-c1-confirmation-adamw_prewar-s10435-6ba72121-a1 | 0.665539 | 3.40732 | 5.09685 | 6.32332 | 8.66094 | 10.0954 | 16.3546 |
| 20261106T052730Z-c1-confirmation-adamw-1.45e--s10436-8f2ed530-a1 | 0.135508 | 2.53703 | 3.23886 | 4.25627 | 5.48747 | 8.05008 | 13.5323 |
| 20261106T053346Z-c1-confirmation-adamw-2.11e--s10436-09e9d7d0-a1 | 0.296614 | 3.88294 | 5.34928 | 7.34717 | 8.98285 | 11.751 | 20.1143 |
| 20261106T054006Z-c1-confirmation-adam_frozen--s10436-8a52448f-a1 | 0.117081 | 2.41756 | 3.24554 | 4.14905 | 5.4339 | 7.7698 | 14.6663 |
| 20261106T054724Z-c1-confirmation-adam_frozen--s10436-62a7f4d9-a1 | 0.256153 | 4.38671 | 5.8839 | 7.76527 | 9.97989 | 12.5802 | 21.8855 |
| 20261106T055440Z-c1-confirmation-adamw_prewar-s10436-9c1bf40b-a1 | 0.119707 | 2.41403 | 3.2321 | 4.12542 | 5.39963 | 7.56652 | 14.6209 |
| 20261106T060200Z-c1-confirmation-adamw_prewar-s10436-afe92eef-a1 | 0.294243 | 3.72355 | 5.13136 | 6.94408 | 8.69571 | 11.1095 | 20.9928 |
| 20261106T061019Z-c1-confirmation-adamw-1.45e--s10437-2d02cfa4-a1 | 0.617905 | 2.48658 | 3.31024 | 4.52622 | 5.46668 | 6.74705 | 9.32536 |
| 20261106T061619Z-c1-confirmation-adamw-2.11e--s10437-475d699e-a1 | 0.924883 | 4.12072 | 5.65459 | 7.4547 | 9.18818 | 10.6954 | 13.8842 |
| 20261106T062217Z-c1-confirmation-adam_frozen--s10437-ffc8c12f-a1 | 0.686012 | 2.3596 | 3.23581 | 4.25182 | 5.11534 | 6.11624 | 9.43661 |
| 20261106T062917Z-c1-confirmation-adam_frozen--s10437-84e4ed22-a1 | 0.86551 | 4.33393 | 5.49272 | 7.40501 | 8.9223 | 10.0415 | 15.0258 |
| 20261106T063616Z-c1-confirmation-adamw_prewar-s10437-a2724948-a1 | 0.691693 | 2.35765 | 3.21544 | 4.23396 | 5.11226 | 6.07112 | 9.37461 |
| 20261106T064323Z-c1-confirmation-adamw_prewar-s10437-b0e2d329-a1 | 0.94011 | 3.85579 | 5.08781 | 6.6178 | 8.22158 | 9.5692 | 13.642 |
| 20261106T065130Z-c1-confirmation-adamw-1.45e--s10438-505849a9-a1 | 0.644703 | 2.96222 | 3.98347 | 5.30035 | 6.48011 | 7.5709 | 12.6287 |
| 20261106T065729Z-c1-confirmation-adamw-2.11e--s10438-d8caab8f-a1 | 1.10548 | 4.42616 | 6.48326 | 8.06056 | 10.7021 | 12.4661 | 19.2631 |
| 20261106T070330Z-c1-confirmation-adam_frozen--s10438-2abf2415-a1 | 0.664648 | 2.72893 | 3.64568 | 4.72041 | 6.06426 | 6.9995 | 13.3743 |
| 20261106T071032Z-c1-confirmation-adam_frozen--s10438-8b0a1461-a1 | 1.21331 | 4.7642 | 6.17107 | 8.11902 | 10.2331 | 11.7392 | 20.2029 |
| 20261106T071736Z-c1-confirmation-adamw_prewar-s10438-99fff2aa-a1 | 0.661364 | 2.71959 | 3.63271 | 4.70299 | 6.03981 | 6.973 | 13.2737 |
| 20261106T072442Z-c1-confirmation-adamw_prewar-s10438-5d922af5-a1 | 1.23913 | 4.67822 | 6.63956 | 8.38726 | 10.5141 | 11.7752 | 19.644 |
| 20261106T073245Z-c1-confirmation-adamw-1.45e--s10439-876c48e5-a1 | 0.130672 | 2.20776 | 3.33735 | 4.57232 | 5.97314 | 7.53766 | 13.3576 |
| 20261106T073902Z-c1-confirmation-adamw-2.11e--s10439-193cf5c6-a1 | 0.193723 | 3.55957 | 5.28518 | 7.28196 | 9.22593 | 11.1585 | 18.6847 |
| 20261106T074514Z-c1-confirmation-adam_frozen--s10439-5badb301-a1 | 0.16443 | 2.62146 | 3.66663 | 5.14143 | 6.24463 | 8.01765 | 12.7076 |
| 20261106T075225Z-c1-confirmation-adam_frozen--s10439-86712ce0-a1 | 0.213853 | 3.55644 | 5.54836 | 7.46471 | 9.20525 | 11.7414 | 15.1163 |
| 20261106T075930Z-c1-confirmation-adamw_prewar-s10439-58d812ea-a1 | 0.12193 | 2.48762 | 3.44268 | 4.99007 | 6.21954 | 7.99739 | 13.1076 |
| 20261106T080650Z-c1-confirmation-adamw_prewar-s10439-83b15d3c-a1 | 0.175393 | 3.6282 | 5.41156 | 7.31072 | 9.13591 | 11.3238 | 16.904 |
| 20261106T081508Z-c1-confirmation-adamw-1.45e--s10440-4e2322ff-a1 | 0.41299 | 2.07264 | 3.01301 | 4.22984 | 5.38775 | 7.50811 | 10.1386 |
| 20261106T082122Z-c1-confirmation-adamw-2.11e--s10440-bdce69ce-a1 | 0.510638 | 3.69798 | 5.04042 | 7.00803 | 8.45013 | 11.4113 | 14.9053 |
| 20261106T082738Z-c1-confirmation-adam_frozen--s10440-eb1dfe56-a1 | 0.418612 | 2.39151 | 3.43665 | 4.68719 | 6.11437 | 7.94881 | 10.941 |
| 20261106T083446Z-c1-confirmation-adam_frozen--s10440-7c105018-a1 | 0.451827 | 4.20468 | 5.67031 | 7.73323 | 9.22674 | 12.8454 | 16.9104 |
| 20261106T084201Z-c1-confirmation-adamw_prewar-s10440-c2a1c9bd-a1 | 0.419619 | 2.38475 | 3.45929 | 4.70318 | 6.07809 | 7.89436 | 10.9106 |
| 20261106T084919Z-c1-confirmation-adamw_prewar-s10440-7abf1dca-a1 | 0.43076 | 4.2574 | 5.77906 | 7.87586 | 9.52236 | 12.5331 | 17.0541 |
| 20261106T085738Z-c1-confirmation-adamw-1.45e--s10441-ee07330c-a1 | 0.660383 | 2.58417 | 3.67702 | 4.73681 | 5.80004 | 6.99215 | 9.78451 |
| 20261106T090334Z-c1-confirmation-adamw-2.11e--s10441-9064d65b-a1 | 0.881506 | 4.20354 | 5.89108 | 7.41102 | 9.30909 | 11.0038 | 14.7127 |
| 20261106T090929Z-c1-confirmation-adam_frozen--s10441-7e4ba88a-a1 | 0.444078 | 2.8818 | 3.8785 | 5.17372 | 6.45545 | 7.67087 | 10.5246 |
| 20261106T091633Z-c1-confirmation-adam_frozen--s10441-afaab4eb-a1 | 1.12735 | 4.53824 | 6.45903 | 8.6495 | 10.6455 | 12.5906 | 18.9019 |
| 20261106T092347Z-c1-confirmation-adamw_prewar-s10441-bc879582-a1 | 0.463073 | 2.87337 | 3.86235 | 5.14137 | 6.39009 | 7.59548 | 10.3404 |
| 20261106T093102Z-c1-confirmation-adamw_prewar-s10441-e72c79e7-a1 | 1.59061 | 4.72658 | 6.72249 | 8.85771 | 10.8945 | 12.5303 | 18.0152 |
| 20261106T093913Z-c1-confirmation-adamw-1.45e--s10442-d546b246-a1 | 0.169582 | 2.70013 | 3.45407 | 4.51281 | 5.66838 | 7.33331 | 11.8033 |
| 20261106T094508Z-c1-confirmation-adamw-2.11e--s10442-020f6463-a1 | 0.39482 | 4.41109 | 5.70796 | 7.52279 | 9.39317 | 11.29 | 18.7076 |
| 20261106T095109Z-c1-confirmation-adam_frozen--s10442-0b388ee6-a1 | 0.142951 | 2.43891 | 3.1594 | 4.13791 | 5.19992 | 6.73394 | 11.1664 |
| 20261106T095759Z-c1-confirmation-adam_frozen--s10442-55f08873-a1 | 0.365363 | 3.97166 | 5.14892 | 6.63841 | 8.44402 | 9.78017 | 15.8139 |
| 20261106T100459Z-c1-confirmation-adamw_prewar-s10442-150e42ad-a1 | 0.142321 | 2.43403 | 3.15141 | 4.12246 | 5.18441 | 6.70892 | 11.1375 |
| 20261106T101200Z-c1-confirmation-adamw_prewar-s10442-1c04c0da-a1 | 0.366884 | 4.01352 | 5.20128 | 6.70249 | 8.53984 | 9.90416 | 15.9729 |
| 20261106T102002Z-c1-confirmation-adamw-1.45e--s10443-be8ba3b8-a1 | 0.621964 | 2.42519 | 3.41336 | 4.5588 | 5.79739 | 7.27159 | 13.4416 |
| 20261106T102601Z-c1-confirmation-adamw-2.11e--s10443-f16af950-a1 | 0.610941 | 3.68669 | 5.33312 | 7.25156 | 8.99605 | 10.8054 | 20.2212 |
| 20261106T103202Z-c1-confirmation-adam_frozen--s10443-4e5fb601-a1 | 0.426001 | 2.51264 | 3.5465 | 4.83734 | 6.28722 | 7.95629 | 13.7821 |
| 20261106T103901Z-c1-confirmation-adam_frozen--s10443-dfbe2b08-a1 | 0.784262 | 3.94561 | 5.04104 | 7.14399 | 9.01352 | 10.9942 | 18.8629 |
| 20261106T104602Z-c1-confirmation-adamw_prewar-s10443-56bd3f76-a1 | 0.476545 | 2.53962 | 3.62998 | 4.89463 | 6.29796 | 8.14457 | 14.1936 |
| 20261106T105310Z-c1-confirmation-adamw_prewar-s10443-6b0730cf-a1 | 0.803896 | 3.97048 | 5.04379 | 7.20317 | 8.97013 | 11.089 | 19.0633 |
| 20261106T110116Z-c1-confirmation-adamw-1.45e--s10444-e0164083-a1 | 0.22209 | 2.95408 | 3.82414 | 5.07842 | 6.27794 | 7.90408 | 16.505 |
| 20261106T110715Z-c1-confirmation-adamw-2.11e--s10444-be7d3f53-a1 | 0.45475 | 5.26564 | 6.45554 | 8.33696 | 10.0798 | 12.0051 | 15.696 |
| 20261106T111317Z-c1-confirmation-adam_frozen--s10444-210e006e-a1 | 0.204953 | 2.75588 | 3.72863 | 4.86241 | 6.19096 | 7.66211 | 15.3398 |
| 20261106T112016Z-c1-confirmation-adam_frozen--s10444-bf80ad9a-a1 | 0.3642 | 4.66887 | 5.80089 | 7.47141 | 9.29614 | 11.6332 | 15.4352 |
| 20261106T112712Z-c1-confirmation-adamw_prewar-s10444-59bb8e7d-a1 | 0.203576 | 2.74494 | 3.70623 | 4.84582 | 6.16146 | 7.62997 | 15.3201 |
| 20261106T113417Z-c1-confirmation-adamw_prewar-s10444-970e7cf9-a1 | 0.370444 | 4.72582 | 5.86105 | 7.55493 | 9.38892 | 11.7276 | 15.4917 |
| 20261106T114227Z-c1-confirmation-adamw-1.45e--s10445-bba36acf-a1 | 0.316444 | 3.04458 | 3.64093 | 4.59003 | 6.45664 | 7.76071 | 11.3947 |
| 20261106T114913Z-c1-confirmation-adamw-2.11e--s10445-5642b8d1-a1 | 0.430436 | 4.6516 | 5.92602 | 7.42143 | 9.85524 | 11.9158 | 16.2538 |
| 20261106T115457Z-c1-confirmation-adam_frozen--s10445-cdcb2d67-a1 | 0.376869 | 3.41535 | 4.15315 | 5.20251 | 7.15919 | 8.72049 | 11.9691 |
| 20261106T120238Z-c1-confirmation-adam_frozen--s10445-ef9e56c4-a1 | 0.433099 | 4.83898 | 6.28279 | 7.61184 | 10.1392 | 12.4093 | 15.9873 |
| 20261106T121037Z-c1-confirmation-adamw_prewar-s10445-81e9d837-a1 | 0.375397 | 3.39702 | 4.08771 | 5.18299 | 7.07899 | 8.6438 | 11.8771 |
| 20261106T121753Z-c1-confirmation-adamw_prewar-s10445-1f8f4285-a1 | 0.44705 | 4.95548 | 6.30357 | 7.66598 | 10.2158 | 12.4798 | 15.8993 |
| 20261106T122554Z-c1-confirmation-adamw-1.45e--s10446-e0c17003-a1 | 0.235978 | 2.31613 | 3.54559 | 4.42879 | 5.76498 | 7.11681 | 16.3956 |
| 20261106T123209Z-c1-confirmation-adamw-2.11e--s10446-496fc9c3-a1 | 0.44375 | 4.02205 | 5.93147 | 7.33879 | 9.52072 | 11.4549 | 20.5353 |
| 20261106T123827Z-c1-confirmation-adam_frozen--s10446-dacca9a8-a1 | 0.28019 | 2.55362 | 4.04428 | 4.98371 | 6.33602 | 8.27606 | 18.1678 |
| 20261106T124546Z-c1-confirmation-adam_frozen--s10446-e8248db7-a1 | 0.255816 | 3.68133 | 5.99109 | 7.20912 | 9.75237 | 11.9243 | 23.0521 |
| 20261106T125259Z-c1-confirmation-adamw_prewar-s10446-6541242e-a1 | 0.23017 | 2.46292 | 3.98199 | 4.93341 | 6.14695 | 7.92677 | 17.9312 |
| 20261106T130022Z-c1-confirmation-adamw_prewar-s10446-f0e1a99e-a1 | 0.239217 | 3.62025 | 6.03254 | 7.2277 | 9.6489 | 11.8708 | 23.0199 |
| 20261106T134639Z-c1-confirmation-adamw-1.45e--s10448-e470f7e5-a1 | 0.201061 | 2.86059 | 3.37463 | 4.56343 | 5.62678 | 6.79089 | 10.3215 |
| 20261106T135248Z-c1-confirmation-adamw-2.11e--s10448-531e57a7-a1 | 0.243538 | 4.2752 | 5.4127 | 7.1049 | 8.48783 | 10.6039 | 15.2756 |
| 20261106T135902Z-c1-confirmation-adam_frozen--s10448-87fa87d5-a1 | 0.17961 | 2.94673 | 3.47853 | 4.74052 | 5.78414 | 7.06547 | 10.8815 |
| 20261106T140610Z-c1-confirmation-adam_frozen--s10448-e6551cb3-a1 | 0.229992 | 4.20941 | 5.59581 | 7.48321 | 8.92967 | 11.1571 | 16.1631 |
| 20261106T141320Z-c1-confirmation-adamw_prewar-s10448-05c45311-a1 | 0.179481 | 2.93058 | 3.45288 | 4.70294 | 5.76401 | 7.03622 | 10.8326 |
| 20261106T142043Z-c1-confirmation-adamw_prewar-s10448-28edca25-a1 | 0.236815 | 4.20097 | 5.57699 | 7.49271 | 8.95729 | 11.2181 | 16.1605 |
| 20261106T142920Z-c1-confirmation-adamw-1.45e--s10449-b4039e6d-a1 | 0.69562 | 2.53594 | 3.26599 | 4.44799 | 5.58541 | 6.76268 | 10.1667 |
| 20261106T143515Z-c1-confirmation-adamw-2.11e--s10449-7efc26a2-a1 | 1.04622 | 3.87933 | 5.63567 | 7.51959 | 9.43527 | 11.0201 | 14.9062 |
| 20261106T144117Z-c1-confirmation-adam_frozen--s10449-fc58b8a8-a1 | 0.615442 | 2.6597 | 3.44939 | 4.58317 | 6.17484 | 6.97212 | 9.58681 |
| 20261106T144801Z-c1-confirmation-adam_frozen--s10449-b1017928-a1 | 0.972366 | 3.84348 | 5.44827 | 7.43148 | 9.54127 | 10.9816 | 13.1977 |
| 20261106T145504Z-c1-confirmation-adamw_prewar-s10449-d78e054e-a1 | 0.612028 | 2.65134 | 3.43218 | 4.56792 | 6.13739 | 6.93597 | 9.54867 |
| 20261106T150201Z-c1-confirmation-adamw_prewar-s10449-fe743b1a-a1 | 0.974642 | 3.86078 | 5.49561 | 7.49356 | 9.64337 | 11.045 | 13.2608 |
| 20261106T151013Z-c1-confirmation-adamw-1.45e--s10450-a5119e85-a1 | 0.303835 | 2.54237 | 3.74305 | 4.93781 | 6.17097 | 7.93034 | 10.5192 |
| 20261106T151634Z-c1-confirmation-adamw-2.11e--s10450-35e97c85-a1 | 0.446995 | 3.87692 | 6.01208 | 7.83408 | 9.58659 | 12.0671 | 15.7824 |
| 20261106T152301Z-c1-confirmation-adam_frozen--s10450-45d64859-a1 | 0.367153 | 2.58419 | 3.76262 | 4.8355 | 6.11014 | 7.63923 | 9.84181 |
| 20261106T153019Z-c1-confirmation-adam_frozen--s10450-d018db69-a1 | 0.580209 | 3.8939 | 6.09483 | 7.95316 | 9.71703 | 11.814 | 18.6346 |
| 20261106T153746Z-c1-confirmation-adamw_prewar-s10450-37fc6692-a1 | 0.363532 | 2.57461 | 3.75048 | 4.82276 | 6.08924 | 7.61804 | 9.80268 |
| 20261106T154519Z-c1-confirmation-adamw_prewar-s10450-216ba616-a1 | 0.762413 | 4.00153 | 6.14297 | 7.97084 | 9.86171 | 12.1018 | 18.8068 |
| 20261106T155354Z-c1-confirmation-adamw-1.45e--s10451-ead89fbc-a1 | 0.621265 | 3.17152 | 3.87511 | 4.62583 | 6.01565 | 7.33344 | 13.7504 |
| 20261106T160010Z-c1-confirmation-adamw-2.11e--s10451-64c7412b-a1 | 0.823157 | 4.823 | 6.20484 | 7.66312 | 9.15302 | 11.3689 | 15.8225 |
| 20261106T160635Z-c1-confirmation-adam_frozen--s10451-0290f7f1-a1 | 0.558992 | 2.84794 | 3.63304 | 4.28401 | 5.53172 | 7.13477 | 12.2929 |
| 20261106T161347Z-c1-confirmation-adam_frozen--s10451-2da4dd0a-a1 | 0.706906 | 4.3281 | 5.40308 | 6.75205 | 8.41304 | 10.1317 | 13.2128 |
| 20261106T162050Z-c1-confirmation-adamw_prewar-s10451-06304257-a1 | 0.559059 | 2.83227 | 3.62016 | 4.27441 | 5.51086 | 7.11428 | 12.2867 |
| 20261106T162752Z-c1-confirmation-adamw_prewar-s10451-fe9f7ae9-a1 | 0.709465 | 4.3736 | 5.44911 | 6.79808 | 8.49223 | 10.2452 | 13.2993 |
| 20261106T163548Z-c1-confirmation-adamw-1.45e--s10452-26cbd45b-a1 | 0.253678 | 2.67378 | 3.64939 | 4.78946 | 6.1567 | 7.64649 | 12.0373 |
| 20261106T164110Z-c1-confirmation-adamw-2.11e--s10452-ada330e5-a1 | 0.382083 | 4.08944 | 6.00469 | 7.80072 | 9.65033 | 11.8493 | 15.5946 |
| 20261106T164654Z-c1-confirmation-adam_frozen--s10452-066c5837-a1 | 0.224153 | 2.30315 | 3.3431 | 4.38842 | 5.75828 | 7.14285 | 13.2937 |
| 20261106T165323Z-c1-confirmation-adam_frozen--s10452-5263ea31-a1 | 0.389767 | 3.66428 | 5.2698 | 6.93941 | 8.84856 | 10.7358 | 15.9217 |
| 20261106T165956Z-c1-confirmation-adamw_prewar-s10452-ba73545c-a1 | 0.223378 | 2.29587 | 3.33224 | 4.37328 | 5.7238 | 7.1162 | 13.2342 |
| 20261106T170632Z-c1-confirmation-adamw_prewar-s10452-116bd477-a1 | 0.394237 | 3.70805 | 5.31822 | 7.02062 | 8.93747 | 10.8168 | 15.9625 |
| 20261106T171414Z-c1-confirmation-adamw-1.45e--s10453-b3dd244b-a1 | 0.135515 | 2.20294 | 3.22117 | 4.33227 | 5.9275 | 7.18169 | 12.0741 |
| 20261106T172009Z-c1-confirmation-adamw-2.11e--s10453-3eb338b4-a1 | 0.192773 | 4.08952 | 5.50433 | 7.15945 | 9.4501 | 10.9619 | 18.0051 |
| 20261106T172605Z-c1-confirmation-adam_frozen--s10453-63e6f15e-a1 | 0.219738 | 2.72667 | 3.4812 | 4.79128 | 6.13165 | 7.48437 | 10.1339 |
| 20261106T173305Z-c1-confirmation-adam_frozen--s10453-e76b98bd-a1 | 0.279231 | 4.14963 | 5.61571 | 7.25438 | 9.23249 | 11.086 | 18.1162 |
| 20261106T174000Z-c1-confirmation-adamw_prewar-s10453-fa7ec0cf-a1 | 0.226021 | 2.70401 | 3.49708 | 4.77157 | 6.19704 | 7.47507 | 10.3185 |
| 20261106T174704Z-c1-confirmation-adamw_prewar-s10453-6de0dce9-a1 | 0.369504 | 4.19234 | 5.7442 | 7.43093 | 9.50135 | 11.3529 | 18.9049 |
| 20261106T175513Z-c1-confirmation-adamw-1.45e--s10454-50d58bda-a1 | 0.298662 | 3.25593 | 3.95605 | 5.04775 | 6.02752 | 7.7217 | 11.3635 |
| 20261106T180114Z-c1-confirmation-adamw-2.11e--s10454-1c281c92-a1 | 0.398652 | 5.04366 | 6.64664 | 8.62755 | 10.3786 | 12.2175 | 16.3944 |
| 20261106T180712Z-c1-confirmation-adam_frozen--s10454-b546f9e3-a1 | 0.263138 | 2.84547 | 3.30173 | 4.20766 | 5.05583 | 6.61005 | 10.0823 |
| 20261106T181410Z-c1-confirmation-adam_frozen--s10454-1aa0a53b-a1 | 0.311164 | 4.62405 | 5.51977 | 7.01425 | 8.46571 | 10.5612 | 14.3979 |
| 20261106T182107Z-c1-confirmation-adamw_prewar-s10454-ed264284-a1 | 0.26217 | 2.82793 | 3.30355 | 4.20243 | 5.04688 | 6.59319 | 10.0603 |
| 20261106T182813Z-c1-confirmation-adamw_prewar-s10454-c8fb712e-a1 | 0.320601 | 4.64846 | 5.56514 | 7.09227 | 8.56699 | 10.6606 | 14.5122 |
| 20261106T183621Z-c1-confirmation-adamw-1.45e--s10455-d0e2fda0-a1 | 0.854551 | 2.53252 | 3.60154 | 4.52515 | 5.57756 | 6.80229 | 11.5141 |
| 20261106T184209Z-c1-confirmation-adamw-2.11e--s10455-410f55d9-a1 | 0.853371 | 4.27313 | 5.82096 | 8.13659 | 9.64003 | 11.1782 | 16.2253 |
| 20261106T184801Z-c1-confirmation-adam_frozen--s10455-95d57fa1-a1 | 0.830823 | 2.52251 | 3.41448 | 4.61979 | 5.41607 | 6.51519 | 11.6474 |
| 20261106T185436Z-c1-confirmation-adam_frozen--s10455-e21e2ee6-a1 | 1.01947 | 4.07824 | 5.38171 | 7.78144 | 9.03172 | 10.4281 | 14.917 |
| 20261106T190132Z-c1-confirmation-adamw_prewar-s10455-31c341cb-a1 | 0.835617 | 2.50586 | 3.38872 | 4.58196 | 5.37631 | 6.49371 | 11.82 |
| 20261106T190818Z-c1-confirmation-adamw_prewar-s10455-60403de3-a1 | 1.00919 | 4.06256 | 5.34354 | 7.77338 | 8.98614 | 10.4457 | 15.6375 |
| 20261106T191625Z-c1-confirmation-adamw-1.45e--s10456-ad18e357-a1 | 0.272178 | 2.84802 | 3.87468 | 4.89654 | 5.66875 | 7.23273 | 10.7439 |
| 20261106T192156Z-c1-confirmation-adamw-2.11e--s10456-cf207430-a1 | 0.253719 | 4.0694 | 5.95974 | 7.87846 | 9.22198 | 11.2865 | 14.8371 |
| 20261106T192742Z-c1-confirmation-adam_frozen--s10456-21fac1a9-a1 | 0.304652 | 3.11645 | 4.23767 | 5.38794 | 6.31318 | 8.20899 | 11.0204 |
| 20261106T193410Z-c1-confirmation-adam_frozen--s10456-56aa51dd-a1 | 0.501455 | 5.00599 | 6.38562 | 8.40377 | 10.1004 | 12.547 | 16.0105 |
| 20261106T194107Z-c1-confirmation-adamw_prewar-s10456-6ea61100-a1 | 0.302007 | 3.13642 | 4.23277 | 5.32401 | 6.24179 | 8.16673 | 10.9587 |
| 20261106T194742Z-c1-confirmation-adamw_prewar-s10456-31a55204-a1 | 0.877905 | 5.13384 | 6.49551 | 8.56449 | 10.3959 | 12.8448 | 16.5365 |
| 20261106T195551Z-c1-confirmation-adamw-1.45e--s10457-320cc91e-a1 | 0.677513 | 2.33084 | 3.57593 | 4.54346 | 5.67161 | 6.98693 | 11.8012 |
| 20261106T200136Z-c1-confirmation-adamw-2.11e--s10457-f046eead-a1 | 0.93158 | 3.75994 | 5.83507 | 7.78488 | 9.56088 | 11.1104 | 19.0778 |
| 20261106T200735Z-c1-confirmation-adam_frozen--s10457-cc88687d-a1 | 0.666648 | 2.58601 | 3.69266 | 5.00174 | 6.05069 | 7.29965 | 12.5383 |
| 20261106T201426Z-c1-confirmation-adam_frozen--s10457-e62b4968-a1 | 0.764716 | 3.81916 | 5.93463 | 8.14794 | 9.80171 | 10.832 | 18.6521 |
| 20261106T202123Z-c1-confirmation-adamw_prewar-s10457-798f3e53-a1 | 0.671077 | 2.54412 | 3.65368 | 4.95782 | 5.99597 | 7.25397 | 12.4703 |
| 20261106T202820Z-c1-confirmation-adamw_prewar-s10457-f10173aa-a1 | 0.753267 | 3.87509 | 5.96851 | 8.22414 | 9.91599 | 10.8895 | 18.6217 |
| 20261106T203627Z-c1-confirmation-adamw-1.45e--s10458-49b9487f-a1 | 0.630265 | 2.27109 | 2.96855 | 4.14966 | 5.33095 | 6.63737 | 11.8865 |
| 20261106T204154Z-c1-confirmation-adamw-2.11e--s10458-f8642123-a1 | 1.10412 | 4.11764 | 5.29034 | 7.01863 | 8.26014 | 10.4466 | 17.6983 |
| 20261106T204736Z-c1-confirmation-adam_frozen--s10458-65e68fb2-a1 | 0.781503 | 2.84867 | 3.72333 | 4.7851 | 6.02221 | 7.77197 | 15.1212 |
| 20261106T205403Z-c1-confirmation-adam_frozen--s10458-858da9bf-a1 | 1.40688 | 4.76307 | 6.20439 | 7.83314 | 9.4163 | 12.3492 | 20.7486 |
| 20261106T210043Z-c1-confirmation-adamw_prewar-s10458-25ba1617-a1 | 0.766306 | 2.83255 | 3.6954 | 4.75847 | 5.98347 | 7.6826 | 15.0038 |
| 20261106T210724Z-c1-confirmation-adamw_prewar-s10458-add3069f-a1 | 1.39577 | 4.7952 | 6.23788 | 7.84658 | 9.4574 | 12.357 | 20.7289 |
| 20261106T211516Z-c1-confirmation-adamw-1.45e--s10459-273ece61-a1 | 0.531036 | 2.60711 | 3.79605 | 4.88715 | 5.72421 | 7.07232 | 10.7661 |
| 20261106T212107Z-c1-confirmation-adamw-2.11e--s10459-3c101b0a-a1 | 1.13072 | 4.59845 | 6.02209 | 7.6106 | 9.20049 | 11.1969 | 17.0806 |
| 20261106T212705Z-c1-confirmation-adam_frozen--s10459-e8b33bab-a1 | 0.549038 | 2.70094 | 3.59823 | 4.46464 | 5.43411 | 6.63551 | 10.4511 |
| 20261106T213356Z-c1-confirmation-adam_frozen--s10459-312c2f4a-a1 | 0.865369 | 3.97344 | 5.21527 | 6.61644 | 7.91496 | 9.68445 | 15.382 |
| 20261106T214053Z-c1-confirmation-adamw_prewar-s10459-d1c2c37d-a1 | 0.546236 | 2.68985 | 3.58608 | 4.44686 | 5.41931 | 6.61868 | 10.4161 |
| 20261106T214752Z-c1-confirmation-adamw_prewar-s10459-abba82b9-a1 | 0.872781 | 4.02249 | 5.26459 | 6.68526 | 8.02962 | 9.79158 | 15.4819 |
| 20261106T223258Z-c1-confirmation-adamw-1.45e--s10461-5e34d70b-a1 | 0.591376 | 2.50533 | 3.47797 | 4.41236 | 5.66911 | 6.98454 | 9.52022 |
| 20261106T223856Z-c1-confirmation-adamw-2.11e--s10461-c1f86ff4-a1 | 0.824514 | 3.90322 | 5.78569 | 7.24905 | 9.11161 | 10.712 | 14.9342 |
| 20261106T224456Z-c1-confirmation-adam_frozen--s10461-bba87de5-a1 | 0.580332 | 2.39103 | 3.23683 | 3.90289 | 5.19619 | 6.43859 | 8.73481 |
| 20261106T225152Z-c1-confirmation-adam_frozen--s10461-3392b665-a1 | 0.734039 | 3.60447 | 5.2841 | 6.34593 | 8.1608 | 9.55728 | 13.8935 |
| 20261106T225851Z-c1-confirmation-adamw_prewar-s10461-14af7100-a1 | 0.575572 | 2.3829 | 3.2249 | 3.89036 | 5.18204 | 6.41703 | 8.71069 |
| 20261106T230555Z-c1-confirmation-adamw_prewar-s10461-71e1d749-a1 | 0.735751 | 3.64155 | 5.33621 | 6.40624 | 8.248 | 9.52082 | 13.9874 |
| 20261106T231404Z-c1-confirmation-adamw-1.45e--s10462-ce6875f9-a1 | 0.262758 | 2.33566 | 3.39335 | 4.22512 | 5.62574 | 7.82547 | 11.9571 |
| 20261106T232002Z-c1-confirmation-adamw-2.11e--s10462-ec81f13b-a1 | 0.570704 | 3.86109 | 5.69413 | 7.16337 | 8.95437 | 11.5571 | 17.6308 |
| 20261106T232607Z-c1-confirmation-adam_frozen--s10462-a56fada9-a1 | 0.257429 | 2.18516 | 3.32577 | 4.11695 | 5.5264 | 7.21415 | 12.941 |
| 20261106T233308Z-c1-confirmation-adam_frozen--s10462-ec58f793-a1 | 0.674342 | 3.73397 | 5.39962 | 6.51541 | 8.08514 | 10.7643 | 19.0977 |
| 20261106T234008Z-c1-confirmation-adamw_prewar-s10462-966488ed-a1 | 0.253993 | 2.17828 | 3.30862 | 4.09328 | 5.50774 | 7.20478 | 12.909 |
| 20261106T234716Z-c1-confirmation-adamw_prewar-s10462-64ce56ac-a1 | 0.682657 | 3.76031 | 5.45631 | 6.57297 | 8.16817 | 10.9139 | 19.0691 |
| 20261107T000723Z-c1-confirmation-adam_frozen--s10463-d2ddc508-a1 | 0.618375 | 3.1427 | 3.93419 | 4.83196 | 6.31267 | 7.39537 | 10.8525 |
| 20261107T001418Z-c1-confirmation-adam_frozen--s10463-e3a0d1a4-a1 | 1.11378 | 4.65017 | 5.8611 | 7.65384 | 9.73231 | 11.8794 | 17.4618 |
| 20261107T002115Z-c1-confirmation-adamw_prewar-s10463-02c7a939-a1 | 0.614069 | 3.12935 | 3.92072 | 4.80829 | 6.29435 | 7.37683 | 10.8112 |
| 20261107T002821Z-c1-confirmation-adamw_prewar-s10463-1c8842fc-a1 | 1.10394 | 4.68749 | 5.91902 | 7.70557 | 9.82962 | 11.9584 | 17.6167 |


## validation_harm

| field | value |
|---|---|
| record | study3-final/c1/validation-harm.json |
| sha256 | 7b5defd40fbde70abc1af103e5ab5bbf5295ca1544fafaa9795a9be1264e6f0d |
| analysis | {"path": "study3-final/c1/analysis.json", "root": "runs", "sha256": "a639d5646cdcfa433253be5bcbe9b80ac6febb1db913c58944c03495144bf145"} |

| comparison | n | status | wording | primary (pp) | missing labels (pp) | all missing harmful (pp) | x (pp) | y (pp) | runs missing a label | labels missing |
|---|---|---|---|---|---|---|---|---|---|---|
| A-F | 27 | descriptive | direction_below_5 | [-4.654, -0.351] | [-4.651, -0.351] | [-4.651, -0.356] | 0.351 | 4.654 | 1 | 1 |
| P-F | 26 | descriptive | below_2_5_direction_undetermined | [-0.413, 0.349] | [-0.415, 0.357] | [-0.413, 0.349] | null | null | 2 | 2 |

| comparison | missing labels: d_low source (pp) | missing labels: d_high source (pp) |
|---|---|---|
| A-F | [-4.651, -0.356] | [-4.654, -0.351] |
| P-F | [-0.415, 0.341] | [-0.410, 0.357] |


## audit

| field | value |
|---|---|
| record | study3-final/c1/audit/estimate.json |
| sha256 | 2b308736dbce24c8b4489521e41a9eeb94fd1ade3d14c7e50e8f4ccb8be30feb |
| selection | {"path": "study3-final/c1/audit/selection.json", "root": "runs", "sha256": "f4bb942c5f36f7e60c6038087614d27c665622a39ae8e4ca36c7f5b095a2ffd1"} |
| missing | {"draws": 43, "items": 43} |

| E | undetermined | MAR B_E (pp) | SE (pp) | worst low (pp) | worst high (pp) | MAR outside the worst case | missing draws | missing items |
|---|---|---|---|---|---|---|---|---|
| AF | false | -0.046 | 1.229 | -4.588 | 3.242 | false | 31 | 31 |
| PF | false | 0.249 | 1.476 | -4.592 | 5.992 | false | 31 | 31 |

| B_E: the MAR estimate and the worst case |
|---|
| B_E's MAR estimate averages over the draws with a resolved reference only, while each missing draw's term in the worst-case bound is fixed in sign by its known judge label: the MAR estimate can lie outside the worst-case bound |

| arm | draws | distinct | tp | fn | fp | tn | missing draws | harmful by reference | sensitivity | sensitivity worst case | FPR | FPR worst case | undetermined |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 400 | 400 | 42 | 8 | 3 | 335 | 12 | 50 | 0.84 | 0.7, 0.846154 | 0.00887574 | 0.00862069, 0.0147059 | false |
| F | 400 | 400 | 61 | 8 | 3 | 308 | 20 | 69 | 0.884058 | 0.709302, 0.888889 | 0.0096463 | 0.00914634, 0.0191083 | false |
| P | 200 | 200 | 25 | 4 | 1 | 159 | 11 | 29 | 0.862069 | 0.641026, 0.866667 | 0.00625 | 0.00588235, 0.0124224 | false |

| reference source | items |
|---|---|
| agreement | 742 |
| arbiter | 215 |
| no decision | 43 |
| no vote | 0 |

| agreements checked | failed | labeled | rejected | share rejected | the check |
|---|---|---|---|---|---|
| 75 | 75 | 0 | 0 | null | the share of the checked agreements the arbiter model rejects (its label differs from the agreed one on the two-class outcome), over those it labeled: an estimate of the consensus's error; the check changes no reference label |

| non-estimable | why |
|---|---|


## labeling

| run with a technical labeling problem |
|---|


## freeze

| field | value |
|---|---|
| path | docs/study3-analysis-freeze-c1.json |
| sha256 | 0e8f2640fb06b58bbfdec873518e90daa0382944f813b958221779f501eb5703 |
| candidate | 1 |
| confirmation | {"N": 60, "active": ["A-F", "P-F"], "cap": 74, "outcome": "confirmed", "started": [10400, 10401, 10402, 10403, 10404, 10405, 10406, 10407, 10408, 10409, 10410, 10411, 10412, 10413, 10414, 10415, 10416, 10417, 10418, 10419, 10420, 10421, 10422, 10423, 10424, 10425, 10426, 10427, 10428, 10429, 10430, 10431, 10432, 10433, 10434, 10435, 10436, 10437, 10438, 10439, 10440, 10441, 10442, 10443, 10444, 10445, 10446, 10447, 10448, 10449, 10450, 10451, 10452, 10453, 10454, 10455, 10456, 10457, 10458, 10459, 10460, 10461, 10462, 10463], "stop": "counters"} |
| K*_h | 5.87602 |
| matched blocks | 34 |
| left out | 26 |

| cohort | reading | rule | blocks | runs | unknown blocks |
|---|---|---|---|---|---|
| A-F | confirmatory | n_reached | 60 | 240 | 10428, 10447, 10460, 10463 |
| P-F | confirmatory | n_reached | 60 | 240 | 10414, 10428, 10447, 10460 |
| A-P | null | null | 62 | 248 | null |
| validation A-F | null | null | 27 | 108 | null |
| validation P-F | null | null | 26 | 104 | null |


## judge_mark

| judge mark |
|---|
| null |


## money

| stage | spent (USD) |
|---|---|
| dry_run | 12.35 |
| engineering | 45.58 |
| simulation | 0 |
| search | 4.21 |
| numerical | 16.38 |
| validation | 21.91 |
| confirmation | 56.42 |
| labeling | 11.05 |
| audit | 0 |

| incident | stage | incident_time | cause | billed (USD) | evidence |
|---|---|---|---|---|---|
| engineering-cycle-c1-8dcfe20 | engineering | 2026-10-31T23:09:03Z | case 17's diagnosis and the re-run of the engineering trajectories under fix 8dcfe204415685976ba219727369f80032fc5314 | 28.24 | launch/study3-c1-engineering-20261030T113942Z.jsonl line 51; launch/study3-c1-engineering-20261030T113942Z.jsonl line 54; launch/study3-c1-engineering-20261030T113942Z.jsonl line 57; launch/study3-c1-engineering-20261030T113942Z.jsonl line 60; launch/study3-c1-engineering-20261030T113942Z.jsonl line 63; launch/study3-c1-engineering-20261030T113942Z.jsonl line 66; launch/study3-c1-engineering-20261030T113942Z.jsonl line 69; launch/study3-c1-engineering-20261030T113942Z.jsonl line 72; launch/study3-c1-engineering-20261030T113942Z.jsonl line 75; launch/study3-c1-engineering-20261030T113942Z.jsonl line 78; launch/study3-c1-engineering-20261030T113942Z.jsonl line 81; launch/study3-c1-engineering-20261030T113942Z.jsonl line 84; launch/study3-c1-engineering-20261030T113942Z.jsonl line 87; launch/study3-c1-engineering-20261030T113942Z.jsonl line 90; launch/study3-c1-engineering-20261030T113942Z.jsonl line 93; launch/study3-c1-engineering-20261030T113942Z.jsonl line 96; pod iu69haoozd6ued |
| dry-run-image-pull | dry_run | 2026-10-28T14:21:47Z | the image pull failed and the container never started | 1.03 | pod k89018y06rfzq1 |
| dry-run-unfit-volume | dry_run | 2026-10-29T18:16:52Z | the Pod's volume was a network file system, unfit for the run | 1.14 | pod 2xyqxu4mgc0yos |
| engineering-unfit-volume | engineering | 2026-10-30T09:51:11Z | the Pod's volume was a network file system, unfit for the run | 0.12 | pod a6n73b0kyvq7hs |
| confirmation-unfit-volume | confirmation | 2026-11-04T22:27:56Z | the Pod's volume was a network file system, unfit for the run | 0.17 | pod 6s0ljn2e0maqm0 |
| labeling-gate-per-run | labeling | 2026-11-07T06:42:10Z | the labeling read the freeze gate once per run and labeled nothing until fix 860700a, restarted 08:04:35Z | 1.5 | pod fnew8k5gjgrbcu |

| incident of the records not billed | stage | incident_time | cause | billed (USD) |
|---|---|---|---|---|

| stage stopped by the internal spending policy |
|---|


## embargo

| stage | what |
|---|---|

