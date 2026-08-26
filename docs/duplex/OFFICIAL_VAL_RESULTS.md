# Weakly supervised baselines — official validation

Code commit: `0e153783fba36f2d5005a327c2caa8d6c1985782`

| Method | Venue | Supervision | Corpus | Seeds | Frame ROC | Frame PR | Video ROC | Video AP | Within-hate ROC |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| vadclip | AAAI 2024 | video-level labels | hatemm | 3 | 0.6423 ± 0.1233 | 0.3683 ± 0.1100 | 0.6767 ± 0.1532 | 0.5769 ± 0.1588 | 0.4420 ± 0.0529 (n=85) |
| vadclip | AAAI 2024 | video-level labels | mhclip_en | 3 | 0.6542 ± 0.0155 | 0.3711 ± 0.0596 | 0.7093 ± 0.0310 | 0.5139 ± 0.0188 | 0.2587 ± 0.0777 (n=44) |
| vadclip | AAAI 2024 | video-level labels | mhclip_zh | 3 | 0.6186 ± 0.0629 | 0.3158 ± 0.0612 | 0.6304 ± 0.0808 | 0.4055 ± 0.0858 | 0.3925 ± 0.1854 (n=8) |
| vadclip | AAAI 2024 | video-level labels | hateclipseg | 3 | 0.5035 ± 0.0111 | 0.5247 ± 0.0117 | 0.5401 ± 0.0121 | 0.8726 ± 0.0143 | 0.5332 ± 0.0190 (n=67) |
| dsanet | AAAI 2026 | video-level labels | hatemm | 3 | 0.7223 ± 0.0258 | 0.4521 ± 0.0300 | 0.8108 ± 0.0179 | 0.7564 ± 0.0110 | 0.5255 ± 0.0496 (n=85) |
| dsanet | AAAI 2026 | video-level labels | mhclip_en | 3 | 0.6856 ± 0.0121 | 0.4007 ± 0.0249 | 0.7479 ± 0.0129 | 0.5093 ± 0.0272 | 0.3300 ± 0.0725 (n=44) |
| dsanet | AAAI 2026 | video-level labels | mhclip_zh | 3 | 0.7126 ± 0.0324 | 0.3823 ± 0.0238 | 0.7297 ± 0.0206 | 0.4901 ± 0.0235 | 0.4047 ± 0.0995 (n=8) |
| dsanet | AAAI 2026 | video-level labels | hateclipseg | 3 | 0.5280 ± 0.0230 | 0.5535 ± 0.0227 | 0.5966 ± 0.0253 | 0.9090 ± 0.0023 | 0.5450 ± 0.0108 (n=67) |
| macilsd | ACM MM 2022 | video-level labels | hatemm | 3 | 0.8068 ± 0.0194 | 0.5733 ± 0.0330 | 0.8146 ± 0.0245 | 0.7334 ± 0.0324 | 0.5948 ± 0.0058 (n=85) |
| macilsd | ACM MM 2022 | video-level labels | mhclip_en | 3 | 0.7240 ± 0.0304 | 0.4370 ± 0.0307 | 0.7226 ± 0.0312 | 0.4923 ± 0.0321 | 0.5435 ± 0.0055 (n=44) |
| macilsd | ACM MM 2022 | video-level labels | mhclip_zh | 3 | 0.7521 ± 0.0047 | 0.4614 ± 0.0068 | 0.7532 ± 0.0090 | 0.5350 ± 0.0155 | 0.3806 ± 0.0728 (n=8) |
| macilsd | ACM MM 2022 | video-level labels | hateclipseg | 3 | 0.4765 ± 0.0196 | 0.5159 ± 0.0192 | 0.4792 ± 0.0244 | 0.8537 ± 0.0292 | 0.5162 ± 0.0209 (n=67) |
| macilsd_audio | ACM MM 2022 | video-level labels | hatemm | 3 | 0.7730 ± 0.0424 | 0.4921 ± 0.0502 | 0.8082 ± 0.0212 | 0.7354 ± 0.0307 | 0.6014 ± 0.0088 (n=85) |
| macilsd_audio | ACM MM 2022 | video-level labels | mhclip_en | 3 | 0.6643 ± 0.0529 | 0.4269 ± 0.0518 | 0.7142 ± 0.0260 | 0.5329 ± 0.0418 | 0.4842 ± 0.0125 (n=44) |
| macilsd_audio | ACM MM 2022 | video-level labels | mhclip_zh | 3 | 0.5982 ± 0.0156 | 0.3043 ± 0.0239 | 0.6513 ± 0.0291 | 0.4257 ± 0.0098 | 0.5094 ± 0.0830 (n=8) |
| macilsd_audio | ACM MM 2022 | video-level labels | hateclipseg | 3 | 0.4300 ± 0.0296 | 0.4789 ± 0.0189 | 0.4556 ± 0.0920 | 0.8575 ± 0.0339 | 0.4914 ± 0.0142 (n=67) |
| macilsd_visual | ACM MM 2022 | video-level labels | hatemm | 3 | 0.6869 ± 0.0045 | 0.4411 ± 0.0076 | 0.7015 ± 0.0093 | 0.5966 ± 0.0029 | 0.5261 ± 0.0020 (n=85) |
| macilsd_visual | ACM MM 2022 | video-level labels | mhclip_en | 3 | 0.6362 ± 0.0127 | 0.3498 ± 0.0079 | 0.6627 ± 0.0145 | 0.4253 ± 0.0098 | 0.5120 ± 0.0205 (n=44) |
| macilsd_visual | ACM MM 2022 | video-level labels | mhclip_zh | 3 | 0.7201 ± 0.0102 | 0.4423 ± 0.0118 | 0.7521 ± 0.0117 | 0.5343 ± 0.0076 | 0.4584 ± 0.0160 (n=8) |
| macilsd_visual | ACM MM 2022 | video-level labels | hateclipseg | 3 | 0.4987 ± 0.0064 | 0.5515 ± 0.0064 | 0.4988 ± 0.0134 | 0.8876 ± 0.0076 | 0.5260 ± 0.0069 (n=67) |
| multihateloc | WWW 2026 | video-level labels | hatemm | 3 | 0.7288 ± 0.0112 | 0.4960 ± 0.0101 | 0.8588 ± 0.0121 | 0.8064 ± 0.0100 | 0.6315 ± 0.0027 (n=85) |
| multihateloc | WWW 2026 | video-level labels | mhclip_en | 3 | 0.6534 ± 0.0436 | 0.3722 ± 0.0395 | 0.7014 ± 0.0052 | 0.4950 ± 0.0167 | 0.5743 ± 0.0390 (n=44) |
| multihateloc | WWW 2026 | video-level labels | mhclip_zh | 3 | 0.6521 ± 0.0172 | 0.3665 ± 0.0219 | 0.7029 ± 0.0099 | 0.4461 ± 0.0453 | 0.5120 ± 0.0165 (n=8) |
| multihateloc | WWW 2026 | video-level labels | hateclipseg | 3 | 0.5244 ± 0.0214 | 0.5390 ± 0.0176 | 0.6027 ± 0.0429 | 0.8984 ± 0.0109 | 0.5242 ± 0.0090 (n=67) |
| cmhkf | ACL 2025 Long | video-level labels | hatemm | 3 | 0.7083 ± 0.0253 | 0.4292 ± 0.0120 | 0.7816 ± 0.0258 | 0.7061 ± 0.0119 | 0.4422 ± 0.0099 (n=85) |
| cmhkf | ACL 2025 Long | video-level labels | mhclip_en | 3 | 0.7272 ± 0.0020 | 0.4519 ± 0.0502 | 0.7402 ± 0.0180 | 0.5133 ± 0.0201 | 0.6004 ± 0.2149 (n=44) |
| cmhkf | ACL 2025 Long | video-level labels | mhclip_zh | 3 | 0.7395 ± 0.0139 | 0.4175 ± 0.0031 | 0.7742 ± 0.0171 | 0.5898 ± 0.0184 | 0.5300 ± 0.0293 (n=8) |
| cmhkf | ACL 2025 Long | video-level labels | hateclipseg | 3 | 0.4872 ± 0.0020 | 0.5152 ± 0.0174 | 0.5353 ± 0.0682 | 0.8898 ± 0.0148 | 0.5081 ± 0.0250 (n=67) |
| fed_wsvad_1client | AAAI 2025 | video-level labels | hatemm | 3 | 0.7186 ± 0.0047 | 0.4350 ± 0.0164 | 0.7885 ± 0.0097 | 0.7263 ± 0.0117 | 0.5102 ± 0.0300 (n=85) |
| fed_wsvad_1client | AAAI 2025 | video-level labels | mhclip_en | 3 | 0.6895 ± 0.0268 | 0.3591 ± 0.0341 | 0.6278 ± 0.0216 | 0.3551 ± 0.0209 | 0.5323 ± 0.0356 (n=44) |
| fed_wsvad_1client | AAAI 2025 | video-level labels | mhclip_zh | 3 | 0.7297 ± 0.0152 | 0.3992 ± 0.0135 | 0.7653 ± 0.0141 | 0.5393 ± 0.0551 | 0.5069 ± 0.0304 (n=8) |
| fed_wsvad_1client | AAAI 2025 | video-level labels | hateclipseg | 3 | 0.5218 ± 0.0341 | 0.5360 ± 0.0170 | 0.5343 ± 0.0649 | 0.8871 ± 0.0338 | 0.5048 ± 0.0102 (n=67) |
| fed_wsvad_3client | AAAI 2025 | video-level labels | hatemm | 3 | 0.7289 ± 0.0065 | 0.4421 ± 0.0159 | 0.7995 ± 0.0097 | 0.6969 ± 0.0317 | 0.5201 ± 0.0189 (n=85) |
| fed_wsvad_3client | AAAI 2025 | video-level labels | mhclip_en | 3 | 0.6895 ± 0.0044 | 0.3845 ± 0.0060 | 0.6445 ± 0.0370 | 0.4186 ± 0.0353 | 0.5217 ± 0.0070 (n=44) |
| fed_wsvad_3client | AAAI 2025 | video-level labels | mhclip_zh | 3 | 0.7491 ± 0.0495 | 0.4134 ± 0.0306 | 0.7845 ± 0.0263 | 0.5774 ± 0.0369 | 0.4456 ± 0.1057 (n=8) |
| fed_wsvad_3client | AAAI 2025 | video-level labels | hateclipseg | 3 | 0.5090 ± 0.0127 | 0.5619 ± 0.0358 | 0.5618 ± 0.0274 | 0.9031 ± 0.0151 | 0.5100 ± 0.0184 (n=67) |
| vera | CVPR 2025 | validation-selected; training-free | hatemm | 1 | 0.6217 | 0.3730 | 0.6275 | 0.5836 | 0.5587 (n=85) |
| vera | CVPR 2025 | validation-selected; training-free | mhclip_en | 1 | 0.5468 | 0.2945 | 0.5481 | 0.3417 | 0.5408 (n=44) |
| vera | CVPR 2025 | validation-selected; training-free | mhclip_zh | 1 | 0.5127 | 0.2540 | 0.5072 | 0.2978 | 0.5000 (n=8) |
| vera | CVPR 2025 | validation-selected; training-free | hateclipseg | 1 | 0.6050 | 0.6194 | 0.7000 | 0.9335 | 0.5619 (n=67) |
