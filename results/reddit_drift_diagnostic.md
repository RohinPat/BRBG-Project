# Drift diagnostic: cross-bin vs within-bin APD

*Generated: 2026-04-22 02:15 UTC*

Model: `roberta-base`; min-bin=20; max-per-bin=120; n-splits=5.

drift_score = apd_cross - mean(apd_within_early, apd_within_late). Values near 0 mean the cross-bin distance is no larger than the distance between two random halves of the same bin -> no evidence of drift beyond within-word variance / polysemy.

## Group summary

| Group | n | mean cross APD | mean within APD | mean drift | frac(drift>0) |
|-------|---:|---------------:|----------------:|-----------:|--------------:|
| Brainrot | 43 | 0.1484 | 0.1438 | +0.0046 | 0.93 |
| General slang | 25 | 0.1893 | 0.1798 | +0.0095 | 0.92 |
| Standard control | 25 | 0.1166 | 0.1162 | +0.0004 | 0.76 |

## Per-word

| Word | Group | cross APD | within APD | drift |
|------|-------|----------:|-----------:|------:|
| vibe | general_slang | 0.5421 | 0.4426 | +0.0995 |
| ghosted | general_slang | 0.3790 | 0.3282 | +0.0508 |
| simp | brainrot | 0.3886 | 0.3443 | +0.0443 |
| slay | general_slang | 0.3914 | 0.3497 | +0.0417 |
| unc | brainrot | 0.1966 | 0.1737 | +0.0229 |
| flex | general_slang | 0.3094 | 0.2884 | +0.0210 |
| rizz | brainrot | 0.2504 | 0.2299 | +0.0206 |
| cap | brainrot | 0.3480 | 0.3280 | +0.0200 |
| yeet | brainrot | 0.2593 | 0.2417 | +0.0176 |
| cope | brainrot | 0.1749 | 0.1591 | +0.0158 |
| ick | brainrot | 0.1307 | 0.1191 | +0.0117 |
| ratio | brainrot | 0.1376 | 0.1295 | +0.0081 |
| glaze | brainrot | 0.7630 | 0.7578 | +0.0052 |
| shook | general_slang | 0.2335 | 0.2283 | +0.0051 |
| mid | brainrot | 0.1673 | 0.1626 | +0.0048 |
| woke | general_slang | 0.1456 | 0.1413 | +0.0043 |
| glazing | brainrot | 0.1673 | 0.1636 | +0.0037 |
| skibidi | brainrot | 0.0687 | 0.0654 | +0.0033 |
| normie | brainrot | 0.0737 | 0.0710 | +0.0027 |
| mog | brainrot | 0.1678 | 0.1652 | +0.0025 |
| pressed | brainrot | 0.2129 | 0.2104 | +0.0025 |
| salty | general_slang | 0.1084 | 0.1059 | +0.0025 |
| door | standard_control | 0.1469 | 0.1444 | +0.0024 |
| mewing | brainrot | 0.0837 | 0.0817 | +0.0020 |
| extra | general_slang | 0.1267 | 0.1247 | +0.0020 |
| lit | general_slang | 0.1619 | 0.1600 | +0.0019 |
| table | standard_control | 0.1478 | 0.1461 | +0.0017 |
| period | general_slang | 0.2043 | 0.2026 | +0.0017 |
| delulu | brainrot | 0.1005 | 0.0991 | +0.0014 |
| based | general_slang | 0.1804 | 0.1792 | +0.0012 |
| market | standard_control | 0.1063 | 0.1054 | +0.0009 |
| snatched | general_slang | 0.2909 | 0.2900 | +0.0009 |
| school | standard_control | 0.1069 | 0.1060 | +0.0009 |
| bet | general_slang | 0.1645 | 0.1636 | +0.0009 |
| letter | standard_control | 0.1333 | 0.1325 | +0.0009 |
| bridge | standard_control | 0.1201 | 0.1193 | +0.0008 |
| poggers | brainrot | 0.0555 | 0.0546 | +0.0008 |
| wig | general_slang | 0.1096 | 0.1088 | +0.0008 |
| summer | standard_control | 0.1003 | 0.0996 | +0.0008 |
| sigma | brainrot | 0.1399 | 0.1392 | +0.0007 |
| hopium | brainrot | 0.1077 | 0.1070 | +0.0007 |
| drip | general_slang | 0.1449 | 0.1442 | +0.0007 |
| lore | brainrot | 0.1160 | 0.1154 | +0.0007 |
| sus | general_slang | 0.1457 | 0.1451 | +0.0007 |
| unalive | brainrot | 0.1078 | 0.1071 | +0.0006 |
| ohio | brainrot | 0.1072 | 0.1066 | +0.0006 |
| aura | brainrot | 0.1951 | 0.1945 | +0.0006 |
| cheugy | brainrot | 0.0439 | 0.0433 | +0.0006 |
| mald | brainrot | 0.1072 | 0.1065 | +0.0006 |
| looksmax | brainrot | 0.1663 | 0.1657 | +0.0006 |
| chronically online | brainrot | 0.1143 | 0.1138 | +0.0005 |
| cringe | brainrot | 0.1405 | 0.1400 | +0.0005 |
| side eye | brainrot | 0.0814 | 0.0809 | +0.0005 |
| river | standard_control | 0.1237 | 0.1233 | +0.0004 |
| paper | standard_control | 0.1185 | 0.1181 | +0.0004 |
| goat | general_slang | 0.1114 | 0.1110 | +0.0004 |
| ocean | standard_control | 0.1344 | 0.1340 | +0.0004 |
| npc | brainrot | 0.1165 | 0.1161 | +0.0004 |
| fire | general_slang | 0.1505 | 0.1502 | +0.0003 |
| no cap | general_slang | 0.0701 | 0.0698 | +0.0003 |
| copium | brainrot | 0.0831 | 0.0828 | +0.0003 |
| book | standard_control | 0.1268 | 0.1266 | +0.0003 |
| chair | standard_control | 0.1130 | 0.1128 | +0.0002 |
| morning | standard_control | 0.1244 | 0.1242 | +0.0002 |
| lowkey | general_slang | 0.0853 | 0.0851 | +0.0002 |
| camp | general_slang | 0.1712 | 0.1710 | +0.0002 |
| fr | general_slang | 0.1577 | 0.1575 | +0.0002 |
| garden | standard_control | 0.1373 | 0.1371 | +0.0002 |
| hospital | standard_control | 0.1003 | 0.1001 | +0.0002 |
| main character | brainrot | 0.0768 | 0.0766 | +0.0002 |
| brainrot | brainrot | 0.1009 | 0.1008 | +0.0001 |
| rent free | brainrot | 0.0914 | 0.0913 | +0.0001 |
| gigachad | brainrot | 0.0638 | 0.0637 | +0.0001 |
| chad | brainrot | 0.1218 | 0.1217 | +0.0001 |
| city | standard_control | 0.1399 | 0.1399 | +0.0001 |
| parasocial | brainrot | 0.1283 | 0.1283 | +0.0001 |
| touch grass | brainrot | 0.0599 | 0.0599 | +0.0001 |
| window | standard_control | 0.1159 | 0.1159 | +0.0000 |
| bussin | brainrot | 0.0478 | 0.0477 | +0.0000 |
| tea | general_slang | 0.1118 | 0.1118 | +0.0000 |
| bottle | standard_control | 0.1032 | 0.1032 | +0.0000 |
| phone | standard_control | 0.0831 | 0.0831 | +0.0000 |
| goated | brainrot | 0.0693 | 0.0693 | -0.0000 |
| forest | standard_control | 0.1141 | 0.1142 | -0.0001 |
| iconic | general_slang | 0.1158 | 0.1158 | -0.0001 |
| music | standard_control | 0.0867 | 0.0868 | -0.0001 |
| friend | standard_control | 0.0778 | 0.0779 | -0.0001 |
| water | standard_control | 0.1101 | 0.1102 | -0.0001 |
| coffee | standard_control | 0.1204 | 0.1205 | -0.0002 |
| highkey | general_slang | 0.1205 | 0.1208 | -0.0003 |
| winter | standard_control | 0.1234 | 0.1238 | -0.0004 |
| seethe | brainrot | 0.1389 | 0.1393 | -0.0004 |
| mogging | brainrot | 0.1098 | 0.1112 | -0.0014 |