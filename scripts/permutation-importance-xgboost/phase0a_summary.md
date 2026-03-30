# Phase 0a: Ground Truth Variation:Permutation Importance Pilot


## Q47: All in all, how would you describe your state of health these days?

### CV accuracy vs. majority-class baseline

- **Germany** (n=1525): CV accuracy 0.477 ± 0.020 (above baseline 0.462) \
-> Model hardly learns anything, only slightly above baseline, but baseline is within 95% CI of CV accuracy

- **Nigeria** (n=1235): CV accuracy 0.587 ± 0.011 (above baseline 0.438) \
-> Model learns something and substantially improves over baseline

- **Japan** (n=1341): CV accuracy 0.441 ± 0.016 (above baseline 0.377) \
-> Model learns something and improves over baseline

- **Brazil** (n=1757): CV accuracy 0.496 ± 0.008 (above baseline 0.453) \
-> Model hardly learns anything, only slightly above baseline, but at least baseline is outside 95% CI

- **Egypt** (n=1200): CV accuracy 0.598 ± 0.031 (above baseline 0.503) \
-> Model learns something and substantially improves over baseline

### Mean pairwise Spearman ρ across countries
Mean pairwise Spearman $\rho$: **0.050** (range: -0.088 to 0.276)

### Which features are universally important vs. country-specific?
- **Universally important features**: 
**Q46 (Feeling of happiness)** stands out as the top predictor in almost every country (Japan, Nigeria, Egypt), with consistently elevated importance, in Brazil it is the second most important feature. In Germany it is also the third most important feature, albeit with a smaller importance value.
Q262 (Age) appears as the most important feature in Brazil, Germany, and Egypt.
Q49 (Overall satisfaction with life) is also among the top predictors in multiple countries (Germany, Japan, Nigeria), though with more variation in importance values across countries and less high importance.
- **Country-specific important features**:
Q262 (Age) is the dominant predictor in Egypt (0.0567) and Germany (0.0141) but drops to near-zero or negative in Nigeria. 
Q48 (Perceived freedom and control over one's life) matters in Japan (0.0151) and Egypt (0.0094) but is negligible in Germany or Brazil. 
Q246 (Importance of civil rights protecting against oppression) lights up in Egypt (0.0087) but is weak elsewhere. Q49 (Overall satisfaction with life) is notably important in Japan (0.0150) compared to other countries.

### Noteworthy pattern
In Japan Q118 (Freqency ordinary people pay a bribe, give a gift or do a favor) is very important in Japan, but not in any other country.

## Q57: Generally speaking, would you say that most people can be trusted or that you need to be very careful in dealing with people?

### CV accuracy vs. majority-class baseline

- **Germany** (n=1482): CV accuracy 0.748 ± 0.015 (above baseline 0.540) \
-> Model learns something and substantially improves over baseline
- **Nigeria** (n=1230): CV accuracy 0.889 ± 0.025 (above baseline 0.873) \
-> Model learns something and improves over baseline
- **Japan** (n=1281): CV accuracy 0.715 ± 0.022 (above baseline 0.644) \
-> Model learns something and improves  substantially over baseline
- **Brazil** (n=1730): CV accuracy 0.936 ± 0.005 (above baseline 0.934) \
-> Model hardly learns anything, only slightly above baseline, but at least baseline is outside 95% CI
- **Egypt** (n=1197): CV accuracy 0.921 ± 0.017 (below baseline 0.926) \
-> Model does not learn anything, CV accuracy is below baseline, and baseline is not within 95% CI

### Mean pairwise Spearman ρ across countries
Mean pairwise Spearman $\rho$: **0.005** (range: -0.098 to 0.106)

### Which features are universally important vs. country-specific?
- **Universally important features**:
None clearly universal. The importance in general is very sparse and low across the board.
- **Country-specific important features**:
Q61 (Trust: People you meet for the first time) is the dominant predictor in Germany (0.0646) and Japan (0.0393) but near-zero elsewhere. Q59 (Trust: Your neighbourhood) matters primarily in Nigeria (0.0104). Q66 (Confidence: The Press) is important in Japan (0.0140) but negligible in all other countries. Q62 (Trust: People of another religion) matters in Japan (0.0090) and to a lesser extent Nigeria (0.0029), but not elsewhere. This target shows the most country-specific structure of all five.

### Noteworthy pattern
Q59 (Trust: Your neighbourhood) is the most important predictor in Nigeria, but has way lower importance in all other countries. In general, Japan has higher importances across the board for this target.

## Q199: How interested would you say you are in politics?

### CV accuracy vs. majority-class baseline

- **Germany** (n=1528): CV accuracy 0.607 ± 0.044 (above baseline 0.459) \
-> Model learns something and substantially improves over baseline
- **Nigeria** (n=1233): CV accuracy 0.567 ± 0.035 (above baseline 0.304) \
-> Model learns something and substantially improves over baseline
- **Japan** (n=1321): CV accuracy 0.640 ± 0.044 (above baseline 0.512) \
-> Model learns something and substantially improves over baseline
- **Brazil** (n=1734): CV accuracy 0.562 ± 0.031 (above baseline 0.396) \
-> Model learns something and substantially improves over baseline
- **Egypt** (n=1195): CV accuracy 0.696 ± 0.015 (above baseline 0.391) \
-> Model learns something and substantially improves over baseline

### Mean pairwise Spearman ρ across countries
Mean pairwise Spearman $\rho$: **0.070** (range: -0.070 to 0.149)

### Which features are universally important vs. country-specific?
- **Universally important features**: **Q200** (How often discusses political matters with friends) and **Q4** (Important in life: Politics) are the top two predictors across all five countries.
- **Country-specific important features**: Q200 is particularly dominant in Egypt and far higher than in any other country. Q202 (Information source: TV news) matters in Egypt (0.0110) but is near-zero or negative elsewhere. Q190 (Justifiability of parents beating children) shows a surprising spike in Egypt (0.0096) with no relevance in other countries. Q208 (Information source: Talk with friends) is important in Egypt (0.0058) but negligible elsewhere. Overall, Egypt has a distinctly different importance profile for this question.

### Noteworthy pattern
Q234 is way more important in Japan than in any other country. Also Q98 (Active/Inactive membership: Political Party) is very important in Nigeria but negligible in other countries.

## Q235: Do you think having a strong leader who does not have to consult parliament or hold elections is a good or bad way of governing this country?

### CV accuracy vs. majority-class baseline

- **Germany** (n=1443): CV accuracy 0.570 ± 0.009 (above baseline 0.527) \
-> Model learns something and improves over baseline
- **Nigeria** (n=1222): CV accuracy 0.575 ± 0.023 (above baseline 0.302) \
-> Model learns something and substantially improves over baseline
- **Japan** (n=1141): CV accuracy 0.452 ± 0.018 (above baseline 0.369) \
-> Model learns something and improves over baseline
- **Brazil** (n=1512): CV accuracy 0.483 ± 0.020 (above baseline 0.416) \
-> Model learns something and improves over baseline
- **Egypt** (n=935): CV accuracy 0.684 ± 0.016 (above baseline 0.366) \
-> Model learns something and substantially improves over baseline

### Mean pairwise Spearman ρ across countries
Mean pairwise Spearman $\rho$: **0.051** (range: -0.082 to 0.188)

### Which features are universally important vs. country-specific?
- **Universally important features**: Q236 (Political system: Having experts, not government, make decisions) is the top predictor in every country: Brazil (0.0699), Japan (0.0642), Nigeria (0.0965), Egypt (0.0421), and in Germany (0.0078), but with a far lower magnitude.

- **Country-specific important features**: Q234A (How much the political system allows people to have a say) matters in Egypt (0.0120) and in Nigeria (second highest) but in general is less important. Q228 (Journalists provide fair coverage) shows up in Brazil (0.0034) and Nigeria (0.0033) but not in other countries. Q277 (Respondent's mother's education level) is important in Japan (0.0078) and Egypt (0.0032) but negligible in Brazil and Nigeria. Q169 (Belief that religion is always right) matters in Japan (0.0043) but barely registers in Germany or Brazil. Despite Q236 being universal, the secondary predictors differ substantially across countries.

### Noteworthy pattern
Germany has in general much lower importance values across the board for this target compared to all other countries.

## Q164: How important is God in your life?

### CV accuracy vs. majority-class baseline

- **Germany** (n=1508): CV accuracy 0.477 ± 0.023 (above baseline 0.268) \
-> Model learns something and substantially improves over baseline
- **Nigeria** (n=1237): CV accuracy 0.877 ± 0.004 (above baseline 0.863) \
-> Model learns something and improves a bit over baseline
- **Japan** (n=1246): CV accuracy 0.307 ± 0.036 (above baseline 0.177) \
-> Model learns something and substantially improves over baseline
- **Brazil** (n=1745): CV accuracy 0.829 ± 0.017 (above baseline 0.800) \
-> Model learns something and improves over baseline
- **Egypt** (n=1189): CV accuracy 0.974 ± 0.011 (below baseline 0.976) \
-> Model does not learn anything, CV accuracy is below baseline, and baseline is not within 95% CI

### Mean pairwise Spearman ρ across countries
Mean pairwise Spearman $\rho$: **0.011** (range: -0.039 to 0.080)

### Which features are universally important vs. country-specific?
- **Universally important features**: Q165 (Believe in God) is overwhelmingly important in Germany (0.1071) and Nigeria (0.0628) and relevant in Brazil (0.0101), making it the closest thing to a universal predictor here. Q172 (How often do you pray) also shows elevated importance in Germany (0.0327) and to a lesser extent Japan (0.0034).
- **Country-specific important features**: Germany has by far the highest importances. Q6 (Important in life: Religion) matters in Japan (0.0208) and Nigeria (0.0114) but not elsewhere. Egypt shows near-zero importance for every feature, confirming the expected ceiling effect (almost everyone answers the maximum). Brazil also shows very weak signal. The predictive structure essentially only exists in Germany and partially in Japan/Nigeria.

### Noteworthy pattern
Nigeria and Egypt have near zero (or even zero) importance values across the board.

## Overall Verdict

Overall mean pairwise Spearman $\rho$ across all targets: **0.038**

- Is there enough cross-national variation in predictive structure to justify the main experiment?\
**Proceed** \
The overall mean pairwise Spearman $\rho$ across all five targets is very low (0.038), indicating that the importance rankings of predictors differ substantially across countries. So the feature importance rankings are essentially uncorellated across countries. 
- Any concerns about data quality, sample size, or modelling choices that should be addressed before Phase 1?\
**No major concerns** \
For Q57 (trust) some models struggel to learn at all and don't improve much over baseline or are even below baseline (Egypt). Also for Q164 (importance of God) the model does not learn anything in Egypt. Also Q47 (health) in Germany is not over the baseline and Brazil is only slightly above baseline. 