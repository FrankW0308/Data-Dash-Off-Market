# Part B speaker script

Approx. 145 words; rehearse at about 116 words/minute for 75 seconds.

My part estimates a home's sale price before we know its listing price. I compared simple price baselines, a random forest, histogram gradient boosting, and several CatBoost configurations. I tuned only on October and November 2025, then froze Histogram gradient boosting before evaluating December through May.

For each validation month, the model learned only from sales before that month. Its inputs describe size, rooms, property type, synthetic location and listing date. Fixed description keywords add condition and amenity information. List price, post-sale fields and external valuations are excluded.

Across 10,069 held-out homes, median absolute percentage error was 9.08 percent, and mean absolute percentage error was 11.60 percent. The community baseline's median error was 17.89 percent.

These are measured validation results, not hidden-test scores. Synthetic data and market changes limit generalization. The final deliverable is a reproducible model and one prediction for every off-market test home.
