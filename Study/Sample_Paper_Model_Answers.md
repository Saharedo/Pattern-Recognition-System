# Pattern Recognition Systems (ISY5002)

## Model answers: the official sample paper (Nov 2026)

*These are model answers written by a tutor from the sample paper, the course notes and our study guides. They are not the official marking scheme. Marks are in square brackets. Use them to see how to structure an answer: claim, numbers from the question, reason, action.*

## Question 1 (20 marks)

**a.** (4 marks: what the tree shows, link to the business, one problem and fix, how to judge value)

**What the tree shows.** Reading the tree: the root is 84% no purchase and 16% purchase. The first split is **PageValue \< 1**. The 78% of sessions with PageValue below 1 are almost all non-buyers (96% FALSE, 4% buy). The 22% of sessions with PageValue of at least 1 buy 58% of the time. Within these, the split on **BounceRate** (cut-off 81e−6, which is essentially zero) separates them: a bounce rate of about zero gives 74% buyers (10% of all sessions), while a higher bounce rate gives 45% buyers (12% of all sessions). So likely purchasers visit pages with high economic value (PageValue) and do not bounce. \[1\]

**Link to the business.** The bookstore wants to find who is about to buy so it can send time-limited coupons. The leaf with PageValue of at least 1 and a near-zero bounce rate (10% of sessions, 74% buy) is the group to target. The 45% group is also worth a coupon if a coupon is cheap, and the 78% group is not. This focuses the coupon budget on the right customers, which improves conversion and profit. \[1\]

**One problem and a fix.** The data is imbalanced (84% against 16%) and the tree at a 0.5 cut-off only predicts TRUE for one leaf (10% of sessions). The leaf with 45% buyers (about 34% of all buyers) and the 78% leaf (about 20% of buyers) are predicted FALSE, so on the training data only about 7.4 of the 15.9 buyers per 100 sessions are found: **recall about 47%**. Fixes: use class weights or a cost-sensitive loss or oversampling (SMOTE); lower the decision threshold on the predicted probability (for example 0.4); grow a slightly deeper tree and prune it with validation data; or use an ensemble. \[1\]

**How to tell the improved model is more valuable.** Compare old and new on the same untouched 30% test set using recall, precision, F1 and AUC (not accuracy, which is 84% for "always no"). Convert to money: profit = (extra sales found × margin) − (coupons sent × coupon cost). Use a lift or gain curve at the coupon budget. Finally run an A/B test in production. \[1\]

**b.** (3 marks) Compute the metrics from each matrix.

|                          | SVM               | NN                 |
|--------------------------|-----------------------|------------------------|
| TP / FN / FP / TN        | 3078 / 72 / 312 / 237 | 3029 / 121 / 261 / 288 |
| Accuracy                 | 3315/3699 = 89.6%     | 3317/3699 = 89.7%      |
| Recall = TP/(TP+FN)      | 3078/3150 = 97.7%     | 3029/3150 = 96.2%      |
| Precision = TP/(TP+FP)   | 3078/3390 = 90.8%     | 3029/3290 = 92.1%      |
| Specificity = TN/(TN+FP) | 237/549 = 43.2%       | 288/549 = 52.5%        |

Accuracy is almost identical, so it cannot separate the models. The difference is the type of mistake: the SVM misses fewer buyers (72 against 121) but raises more false alarms (312 against 261). The business objective is to find the buyers (a wasted coupon is cheap, a lost sale is costly), so favour **recall**: the **SVM** best meets the objective (97.7% against 96.2%), accepting a few more wasted coupons. Mention the other trade-offs: a linear SVM is simpler, the NN is a black box and needs more data. \[3 in total: metrics 1, comparison 1, choice with reason 1\]

**Caution (label swap).** The row labelled "Actual True" totals 3,150 of 3,699 test rows (85%), yet only 15.5% of sessions were purchases (about 573 in the test set). The 549 row looks like the buyers, so the "True" and "False" labels may be swapped. If the 549 row is the buyers, the buyer recall is SVM 237/549 = 43.2% and NN 288/549 = 52.5%, and precision is SVM 237/309 = 76.7% and NN 288/409 = 70.4%, which would favour the NN for finding buyers. In the exam, **state which class you treat as positive and why**, compute under that choice, and add one line about the alternative.

**c.** (3 marks) AdaBoost importance against the single tree.

**Difference.** The single tree uses only two variables, PageValue (root) and BounceRate, and ignores everything else, so its importance is very concentrated. The AdaBoost plot is flat: nearly all variables score between about 0.07 and 0.10, **ProductRelated is the top variable (about 0.10)** even though the tree never uses it, PageValues is only mid-table (about 0.08), and only Weekend (0.00), Region and Month are low. \[1\]

**Reasons.** (1) A single tree is greedy: once PageValue takes the root, correlated variables (ProductRelated and its duration, BounceRates and ExitRates) add little and are never used, so their importance is masked. (2) AdaBoost trains many trees on reweighted data; later trees must fix earlier errors and use other variables, and the importance is summed over all those trees. (3) Impurity-based importance favours variables with many distinct values or levels (Browser and OperatingSystems look surprisingly high), so part of the score is bias. (4) Flat scores suggest many weak signals and no single dominant variable. \[1\]

**Use in further modelling.** Feature selection: drop the lowest-ranked variables (Weekend, Region, Month, TrafficType), retrain the SVM, NN or logistic regression on the rest and compare by cross-validation. Cross-check with permutation importance before trusting the ranking, and keep business-critical variables (PageValue). \[1\]

**d.** (4 marks) Clickstream and purchase probability.

**Technique:** an LSTM (or GRU), because the order of pages, the time on each and the navigation path matter and a plain RNN forgets long sequences. \[1\]

**Training data structure:** each visitor session is one sequence. Each step has features: page category (one-hot or an embedding), time spent, how the user arrived, and where they went next. The label is whether the session ended in a purchase (many-to-one: the final state of the LSTM goes into a Dense(1) with a sigmoid and binary cross-entropy). Alternatively predict the next page. \[1\]

**Pre-processing:** sort events by time; pad or cut sessions to a fixed length (with masking) or use windows; scale numeric inputs; encode categories; handle class imbalance (class weights, oversampling); split train and test **by session (and by time)**, not by random rows. \[1\]

**Combination with earlier models:** feed the LSTM’s purchase probability (or its final hidden state) as an extra feature into the tabular model from part (b) or the AdaBoost model, or stack or average the LSTM and tabular predictions (a hybrid or ensemble). The sequence model adds browsing behaviour that aggregated page counts miss. \[1\]

**e.** (6 marks) PCA on the 18 features.

**(i) How many factors?** (3 marks) With 18 standardised features the total variance is 18, so eigenvalue = %variance × 18 / 100.

| PC       | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|--------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| % var        | 20.7  | 12.3  | 10.0  | 9.3   | 7.5   | 7.2   | 6.9   | 6.7   | 6.2   | 5.4    |
| Cumulative % | 20.7  | 33.0  | 43.0  | 52.3  | 59.8  | 67.0  | 73.9  | 80.6  | 86.8  | 92.2   |
| Eigenvalue   | 3.73  | 2.21  | 1.80  | 1.67  | 1.35  | 1.30  | 1.24  | 1.21  | 1.12  | 0.97   |

**Kaiser rule** (eigenvalue above 1, equivalent to more than 100/18 = 5.56% of the variance): 9 components pass (PC10 at 5.4% just fails). **80% cumulative rule:** 8 components (80.6%). **Scree plot:** PC1 and PC2 stand out, then the line declines slowly with no sharp elbow, which suggests about 3 at most. Answer: about **8 to 9 components** carry the information; I would choose 8 (80.6% of the variance) and note that Kaiser allows 9. The Chief AI Scientist’s idea of a handful of factors (3 or 4) is only weakly supported, because 3 to 4 components keep only 43% to 52% of the variance. \[3\]

*(If you count only the 14 variables actually shown in the table, p = 14 and the Kaiser threshold is 7.1%, which keeps 6 components. State the p you use.)*

**(ii) What each factor stands for, and use as predictors.** (3 marks) A factor is named from the variables with the largest loadings (near ±1), and the signs show which move together.

- **PC1 (20.7%), engaged browsing:** ProductRelated 0.45, ProductRelated_Duration 0.45, Informational 0.36 and Informational_Duration 0.32 are high, and BounceRates −0.35 and ExitRates −0.40 are negative. Visitors who view many pages and stay, and rarely leave straight away.

- **PC2 (12.3%), disengagement:** BounceRates 0.47 and ExitRates 0.45 are high while PageValues −0.31 and Revenue −0.27 are negative. Visitors who leave quickly and do not buy.

- **PC3 (10.0%), visitor technology and source:** OperatingSystems 0.52, Browser 0.49, TrafficType 0.45.

- **PC4 (9.3%), purchase value:** PageValues 0.54 and Revenue 0.53, the pages that lead to money and the purchase itself.

**Use as predictors in a regression.** Good: the components are uncorrelated (no multicollinearity) and there are fewer inputs, which reduces overfitting. Cautions: (1) **leakage**: Revenue, the target, is inside the PCA and PC4 loads on it, so predicting Revenue from these components leaks the answer; run the PCA on the input features only. (2) A component mixes several ideas and loses the meaning of individual features, so it is harder to explain to the business. (3) PCA ignores the target, so the top-variance components are not guaranteed to be the most predictive; check by modelling. (4) Fit the scaler and PCA on training data only, then apply them to the test data. \[3 in total\]

## Question 2 (20 marks)

**a.** (8 marks) The table’s numbers: Flatten gives 9,408 values (14×14×48), Dense 256 has 2,408,704 parameters, Dense 128 has 32,896, Dense 918 has 118,422 and the total is 2,578,982.

**(i) Do you agree?** (2 marks) **Yes.** The first Dense layer after Flatten holds 2,408,704 of 2,578,982 parameters (**about 93%**), far more than 5,000 images can support, so the model will overfit. The output is one 918-way softmax over brand × type × value: with 5,000 images that is only about 5 images per class (5,000/918), unevenly spread, and the classes are treated as unrelated (Samsung-kettle-high and Samsung-kettle-low share nothing). There is also no dropout, batch normalisation or augmentation, and no activation or loss is stated. \[2\]

**(ii) A better model.** (2 marks) Keep the convolutional feature extractor. Replace Flatten with **Global Average Pooling** (14×14×48 becomes 48 values). Then a small shared dense part and **three softmax heads**: Dense(18) for brand, Dense(17) for type and Dense(3) for value (the Keras Functional API with three outputs). The bin is the combination of the three predictions, so all 918 bins are still reachable. Add dropout and batch normalisation, and use augmentation (flip, rotate, crop, brightness) and class weights for the uneven categories. \[2\]

**(iii) Reasons and advantages.** (2 marks) Each head sees many more examples per class: about 5,000/18 ≈ 280 per brand, 5,000/17 ≈ 290 per type and 5,000/3 ≈ 1,670 per value, instead of about 5. Knowledge about a brand is shared across all its types. GAP removes the parameter explosion (9,408 inputs down to 48), makes the model less sensitive to where the object sits and cuts overfitting. Dropout, batch norm and augmentation regularise and exploit the small, uneven dataset. Transfer learning from a pretrained backbone is an option for so few images. \[2\]

**(iv) Parameters, activations and loss.** (2 marks)

| Part          | Original                       | New                                                             |
|-------------------|------------------------------------|---------------------------------------------------------------------|
| Three conv layers | 448 + 4,640 + 13,872 = 18,960      | 18,960 (unchanged)                                                  |
| Dense 256         | 9,408×256 + 256 = 2,408,704        | 48×256 + 256 = 12,544                                               |
| Dense 128         | 32,896                             | 32,896                                                              |
| Output            | 918 units: 128×918 + 918 = 118,422 | (128×18+18) + (128×17+17) + (128×3+3) = 2,322 + 2,193 + 387 = 4,902 |
| Total             | 2,578,982                          | 69,302 (about 97% fewer)                                            |

If you kept Flatten and only split the output into three heads the total would be 2,578,982 − 118,422 + 4,902 = 2,465,462 (only 4% fewer), so the Global Average Pooling change is what matters. **Activations:** ReLU in all hidden layers; **softmax** on each of the three heads (one class out of several). **Loss:** categorical cross-entropy for each head, summed (optionally weighted). If one image could have several labels at once, use sigmoid outputs and binary cross-entropy instead. \[2\]

**b.** (4 marks) Deep reinforcement learning for the sorting arm.

**(i) Three reward factors** (each tied to a criterion, with a sign). (1) **Task success and efficiency:** a large positive reward for placing the correct object in the correct bin, small shaping rewards for moving closer, a good grip point and matching the belt speed, and a small negative reward per time step so it prefers fast picks (effective and efficient picking even in odd positions). (2) **Collision avoidance and safety:** a large negative reward for contact with non-target objects, other robots or humans, and a graded penalty for entering a safe distance, using the external cameras’ tracked positions and trajectories. (3) **Hardware protection:** negative rewards for joint angles near their limits, awkward arm configurations, bad grip points, high torque or jerky motion. \[1\]

**(ii) Architecture.** The RGB-D cameras (on the arm and external) observe the conveyor belt, the **target object**, **other objects** (non-target items, other robot arms, humans) and the **physical robot hardware**. The **identification system** (the classifier from part a) and a perception module turn the images into the **state**: object identity, position and trajectory, gripper pose, joint angles and the positions of obstacles. The **decision-making system** is the multi-layer agent network (for example a CNN encoder for the image part and dense layers that output joint commands, plus a critic in an actor-critic design). A **safety layer** checks every **action** (joint limits, speed, protected zones, emergency stop) before it reaches the arm. A **reward module** scores the result and the **reward** returns to the agent. Draw the loop with arrows labelled state, action and reward. \[2\]

**(iii) Reducing the danger of learning on the real robot.** Train in **simulation** first (a digital twin of the arm, belt and obstacles), where collisions cost nothing, with domain randomisation (lighting, textures, object positions) so the policy transfers to the real robot (sim-to-real), then fine-tune slowly on the real arm. Use a **safety layer** with hard limits on joint angle, speed and force and protected zones around humans, so unsafe commands never reach the arm and it can explore safely. **Penalise proximity** (from the cameras’ distances), not only impact, so it learns to keep clear before any collision. Start from **demonstrations or human feedback** and use a **curriculum** (one slow object, no obstacles, then add difficulty); share experience across robots or use logged data to cut real trials. Give at least two with the reason. \[1\]

**c.** (4 marks) Anomaly detection with 3,000 normal and 2 abnormal sequences.

**Do you agree?** **No, not as proposed.** Supervised two-class classification needs enough examples of both classes, and here the abnormal class has 2 of 3,002 (about 1 in 1,500). \[clear answer\]

**Challenge 1: extreme imbalance and almost no abnormal examples.** A multilayer perceptron could score 99.93% by always predicting "normal". Two samples cannot be split into training, validation and test sets, and they cannot represent the many ways a machine may fail (future anomalies may differ from these two). **Solution:** switch to anomaly detection trained on normal data only: an autoencoder on the 30-dimensional features (or the raw signal) with a reconstruction-error threshold (for example mean + 3 standard deviations of the normal errors), or a one-class or Gaussian model. Use the 2 abnormal sequences only to test the detector and sanity-check the threshold. If a classifier is still wanted, create more abnormal examples (time-warping, jittering, scaling) and judge with recall, precision and F1, not accuracy. \[1.5\]

**Challenge 2: the chosen features lose information (Figure 1).** The 10-bin histogram shows a clear difference only in the first bin (abnormal about 63 against normal about 19), which matches the raw trace: the abnormal signal sits flat at about 0 between roughly 2.0 and 2.7 s. But a histogram ignores the time order, so very different waveforms can give the same histogram, and the other bins overlap heavily. The frequency-domain features are nearly the same for the two signals except around features 3 and 4 (about 1.5 and 2.1 for abnormal against about 1.0 and 0.9 for normal), and most of the 20 values are near zero, so the discriminating information sits in very few dimensions. With only two abnormal cases we cannot tell which differences are reliable. **Solution:** keep the temporal structure with a 1D convolutional or LSTM autoencoder on the raw sequence; add time-frequency features that keep location (wavelet coefficients over time) and a run-length feature; use DTW or another shift-tolerant similarity (normal sequences also differ in phase); choose features with domain knowledge and validate them. \[1.5\]

**Optional third challenge:** the amplitude is normalised to 0 to 1 per sequence, which hides an absolute level change (for example a loose part vibrating harder). Keep the level as a separate feature. \[1\]

**d.** (4 marks) Bayesian diagnosis from Table 6 (9 records), evidence: Vibration = Fair, Acoustic = Low.

**Counts.** Carryback = records 1, 4, 6, 8, 9 (5 records). Mistracking = records 2, 3, 5, 7 (4 records). **Priors:** P(Carryback) = 5/9, P(Mistracking) = 4/9. \[1\]

**Likelihoods (count inside each cause).** Carryback: P(Fair \| Carryback) = 1/5 (record 8), P(Low \| Carryback) = 3/5 (records 6, 8, 9). Mistracking: P(Fair \| Mistracking) = 2/4 (records 2, 5), P(Low \| Mistracking) = 2/4 (records 5, 7). \[1\]

**Scores** (prior × likelihoods, naive Bayes). Carryback = 5/9 × 1/5 × 3/5 = 15/225 = **1/15 ≈ 0.067**. Mistracking = 4/9 × 2/4 × 2/4 = 4/36 = **1/9 ≈ 0.111**. \[1\]

**Decision: Mistracking.** The ratio is 1/9 : 1/15 = 5 : 3, so, normalising, P(Mistracking \| Fair, Low) = 5/8 = **62.5%** and P(Carryback \| Fair, Low) = 3/8 = 37.5%. **Assumption:** vibration and acoustic readings are independent given the cause (the naive assumption); it is only roughly true, because a mistracking belt can affect both readings through the same physical mechanism, so the probabilities may be overconfident, though the ranking is likely correct. No count is zero, so Laplace smoothing is not needed here (add 1 to each count if one were). \[1\]

## Summary of marks

| Question | Parts and marks                 |
|--------------|-------------------------------------|
| Q1 (20)      | a 4, b 3, c 3, d 4, e(i) 3, e(ii) 3 |
| Q2 (20)      | a 8, b 4, c 4, d 4                  |
