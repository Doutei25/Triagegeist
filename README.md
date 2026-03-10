# Triagegeist
Triagegeist: LightGBM-Powered Emergency Triage Acuity Prediction
Clinical Problem Statement
Every minute counts in the emergency department. Triage nurses and physicians must rapidly assign acuity scores using the Emergency Severity Index (ESI, levels 1–5) under extreme cognitive load, with incomplete information, and in chronically understaffed environments. Inter-rater variability in triage scoring is well-documented, and systematic undertriage of certain patient populations remains an active patient safety concern.
This project addresses a specific clinical question: can a machine learning model accurately predict ESI triage acuity from structured intake data and free-text chief complaints? A reliable predictive model could serve as a real-time decision support layer — flagging potentially undertriaged patients, reducing variability, and extending the capacity of overburdened triage teams.

What We Built
A full end-to-end ML pipeline combining structured clinical data, comorbidity history, and free-text chief complaint NLP to predict ESI triage acuity levels 1–5. The system is built on LightGBM with 5-fold stratified cross-validation and achieves a CV Macro F1 of 0.9867 across all five acuity classes.

Dataset
We used all four files from the Triagegeist dataset joined on patient_id:
FileRowsDescriptiontrain.csv80,000Structured vitals, demographics, context + targettest.csv20,000Same features, no labelchief_complaints.csv100,000Free-text nurse-entered triage complaintspatient_history.csv100,00025 binary comorbidity flags

Methodology
1. Data Merging
All four files were merged on patient_id with care taken to avoid duplicate columns. The chief_complaint_system column already present in train.csv was retained, and only chief_complaint_raw was pulled from the complaints file.
2. Missing Value Handling
Missingness was treated as clinically meaningful rather than noise. Indicator flags were created before imputation, preserving the signal that lower-acuity patients are less likely to have certain vitals recorded. Missing values were filled using training set medians only — no test data leakage. Pain scores encoded as -1 were converted to NaN prior to processing.
3. Feature Engineering
Eleven binary clinical risk flags were engineered from raw vitals using established emergency medicine thresholds:
FlagThresholdHypoxiaSpO2 < 94%TachycardiaHR > 100 bpmHypotensionSBP < 90 mmHgFeverTemp > 38.3°CAltered ConsciousnessGCS < 15TachypneaRR > 20 breaths/minHigh PainPain score ≥ 8High Risk ArrivalAmbulance or helicopterNight ShiftOvernight presentationElderlyAge ≥ 65PediatricAge < 18
A total comorbidity burden score was computed by summing all 25 binary history flags. Patients with ESI 1 had a mean burden of 6.62 vs 5.02 for ESI 5 — a clinically meaningful difference.
4. NLP Pipeline
Free-text chief complaints were processed using TF-IDF (300 features, unigrams and bigrams) followed by Truncated SVD (Latent Semantic Analysis) compressed to 30 dense components explaining 56% of text variance. Top terms for ESI 1 patients included: severe, acute, worsening, diaphoresis, fever — exactly what emergency clinicians expect to flag critical presentations.
5. Model Training
LightGBM multiclass classifier trained with 5-fold stratified cross-validation. Inverse-frequency class weights were applied to handle the imbalanced target distribution (ESI 3 = 36%, ESI 1 = 4%). Early stopping with patience of 50 rounds was used throughout. Final predictions were averaged across all five folds.
Final feature matrix: 110 features spanning vitals, engineered clinical flags, comorbidity burden, demographics, and NLP components.

Results
FoldMacro F1Fold 10.9856Fold 20.9855Fold 30.9881Fold 40.9870Fold 50.9871Mean0.9867 ± 0.0010
Per-class recall (out-of-fold):
ESI LevelRecall1 — Critical96.3%2 — Emergent98.7%3 — Urgent99.7%4 — Less Urgent99.3%5 — Non-Urgent99.1%
Critically, all misclassifications were adjacent-class errors — ESI 1 patients predicted as ESI 2, never as ESI 4 or 5. A triage support tool that never confuses a critical patient with a minor one is the most important clinical property this system could have.

Top Predictive Features
The model's top features by information gain mirror real emergency medicine clinical priorities:

GCS Total — consciousness level is the single strongest predictor
NEWS2 Score — composite early warning score used in real EDs
Pain Score — self-reported severity
SpO2 — peripheral oxygen saturation
Chief Complaint NLP — five SVD text components in the top 10

The strong showing of NLP features validates the value of processing free-text complaints alongside structured vitals — something current triage systems do not do algorithmically.

Limitations

Synthetic data — the model has not been validated on real patient data; real-world performance may differ due to transcription noise and population shifts
No temporal modelling — patient deterioration while waiting was not captured
No SHAP explanations — per-patient explanations would be required for safe clinical deployment
Demographic bias not audited — subgroup performance across age, sex, and language groups was not formally evaluated


Reproducibility
All code is in the attached Kaggle notebook. The pipeline runs end-to-end without errors on Google Colab (CPU, approximately 5–10 minutes). Random seed fixed at 42 throughout. Dependencies: pandas, numpy, scikit-learn, lightgbm, matplotlib, seaborn.
