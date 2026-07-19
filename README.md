Bank Term Deposit Subscription Prediction & Call-Targeting Pipeline

Predict which bank clients are likely to subscribe to a term deposit, and use that
prediction to cut down the number of telemarketing calls an agent has to make while
still reaching most of the actual subscribers.

Overview

Banks running telemarketing campaigns typically call every client in their database,
which is expensive in agent time and often low-yield (this dataset's true subscription
rate is only ~11.7%). This project builds a two-stage machine learning pipeline that
scores clients before any call is made, so outreach can be targeted at the clients
most likely to convert — and simulates the best way to reach each of them.

Dataset: UCI Bank Marketing dataset
(bank-full.csv), ~45,000 rows, binary target y (subscribed: yes/no).

Approach


EDA — understand the target imbalance and how client attributes relate to
subscription likelihood (age, job, education, month/day of contact, etc.).
Feature engineering — bucket day into beginning/middle/end-of-month, bucket
campaign (contact count) into groups, encode categoricals, and apply a
Yeo-Johnson transform to skewed numeric features (age, balance).
Two-stage modelling, rather than one model on every feature at once:

Stage 1 ("pre-call" gate) — AdaBoost, trained on only the static client
attributes known before any contact decision is made (age, job, balance,
education, etc.). Used purely to filter out clients unlikely to be worth
contacting at all.
Stage 2 ("campaign" scorer) — LightGBM, trained on static features plus
campaign context (contact channel, month, day-of-month bucket, previous-campaign
outcome). Only runs on clients that clear the Stage 1 gate, and its own
probability output is the final ranking score used to decide who to call.



Class imbalance is handled via combined oversampling (minority class) and
undersampling (majority class) on the training data only, with the Yeo-Johnson
transform fit strictly before resampling to avoid contaminating its statistics
with duplicated/dropped rows.
Threshold selection — the operating decision threshold is chosen via the
precision/recall crossover on a dedicated validation set, kept fully separate
from the test set used for final reporting, so the reported numbers aren't
optimistically biased by tuning and evaluating on the same data.
Campaign simulation — for clients who clear Stage 1, the pipeline simulates
every combination of contact channel and day-of-month bucket to recommend the
outreach strategy the model scores most favourably for that client.
Business impact analysis — translates model performance into calls saved,
agent time saved, and subscribers still reached, versus a call-everyone baseline.


Results

Evaluated on a held-out test set, using a threshold selected on a separate validation
set:

MetricValueThreshold (validation-selected)0.62Test precision0.427Test recall0.421Baseline duration per subscription2195.7sModel duration per subscription662.6sTime efficiency improvement3.31x
