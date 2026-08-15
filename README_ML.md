# DriftSense ML localization

Pipeline:
1. `train_ranker.py` trains a Siamese-style CNN matcher on samples 0000-0399.
2. `localize_ml.py` proposes candidates at multiple scales/rotations and lets the CNN rank them.
3. `evaluate_ml.py` evaluates the untouched samples 0450-0499.

Install:
    pip install -r requirements.txt

Train:
    python train_ranker.py

Test sample 0000:
    python localize_ml.py --reference dataset\sample_0000\reference.png --search dataset\sample_0000\search.png

Held-out evaluation:
    python evaluate_ml.py

Important:
- Do not use samples 0450-0499 for training/tuning.
- Do not report validation accuracy as localization accuracy.
- The final metric is pixel localization error on the held-out test set.
- No accuracy can be guaranteed until the code is run on the actual 500-pair dataset.
