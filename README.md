# AIE25008_skindisease

Lightweight skin lesion classification (HAM10000, 7 classes) and acne severity grading (ACNE04, 4 grades) with
attention, Monte Carlo dropout uncertainty, uncertainty-based specialist referral, class activation maps and
verifier-constrained LLM reporting.

**Live demo:** https://ksharsha2810.github.io/AIE25008_skindisease/

The demo runs the deployed model entirely in the browser (ONNX Runtime Web); uploaded images never leave the device.

> Research prototype only. It is not a medical device and must not be used for diagnosis or treatment.

## Pipeline

1. **Data integrity.** SHA-256 de-duplication, removal of identical images with conflicting labels, and 60/20/20
   `StratifiedGroupKFold` splits grouped by lesion (HAM10000), patient (PAD-UFES-20) or image ID (ACNE04).
2. **Backbone selection.** Frozen ImageNet-pretrained MobileNetV3-Large, MobileNetV4-Conv-Small and PP-LCNet-1.0
   are compared on HAM10000 and PAD-UFES-20 with linear probes. PP-LCNet-1.0 is selected with a validation score
   (0.70 accuracy, 0.15 parameter efficiency, 0.15 speed).
3. **Dual-head model.** PP-LCNet-1.0 trunk -> CBAM -> global average pooling -> dropout (0.3) -> a 7-class lesion
   head (HAM10000) and a 4-grade acne head (ACNE04, trained afterwards with everything else frozen).
4. **Uncertainty.** 30-pass MC dropout; cases with normalised predictive entropy above 0.4 are flagged for review.
5. **Explanation.** Grad-CAM on the last backbone block.
6. **Reporting.** A schema-validated JSON payload is formatted by Gemini; the output is accepted only if it matches
   the payload exactly (no image is ever sent to the LLM).

## Results (held-out test splits)

| Task | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| HAM10000 lesions (2,003 images) | 78.18% | 51.40% | ECE 0.023, macro AUC 0.932 |
| ACNE04 acne grade (284 images) | 60.56% | 51.68% | Quadratic kappa 0.649 |
| Referral rule, HAM10000 | refers 38.1% | – | catches 81.3% of errors; 93.3% accuracy on retained cases |

About 1.04 M executed parameters, 0.15 GMACs and ~15 ms per image on a laptop CPU.

## Repository layout

```
notebook/     skin_disease_pipeline_consolidated.ipynb   end-to-end pipeline (setup, data, training, explain, report)
src/          skin_pipeline_runtime.py                   pipeline implementation used by the notebook
experiments/  evaluation scripts (backbone benchmark, head/attention ablation, cross-dataset transfer,
              calibration and selective prediction, robustness, explanations, complexity, report verifier)
demo/         export_onnx.py                             exports the trained model for the web demo
docs/         static web demo served by GitHub Pages
```

## Running

```bash
pip install -r requirements.txt
```

1. Open `notebook/skin_disease_pipeline_consolidated.ipynb` and set the project/artifact paths in the first cell.
2. Provide credentials through environment variables or the notebook's hidden prompt, never in code:
   `KAGGLE_USERNAME`, `KAGGLE_KEY` (or a local `kaggle.json`) for dataset download and `GEMINI_API_KEY` for reporting.
3. Run the cells in order. Datasets are downloaded from Kaggle (`kmader/skin-cancer-mnist-ham10000`,
   `mahdavi1202/skin-cancer`, `jincyjis/acne04`); checkpoints and metrics are written to the artifact folder.

Evaluation scripts (`experiments/`) read the artifacts produced by the notebook. Set `SKIN_PROJECT_ROOT` to the
folder that contains `skin_pipeline_artifacts/`, then for example:

```bash
cd experiments
python exp4_uncertainty.py        # calibration, MC dropout, risk-coverage, referral analysis
python exp3_cross_dataset.py      # label-compatible HAM10000 <-> PAD-UFES-20 transfer
python exp8_reporting.py offline  # verifier fault-injection test
```

Rebuild the web demo model from a trained checkpoint:

```bash
python demo/export_onnx.py --checkpoint <artifacts>/pretrained_candidates/full/dual_final.pt --out docs/model
```

## Datasets

- HAM10000: Tschandl, Rosendahl and Kittler, *Scientific Data* 5, 180161 (2018). CC BY-NC 4.0.
- PAD-UFES-20: Pacheco et al., *Data in Brief* 32, 106221 (2020).
- ACNE04: Wu et al., ICCV 2019.

The demo includes five HAM10000 test images for illustration under the dataset's non-commercial licence.

## Authors

Jayakrishnan Anandakrishnan and K S Harsha Vardhan
