# Move MoodJournal to an M2 Mac

The Windows experiment is complete: three additional epochs, selected additional epoch 3, test accuracy 90.30%, macro-F1 0.8636. All eight tests passed. The trained model can run immediately on the Mac; further training is not required to use the prototype.

## Transfer

Copy `transfer/MoodJournal-Mac.zip` to the Mac using OneDrive, a USB drive, or another transfer method you control. Extract it into a normal local folder such as `~/Projects/MoodJournal`. Keep the archive private: it includes a snapshot of your local journal database so your entries move with the project.

The package includes source code, dataset archives, research fixtures, original notes, evaluation reports, both trained transformer versions, and the full optimizer/scheduler/RNG checkpoint. It excludes the Windows virtual environment, downloaded Hugging Face cache, temporary files, and running-process logs/PIDs. `TRANSFER_MANIFEST.json` records file hashes.

## Set up and run

Use a native Apple-silicon Python 3.13 installation. In the Mac Terminal, change to the extracted project folder (adjust the first path to wherever you extracted it):

```bash
cd ~/Projects/MoodJournal
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mac.txt
python check_mac.py
python -m streamlit run app.py
```

Open http://localhost:8501. The model selector includes the continued DistilBERT checkpoint. Journal inference currently uses CPU; that is sufficient to use the app without additional training. `check_mac.py` separately checks whether PyTorch can access the M2 GPU for future training.

The Mac environment has not been tested from Windows. If a pinned package has no wheel for your Python/macOS combination, keep the installation error and resolve the environment before changing model serialization dependencies. The baseline was serialized with scikit-learn 1.9.1 and joblib 1.6.0; matching those versions avoids loading it with an incompatible runtime.

To verify functionality after setup:

```bash
python -m unittest discover -s tests -v
```

## Further training

Do **not** run `continue_transformer.py --resume` expecting another epoch: the existing experiment is complete and intentionally protected against overwriting. The `.ps1` scripts are Windows helpers, not Mac launchers.

Before another experiment, adapt the trainer for `mps`, move model/input tensors and restored optimizer tensors to the selected device, and create a separate output directory with a newly declared epoch limit. Preserve the existing reports. The transfer includes `artifacts/distilbert_continued/training_checkpoint.pt` for continuing optimizer/scheduler state, and `artifacts/distilbert_continued/best/` for inference or weight-only initialization.

Cross-device continuation is not expected to be bitwise identical to CPU training. MPS needs a supported macOS/PyTorch combination; check it on the actual Mac. No speedup or memory fit is assumed until measured. Only load this project's trusted `.pt`/`.joblib` files.

Official MPS setup: https://developer.apple.com/metal/pytorch/
