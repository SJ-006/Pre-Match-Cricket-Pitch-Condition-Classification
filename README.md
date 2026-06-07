# Pre-Match Cricket Pitch Condition Classification

Predict whether an Indian cricket pitch is likely to be batting-friendly, pace-friendly, or spin-friendly before the match starts.

## Dataset

The project runs immediately with a synthetic India-focused dataset. When you get real files, add them here:

```text
data/raw/
```

Accepted raw inputs:

- Cricsheet/IPL `.json` files
- Cricsheet/IPL `.zip` archives containing JSON files

Recommended files to add later:

- IPL JSON
- T20I JSON
- ODI JSON

The pipeline filters matches to configured Indian venues in `src/config.py`, extracts match-level signals, and combines real rows with the synthetic training dataset.

## Run

```bash
python run_pipeline.py
```

This generates:

- `data/cricket_pitch_dataset.csv`
- `models/preprocessor.pkl`
- `models/rf_model.pkl`
- `reports/metrics.json`
- plots in `reports/figures/`

## App

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```text
data/
  raw/
  cricket_pitch_dataset.csv
models/
  preprocessor.pkl
  rf_model.pkl
src/
  config.py
  generate_data.py
  features.py
  preprocess.py
  train.py
  evaluate.py
  explain.py
app/
  streamlit_app.py
reports/
  figures/
notebooks/
run_pipeline.py
requirements.txt
README.md
```
