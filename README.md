# MediCare Diagnostics: Heart Disease Risk Screening

An educational Streamlit demonstration that applies a trained UCI heart-disease
classification pipeline to user-entered sample values.

> **Not for clinical use.** This project is not a medical device and must not be
> used to diagnose, treat, or guide healthcare decisions. Do not enter real
> patient information. Predictions are estimates from a small educational dataset.

## Quick start

Requires Python 3.11 or newer. From the repository root, create and activate a
virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux, activate it with `source .venv/bin/activate`. Then install the
app and start it:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open <http://localhost:8501>. Stop the app with `Ctrl+C`.

To run the automated tests, install the development extra and run pytest:

```sh
python -m pip install -e ".[dev]"
python -m pytest
```

## Public demo link with ngrok

The app binds to localhost by default. To share it temporarily, install ngrok on
your own computer, sign in to your own ngrok account, and configure your own
authtoken. On Windows with winget:

```powershell
winget install --id Ngrok.Ngrok --exact
```

Open a new terminal after installation, then add your account token. Never share
or commit the token:

```sh
ngrok config add-authtoken YOUR_NGROK_AUTHTOKEN
```

Keep Streamlit running in one terminal. In another terminal, run:

```sh
ngrok http 8501
```

Share the HTTPS forwarding URL ngrok prints. The URL is temporary and may change
when the tunnel restarts. Anyone with the URL can access the app while both
processes are running. See [deployment notes](docs/deployment.md).

## Repository layout

```text
app/                    Streamlit user interface
src/medicare_app/       Prediction and input-validation package
models/                 Versioned trained model and metadata
tests/                  Automated model and input validation tests
docs/                   Deployment and operational notes
research/data/          Source datasets used in the project
research/notebooks/     Original assignment and analysis notebooks
research/outputs/       Generated figures
research/reports/       Analysis, audit, and modeling reports
```

## Model and input details

The app loads the repository's trusted model artifact at
`models/heart_disease_final_model_v1.pkl`. Its preprocessing pipeline handles
missing values and categorical encoding. The prediction module validates the
13 expected input fields and category codes before calling the model.

Only load model artifacts from a trusted source: joblib model files are
Python-pickle based and can execute code when loaded.

The project is a student ML demonstration, not a production clinical service.
It has no authentication, patient-data persistence, or clinical validation.
