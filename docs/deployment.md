# Deployment and operations

## Local development

Install Python 3.11 or newer. From the repository root, create and activate a
virtual environment (PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.
Then install the app and run it:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

The server binds to localhost so it is not directly exposed to the local
network. The model file and application code must remain together in the
repository layout.

## Temporary public access with ngrok

Each host needs ngrok installed and an ngrok account/authtoken of their own.
Install ngrok from <https://ngrok.com/download>. On Windows with winget:

```powershell
winget install --id Ngrok.Ngrok --exact
```

Open a new terminal after installation, configure the token privately, and
start the tunnel in a second terminal after Streamlit is running:

```sh
ngrok config add-authtoken YOUR_NGROK_AUTHTOKEN
ngrok http 8501
```

Share only the HTTPS forwarding URL. Do not put tokens in source control,
screenshots, or shared command history. Stop the tunnel with `Ctrl+C`; the
temporary URL can change each time a tunnel starts.

## Operational and safety limits

- The app is an educational prototype, not a clinical diagnostic service.
- Never submit real patient or other sensitive health information.
- The app does not persist form inputs, but a public tunnel permits anyone with
  its URL to access the running app.
- There is no authentication, rate limiting, audit trail, monitoring, or
  clinical validation. Do not deploy it for real patient care.
- Only load the checked-in model from a trusted repository; joblib artifacts
  use pickle serialization and can execute code when loaded.
- Pin and review dependency updates before deployment. The scikit-learn
  dependency is pinned to the version used to serialize the model.
