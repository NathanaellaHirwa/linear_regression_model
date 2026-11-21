# Garment Team Productivity Prediction — ML Summative (Linear Regression → API → Flutter)

## Mission & Problem

Supervisors in labour-intensive garment factories learn that a team missed its target only at the end of the day, when it is too late to add workers, overtime or incentives. This project predicts a team's actual productivity (1.0 = 100% of planned output) from 12 facts known before the shift, so planners can flag teams likely to fall short of their target. It supports industrial growth and decent work in Africa (UN SDG 8 and 9).

**Dataset:** *Productivity Prediction of Garment Employees*, UCI Machine Learning Repository (Imran et al., 2019): <https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees>. 1,197 rows, 15 columns, one row per team per working day (Jan–Mar 2015). File: `garment_productivity.csv`.

## Live Links

- **API (Swagger UI):** https://garment-productivity-api.onrender.com/docs
- **YouTube demo video:** https://youtu.be/D4m5DS8SbmI

## Repo Structure

```
linear_regression_model/
├── summative/
│   ├── linear_regression/
│   │   ├── multivariate.ipynb       # EDA, 4-model comparison, loss curve, saves best model
│   │   └── garment_productivity.csv # dataset
│   ├── API/
│   │   ├── prediction.py            # FastAPI app (/predict, /retrain, /docs + CORS)
│   │   ├── retrain.py               # CSV validation + retraining logic
│   │   ├── best_model.pkl           # saved Pipeline (preprocessing + RandomForest)
│   │   ├── training_data.csv        # original training split, used when /retrain refits
│   │   ├── sample_retrain.csv       # ready-made CSV to demo /retrain
│   │   ├── pyproject.toml, uv.lock
│   │   └── requirements.txt         # used by Render
│   ├── FlutterApp/
│   │   ├── lib/main.dart            # single-page prediction app
│   │   └── pubspec.yaml
│   ├── pyproject.toml
│   └── uv.lock
└── README.md
```

## Task 1 — Model

Open `summative/linear_regression/multivariate.ipynb`. It:

- Loads and cleans the dataset (`sweing` typo fixed, `quarter` converted to `week_of_month`, missing `wip` for the finishing department set to 0, `date` and redundant `idle_time` dropped)
- Visualizes correlations and distributions with interpretation
- Selects 12 inputs by permutation importance: `department, day, team, week_of_month, targeted_productivity, smv, wip, over_time, incentive, idle_men, no_of_style_change, no_of_workers`
- Trains and compares 4 regressors: `SGDRegressor` (gradient descent), OLS `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`
- Plots the SGD train/test loss curve and a before/after best-fit scatter plot
- Saves the best model (RandomForest) to `best_model.pkl`

Results on the test set (240 team-days):

| Model | Train MSE | Test MSE | Test R² |
|---|---|---|---|
| **RandomForestRegressor (saved)** | 0.0039 | **0.0125** | 0.53 |
| DecisionTreeRegressor | 0.0132 | 0.0169 | 0.36 |
| SGDRegressor (tuned) | 0.0210 | 0.0221 | 0.17 |
| LinearRegression (closed form) | 0.0208 | 0.0226 | 0.15 |

Predicting the training mean for every team-day gives a test MSE of 0.0267, so the forest cuts the error by about half. Non-linear models win because productivity depends on thresholds and interactions a straight line cannot capture.

To run the notebook (Windows, PowerShell; install uv once with `winget install --id=astral-sh.uv -e`):

```powershell
cd linear_regression_model\summative
uv sync
uv run jupyter lab      # open linear_regression\multivariate.ipynb
```

## Task 2 — API

```powershell
cd linear_regression_model\summative\API
uv sync                 # or: pip install -r requirements.txt
uv run uvicorn prediction:app --reload
```

Visit http://127.0.0.1:8000/docs for Swagger UI. Endpoints:

- `GET /` — service status
- `POST /predict` — body: the 12 inputs above (strict types and realistic ranges are enforced); returns the predicted productivity
- `POST /retrain` — multipart CSV upload (same schema as the training data, max 5 MB); the file is validated and the model is refit. Use `sample_retrain.csv` to try it.

**CORS:** explicit origin allow-list, GET/POST/OPTIONS only, `Content-Type`/`Accept` headers only, credentials disabled — see comments in `prediction.py` for the reasoning.

### Deploying to Render (free tier)

On render.com → **New → Web Service**, connect the GitHub repo and set:

- Root directory: `summative/API`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn prediction:app --host 0.0.0.0 --port $PORT`

A free Render service sleeps after 15 minutes without traffic; the first request can take up to a minute.

## Task 3 — Flutter App

Single page: 12 input fields (the hint under each shows the allowed values), a **Predict** button, and a result/error display area.

1. Install [Flutter](https://docs.flutter.dev/get-started/install/windows/mobile) and Android Studio, then run `flutter doctor`.
2. Start an Android emulator or plug in a phone, and check it is listed with `flutter devices`.
3. Run:

```powershell
cd linear_regression_model\summative\FlutterApp
flutter pub get
flutter run --dart-define=API_BASE_URL=https://garment-productivity-api.onrender.com
```

Example input: `sewing`, `Tuesday`, `1`, `2`, `0.75`, `22.5`, `900`, `5000`, `0`, `0`, `0`, `40`. Leave a field empty or enter an out-of-range value to see the error messages.

To use an API running on your own PC from the emulator, use `--dart-define=API_BASE_URL=http://10.0.2.2:8000`.

## Task 4 — Video Demo (≤7 min)

YouTube demo video: https://youtu.be/D4m5DS8SbmI
