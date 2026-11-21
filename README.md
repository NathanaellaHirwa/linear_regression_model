# Predicting garment-factory team productivity

**Mission:** support industrial growth and decent work in Africa (UN SDG 8 and 9) by helping labour-intensive garment factories hit their daily targets.
**Problem:** supervisors learn that a team missed its target only at the end of the day, when it is too late to add workers, overtime or incentives.
**Solution:** predict a team's actual productivity (1.0 = 100% of planned output) from 12 facts known before the shift, and flag teams likely to fall short of their target.
**Users:** production planners and line supervisors, through a one-page mobile app backed by a public API.

## Dataset

* **Productivity Prediction of Garment Employees**, UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees> (Imran et al., 2019) - real production records of a garment factory, one row per team per working day (Jan-Mar 2015).
* **1,197 rows, 15 columns** (9 numeric such as target productivity, SMV, work in progress, overtime, incentive, workers; categorical: department, day, team, week of month). The file is `summative/linear_regression/garment_productivity.csv`.
* **Target:** `actual_productivity` (about 0.23 to 1.12). Cleaning: `sweing` typo fixed, `quarter` converted to `week_of_month`, missing `wip` (only the finishing department, which has no work-in-progress queue) set to 0, `date` and the redundant `idle_time` dropped. The model uses 12 inputs chosen by permutation importance: `department, day, team, week_of_month, targeted_productivity, smv, wip, over_time, incentive, idle_men, no_of_style_change, no_of_workers`.

## Public API (Swagger UI)

**<https://garment-productivity-api.onrender.com/docs>**

Routes: `POST /predict`, `POST /retrain` (CSV upload), `GET /docs`. A free Render service sleeps after 15 minutes without traffic; the first request can take up to a minute.

## Video demo (YouTube)

**<https://www.youtube.com/watch?v=YOUR-VIDEO-ID>**  <!-- TODO: replace with your real YouTube link before submitting -->

## Run the mobile app (Windows, Android emulator or phone)

1. Install [Flutter](https://docs.flutter.dev/get-started/install/windows/mobile) and Android Studio (Android SDK + an emulator). Run `flutter doctor` and fix anything marked with an X; accept licences with `flutter doctor --android-licenses`.
2. Clone this repository and open PowerShell in the app folder:
   ```powershell
   cd linear_regression_model\summative\FlutterApp
   ```
3. Point the app at your API. Either edit `kApiBaseUrl` at the top of `lib\main.dart`, or pass it on the command line in step 5 (`--dart-define`).
4. Start an emulator (Android Studio -> Device Manager -> Play) or plug in an Android phone with USB debugging, then check it is listed: `flutter devices`.
5. Install packages and run:
   ```powershell
   flutter pub get
   flutter run --dart-define=API_BASE_URL=https://garment-productivity-api.onrender.com
   ```
6. Fill in the 12 fields (the hint under each field shows the allowed values), press **Predict**, and read the result in the card below the button. Example: `sewing`, `Tuesday`, `1`, `2`, `0.75`, `22.5`, `900`, `5000`, `0`, `0`, `0`, `40`. Leave a field empty, type text in a number field, or enter an out-of-range value to see the error messages.

To use an API running on your own PC from the emulator, start it with `uv run uvicorn prediction:app` (see below) and use `--dart-define=API_BASE_URL=http://10.0.2.2:8000`.

## Results (test set, 240 team-days)

| Model | Train MSE | Test MSE | Test R² |
|---|---|---|---|
| **RandomForestRegressor (saved)** | 0.0039 | **0.0125** | 0.53 |
| DecisionTreeRegressor | 0.0132 | 0.0169 | 0.36 |
| SGDRegressor (gradient descent, tuned) | 0.0210 | 0.0221 | 0.17 |
| LinearRegression (closed form) | 0.0208 | 0.0226 | 0.15 |

Predicting the training mean for every team-day gives a test MSE of 0.0267, so the forest cuts the error by about half. The non-linear models win clearly because productivity depends on thresholds and interactions that a straight line cannot capture.

## Repository layout

```
linear_regression_model/
├── README.md
└── summative/
    ├── pyproject.toml, uv.lock          # notebook environment (uv)
    ├── linear_regression/
    │   ├── multivariate.ipynb           # EDA, 4 models, loss curve, comparison, saves the best model
    │   └── garment_productivity.csv
    ├── API/
    │   ├── prediction.py                # FastAPI app: /predict, /retrain, /docs + CORS
    │   ├── retrain.py                   # CSV validation + retraining logic
    │   ├── best_model.pkl               # saved Pipeline (preprocessing + best model)
    │   ├── pyproject.toml, uv.lock, requirements.txt   # API environment (Render uses requirements.txt)
    │   ├── training_data.csv            # original training split, used when /retrain refits the model
    │   └── sample_retrain.csv           # ready-made CSV to demo /retrain
    └── FlutterApp/                      # one-page Flutter app
```

## Run the API and the notebook locally (Windows, PowerShell)

```powershell
# install uv once:  winget install --id=astral-sh.uv -e      (then open a NEW terminal)
cd linear_regression_model\summative\API
uv sync
uv run uvicorn prediction:app --reload        # http://127.0.0.1:8000/docs

cd ..\                                         # summative\
uv sync
uv run jupyter lab                             # open linear_regression\multivariate.ipynb
```
