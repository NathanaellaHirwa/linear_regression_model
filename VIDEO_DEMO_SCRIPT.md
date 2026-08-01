# Video Demo Script (≤ 7 minutes)

Recording checklist before you start:
- [ ] API is deployed on Render and the `/docs` URL is live and pasted into the main README
- [ ] `kApiBaseUrl` in `summative/FlutterApp/lib/main.dart` points at the deployed Render URL (not localhost)
- [ ] Screen sharing is on for the ENTIRE recording, and your camera is on the whole time
- [ ] Have the notebook, `prediction.py`, `main.dart`, and Swagger UI (`/docs`) all open in tabs/windows ready to switch to

Rule: this is a demo, not a retrospective. Don't narrate problems you hit while building — show the working thing and explain the results.

---

## 0:00–0:30 — Intro
- State your name, the mission ("predicting crop yield from country/crop/climate inputs to support food-security planning"), and that this is the Task 1–3 demo.

## 0:30–2:00 — Mobile app making a live prediction (do this FIRST, rubric wants it in the first ~2 min)
- Open the Flutter app on a device/emulator.
- Fill in the 6 fields with a real row from the dataset (e.g. Area=Rwanda, Item=Cassava, Year=2013, Rainfall=1212, Pesticides=157, Temp=19.39) — use exact values/spelling from `yield_df.csv` so it succeeds cleanly.
- Tap **Predict**, show the result appear in the display area.
- Quickly show ONE error case: clear a field or enter an out-of-range value (e.g. Year=1500), tap Predict, show the error message rendered in the app.

## 2:00–2:45 — Flutter code where the API is called
- Switch to `summative/FlutterApp/lib/main.dart`, scroll to `_predict()`.
- Point out: the POST to `$kApiBaseUrl/predict`, the JSON body matching the 6 Pydantic fields, and `_extractErrorMessage` surfacing FastAPI/Pydantic validation errors in the UI.

## 2:45–4:00 — Swagger UI tests of the deployed API
- Open the **public** Render `/docs` URL (not localhost — say the URL out loud or show the address bar).
- Expand `POST /predict`, "Try it out", submit a valid payload → show 200 response.
- Submit an invalid payload: wrong datatype (e.g. `"year": "abc"`) → show the 422 validation error.
- Submit an out-of-range value (e.g. negative rainfall) → show the 422 range-constraint error.
- Briefly show `POST /retrain` in Swagger (don't necessarily run it live if it's slow) — explain it accepts a CSV upload and retrains/overwrites the saved model.

## 4:00–5:30 — Model creation (show the notebook)
- Switch to `multivariate.ipynb`.
- Show the correlation heatmap + distributions cells and read your one-line interpretation of each.
- Show the 4-model comparison table/cell (`SGD_LinearRegression`, `OLS_LinearRegression`, `DecisionTree`, `RandomForest`) and the results:
  - RandomForest: MSE ≈ 2.99e8, R² ≈ 0.959 (best, saved)
  - DecisionTree: MSE ≈ 5.62e8, R² ≈ 0.922
  - OLS LinearRegression: MSE ≈ 1.82e9, R² ≈ 0.749
  - SGD LinearRegression: MSE ≈ 1.87e9, R² ≈ 0.742
- Show the SGD loss curve plot and the before/after best-fit scatter plot.

## 5:30–7:00 — Answer the 4 required questions
Speak to camera/screen for each — keep each answer to ~20–25 seconds:

1. **Is your loss high or low, and how would you reduce it further?**
   Loss is moderate-to-high for the linear models (R²≈0.74–0.75) but low for the saved RandomForest (R²≈0.96, MSE ~6x lower than linear). To reduce it further: engineer interaction features (crop × region), add more climate variables (soil quality, irrigation), or tune RandomForest depth/estimators further.

2. **Are there hyperparameters that could improve performance?**
   Yes — for RandomForest: `n_estimators`, `max_depth`, `min_samples_leaf` (currently 80/12/2, could grid-search). For SGD: `eta0`, `learning_rate` schedule, number of epochs. A `GridSearchCV`/`RandomizedSearchCV` pass over these would likely improve on the current defaults.

3. **What would you do if you had new data — how would you update the deployed model?**
   Use the `/retrain` endpoint: POST a CSV with the same schema (Area, Item, Year, hg/ha_yield, rainfall, pesticides, avg_temp) to `/retrain`, which retrains a fresh RandomForest, evaluates it on a held-out split, and overwrites `best_model.joblib` on disk — no redeploy needed, the running API picks up the new model in memory immediately.

4. **What was the basis for your CORS configuration?**
   Explicit origin allow-list instead of `*` because `/predict` and `/retrain` are POST endpoints that could be abused for scraping or spam if left fully open; restricted to `GET`/`POST` since there's no other mutable resource; `Content-Type`/`Authorization` headers only since those are the only ones the API reads; `allow_credentials=False` since there's no cookie/session auth in use.

## 7:00 — End
- Thank the viewer, stop recording. Trim any dead air before uploading.

---

**Upload to YouTube (unlisted or public), then paste the link into the main `README.md` under "Live Links".**
