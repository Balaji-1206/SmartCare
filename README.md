# SmartCare

SmartCare is a healthcare operations project that combines:
- A **FastAPI backend** for predictions and operational endpoints
- A **React Native (Expo) mobile app** for frontline workflows
- **ML training pipelines** for patient volume, demand, and syndrome forecasting
- **ETL/data utilities** for preprocessing and feature engineering

## Repository Structure

- `/api` – FastAPI app and service logic
- `/smartcare-mobile` – Expo/React Native mobile client
- `/ml` – model training code and model artifacts
- `/etl` – data preparation scripts
- `/data/raw` – input and persisted JSON/CSV data
- `/tests` – test placeholders
- `/docs` – design/workflow placeholders

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm
- (Optional) Docker and Docker Compose

## Setup

### 1) Backend dependencies

```bash
pip install -r requirements.txt
```

### 2) Mobile dependencies

```bash
cd smartcare-mobile
npm install
```

### 3) Optional local services (Postgres + MLflow)

```bash
docker-compose up -d
```

## Run the Backend API

From repository root:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: `http://127.0.0.1:8000/docs`

### Optional Environment Variable

- `OPENWEATHER_API_KEY` – required only for `POST /weather/fetch`

You can place this in a `.env` file at repository root.

## Run the Mobile App

From `/smartcare-mobile`:

```bash
npm start
```

Useful scripts:
- `npm run android`
- `npm run ios`
- `npm run web`

The mobile API base URL defaults to:
- Android emulator: `http://10.0.2.2:8000`
- iOS/Web: `http://127.0.0.1:8000`

## Key API Endpoints

### Predictions
- `POST /predict/volume`
- `POST /predict/demand`
- `POST /predict/syndromes`

### Mobile Aggregation
- `GET /mobile/today`
- `GET /alerts`

### Nurse Log
- `POST /nurse/log`
- `GET /nurse/log/{date}`

### Inventory
- `GET /inventory`
- `POST /inventory/upsert`

### Weather
- `POST /weather/upsert`
- `GET /weather/today`
- `POST /weather/fetch`

## Data and Artifacts

The API expects:
- `data/raw/data10yrs.csv`
- Model artifacts under `ml/artifacts/` (volume, demand, syndromes)

These are already present in this repository and are loaded at runtime.
