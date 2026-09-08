# Labour Market Intelligence

An end-to-end analytics and machine learning system for exploring labour market data, mapping skills, assessing skill gaps, predictive modelling, and explainability.

## Project Structure

```text
labour-market-intelligence/
│
├── data/
│   ├── raw/                # Original, immutable data dumps
│   └── processed/          # Cleaned and transformed datasets
│
├── src/
│   ├── data_exploration.py # EDA and statistical profiling
│   ├── preprocessing.py    # Data cleaning and text normalization
│   ├── skill_mapping.py    # Skill extraction & taxonomy mapping
│   ├── skill_gap.py        # Gap analysis between demand & candidate profiles
│   ├── modelling.py        # Predictive modelling & evaluation
│   └── explainability.py   # Model interpretability (e.g. SHAP, feature importance)
│
├── app.py                  # Main application entry point
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Setup Instructions

1. **Clone the repository and navigate to root**:
   ```bash
   cd labour-market-intelligence
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```
