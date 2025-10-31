import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Salary Predictor & Model Comparison", layout="wide")

st.title("💼 Salary Predictor and Model Comparison (with XGBoost & Feature Insights)")
st.write("Compare multiple regression models, visualize performance, and explore feature importance.")

# Sidebar controls
st.sidebar.header("⚙️ Data & Model Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset", value=True if uploaded_file is None else False)
test_size = st.sidebar.slider("Test set proportion", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42)
use_log_transform = st.sidebar.checkbox("Use log-transform for Salary (recommended)", value=False)
train_button = st.sidebar.button("Train & Compare Models")

# Sample data
@st.cache_data
def generate_sample_data(n=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    ages = rng.randint(21, 65, size=n)
    genders = rng.choice(["male", "female"], size=n)
    educations = rng.choice(["high school", "bachelor's degree", "master's degree", "PhD"], size=n)
    job_titles = rng.choice(["engineer", "analyst", "manager", "administrator", "data scientist", "sales"], size=n)
    years_exp = np.clip((ages - 21) * rng.rand(n), 0, 40)
    base = 30000
    salary = (base + ages * 300 + years_exp * 1200 +
              (educations == "master's degree") * 8000 + (educations == "PhD") * 15000 +
              (job_titles == 'manager') * 10000 + (job_titles == 'data scientist') * 15000 + rng.randn(n) * 5000)
    df = pd.DataFrame({
        'Age': ages,
        'Gender': genders,
        'Education Level': educations,
        'Job Title': job_titles,
        'Years of Experience': np.round(years_exp, 1),
        'Salary': np.round(salary, 2)
    })
    return df

# Load data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ CSV loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        df = None
elif use_sample:
    df = generate_sample_data(500, random_state)
else:
    df = None

if df is None:
    st.info("Upload a CSV or enable 'Use sample dataset' to proceed.")
    st.stop()

st.subheader("📄 Data Preview")
st.dataframe(df)

required_cols = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Salary']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

df.dropna(subset=required_cols, inplace=True)

# Features and target
X = df.drop(columns=['Salary'])
y_raw = df['Salary']

# Log-transform
if use_log_transform:
    y = np.log1p(y_raw).values.reshape(-1, 1)
else:
    y = y_raw.values.reshape(-1, 1)

salary_scaler = MinMaxScaler()
y_scaled = salary_scaler.fit_transform(y)

# Preprocessing
numeric_features = ['Age', 'Years of Experience']
categorical_features = ['Gender', 'Education Level', 'Job Title']

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
cat_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

# Models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=random_state),
    "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        objective='reg:squarederror'
    )
}

# Training
if train_button:
    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=test_size, random_state=random_state)
    results = []
    best_model = None
    best_r2 = -999

    st.info("⏳ Training models... Please wait.")

    for name, model in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipe.fit(X_train, y_train.ravel())
        y_pred_scaled = pipe.predict(X_test)

        # Reverse normalization & log
        y_pred = salary_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_test_inv = salary_scaler.inverse_transform(y_test.reshape(-1, 1))
        if use_log_transform:
            y_pred = np.expm1(y_pred)
            y_test_inv = np.expm1(y_test_inv)

        mae = mean_absolute_error(y_test_inv, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
        r2 = r2_score(y_test_inv, y_pred)

        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 3)
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model = pipe
            best_model_name = name

    results_df = pd.DataFrame(results)
    st.subheader("📈 Model Comparison Results")
    st.dataframe(results_df)

    # Visual comparison
    st.subheader("📊 Model Performance Visualization")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(results_df["Model"], results_df["R2"])
    axes[0].set_title("R² Score")
    axes[1].bar(results_df["Model"], results_df["MAE"])
    axes[1].set_title("Mean Absolute Error")
    axes[2].bar(results_df["Model"], results_df["RMSE"])
    axes[2].set_title("Root Mean Squared Error")
    for ax in axes:
        ax.set_xlabel("Model")
        ax.tick_params(axis='x', rotation=30)
    st.pyplot(fig)

    # Save best model
    joblib.dump({'pipeline': best_model, 'salary_scaler': salary_scaler, 'use_log': use_log_transform}, 'salary_model.joblib')
    st.success(f"🏆 Best Model: **{best_model_name}** (R²={best_r2:.3f}) saved successfully!")

    # 🔍 Feature importance (for tree-based models only)
    if best_model_name in ["RandomForest", "GradientBoosting", "XGBoost"]:
        st.subheader("🔍 Feature Importance (Best Model)")

        # Get feature names from preprocessor
        cat_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_features = np.concatenate([numeric_features, cat_names])

        regressor = best_model.named_steps['regressor']
        importances = getattr(regressor, "feature_importances_", None)

        if importances is not None:
            importance_df = pd.DataFrame({
                "Feature": all_features,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            st.dataframe(importance_df.head(15))

            # Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(importance_df["Feature"][:15][::-1], importance_df["Importance"][:15][::-1])
            ax.set_title("Top 15 Feature Importances")
            st.pyplot(fig)
        else:
            st.info("Feature importances not available for this model.")
    else:
        st.info("Feature importance only available for tree-based models.")

# Prediction section
st.sidebar.header("🔮 Predict Salary")
if os.path.exists('salary_model.joblib'):
    saved = joblib.load('salary_model.joblib')
    model_pipeline = saved['pipeline']
    salary_scaler = saved['salary_scaler']
    use_log_transform = saved.get('use_log', False)

    genders = sorted(df['Gender'].dropna().unique())
    educations = sorted(df['Education Level'].dropna().unique())
    jobs = sorted(df['Job Title'].dropna().unique())

    with st.sidebar.form("predict_form"):
        p_age = st.number_input('Age', min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=30)
        p_gender = st.selectbox('Gender', genders)
        p_education = st.selectbox('Education Level', educations)
        p_job = st.selectbox('Job Title', jobs)
        p_exp = st.number_input('Years of Experience', min_value=float(df['Years of Experience'].min()),
                                max_value=float(df['Years of Experience'].max()), value=5.0)
        submit_pred = st.form_submit_button('Predict')

    if submit_pred:
        try:
            sample = pd.DataFrame([{
                'Age': p_age,
                'Gender': p_gender,
                'Education Level': p_education,
                'Job Title': p_job,
                'Years of Experience': p_exp
            }])
            pred_scaled = model_pipeline.predict(sample)
            pred = salary_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
            if use_log_transform:
                pred = np.expm1(pred)
            st.sidebar.success(f'Predicted Salary: ${pred[0][0]:,.2f}')
        except Exception as e:
            st.sidebar.error(f'Prediction failed: {e}')
else:
    st.sidebar.info("Train models first to enable prediction.")

# Download dataset
csv = df.to_csv(index=False).encode('utf-8')
st.download_button('⬇️ Download Updated Dataset', csv, 'updated_dataset.csv', 'text/csv')
