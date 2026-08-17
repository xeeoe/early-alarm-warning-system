"""
Early Alarm Warning System
==========================
واجهة Streamlit بثلاثة أقسام مستقلة:
1) Training Data Analysis: تحليل بيانات التدريب واختيار الأعمدة والرسومات.
2) New Data Prediction: رفع أو تحميل بيانات جديدة وتجهيزها ثم التنبؤ عليها.
3) Ready Prediction Viewer: عرض ملف crisis_predictions_full.csv الجاهز فقط.

مهم:
- لا يتم عرض global_crisis_new.csv كجدول رئيسي للتوقعات.
- ملف التوقعات الجاهز المستخدم افتراضيًا هو crisis_predictions_full.csv.
- ملف economic_indicators يستخدم فقط في قسم التنبؤ على البيانات الجديدة.
- الملفات يتم تحميلها مرة واحدة باستخدام Streamlit cache.
"""

from pathlib import Path
import warnings
import json
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =============================================================================
# 1) المسارات الأساسية للمشروع
# =============================================================================
# المسارات المحلية الأصلية على جهاز Windows.
LOCAL_MODEL_PATH = Path(r"D:\project\code\code finale\best model\final model best\xgboost model best\best_model_xgb_optimization.joblib")
LOCAL_TRAIN_DATA_PATH = Path(r"D:\project\data sets\econmic crisis data for train\final data set\global_crisis_new.csv")
LOCAL_PREDICTIONS_PATH = Path(r"D:\project\data sets\predicted results csv\new_best\crisis_predictions_full.csv")
LOCAL_ECONOMIC_DATA_PATH = Path(r"D:\project\data sets\DATA FOR PREDICT\economic_indicators_2000_2025_with_fedfunds.csv")

# عند النشر على Streamlit Cloud لا يوجد قرص D:، لذلك نستخدم الملفات
# الموجودة بجانب ملف التطبيق داخل مستودع GitHub كمسارات بديلة.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = LOCAL_MODEL_PATH if LOCAL_MODEL_PATH.exists() else BASE_DIR / "best_model_xgb_optimization.joblib"
TRAIN_DATA_PATH = LOCAL_TRAIN_DATA_PATH if LOCAL_TRAIN_DATA_PATH.exists() else BASE_DIR / "global_crisis_new.csv"
PREDICTIONS_PATH = LOCAL_PREDICTIONS_PATH if LOCAL_PREDICTIONS_PATH.exists() else BASE_DIR / "crisis_predictions_full.csv"
ECONOMIC_DATA_PATH = LOCAL_ECONOMIC_DATA_PATH if LOCAL_ECONOMIC_DATA_PATH.exists() else BASE_DIR / "economic_indicators_2000_2025_with_fedfunds.csv"

TARGET = "crisis_next_12m"
DEFAULT_THRESHOLD = 0.50

st.set_page_config(
    page_title="Early Alarm Warning System",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 2) CSS بسيط للألوان والحالات
# =============================================================================
st.markdown(
    """
    <style>
    .main-title {font-size: 2.2rem; font-weight: 700; color: #12355B;}
    .sub-title {color: #536878; font-size: 1.05rem;}
    .status-crisis {background:#ffe5e5; color:#a40000; padding:14px; border-radius:10px; font-weight:700;}
    .status-safe {background:#e6f7ed; color:#087f3e; padding:14px; border-radius:10px; font-weight:700;}
    .status-warning {background:#fff5d6; color:#8a5a00; padding:14px; border-radius:10px; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 3) التحميل بالكاش: لا يعاد تحميل الملفات عند كل refresh أو تغيير اختيار
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """تحميل الموديل مرة واحدة فقط داخل جلسة Streamlit."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_csv_from_path(path_string: str) -> pd.DataFrame:
    """تحميل CSV من المسار مع cache_data."""
    return pd.read_csv(path_string)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """قراءة الملف المرفوع من bytes حتى يعمل الكاش بشكل مستقر."""
    return pd.read_csv(pd.io.common.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def extract_analysis_code(file_name: str, file_bytes: bytes) -> dict:
    """قراءة ملف التحليل كبيانات نصية فقط دون تنفيذ أي كود مرفوع."""
    text = file_bytes.decode("utf-8", errors="replace")
    if file_name.lower().endswith(".ipynb"):
        notebook = json.loads(text)
        cells = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                cells.append("".join(cell.get("source", [])))
        source = "\n\n".join(cells)
    else:
        source = text

    patterns = {
        "قراءة البيانات": ["read_csv", "read_excel"],
        "فحص شكل البيانات": ["shape", "info()", "describe"],
        "القيم المفقودة": ["isnull", "isna", "missing", "fillna", "dropna"],
        "ترتيب البيانات": ["sort_values", "reset_index"],
        "Feature Engineering": ["groupby", "diff(", "inflation_change", "gdp_growth_change"],
        "Correlation / Heatmap": ["corr(", "heatmap", "imshow"],
        "Target Distribution": ["value_counts", "target distribution", "countplot", "barplot"],
        "Histograms": ["histplot", "hist("],
        "Boxplots": ["boxplot", "px.box"],
        "Scatter / Pair Plot": ["scatter", "pairplot", "scatter_matrix"],
        "Country Analysis": ["country", "iso3"],
        "Outlier Analysis": ["quantile", "IQR", "outlier"],
        "Model Prediction": ["predict", "predict_proba", "feature_importances"],
    }
    detected = [name for name, tokens in patterns.items() if any(token.lower() in source.lower() for token in tokens)]
    return {"source": source, "detected": detected, "file_name": file_name}


@st.cache_data(show_spinner=False)
def get_high_correlation_pairs(df: pd.DataFrame, columns: list[str], threshold: float = 0.90) -> pd.DataFrame:
    """حساب أزواج الارتباط العالي مع cache حتى لا يعاد الحساب عند كل تفاعل."""
    if len(columns) < 2:
        return pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation"])
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().reset_index()
    if pairs.empty:
        return pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation"])
    pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    pairs = pairs[pairs["Correlation"].abs() >= threshold].copy()
    return pairs.sort_values("Correlation", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def render_heatmap(df: pd.DataFrame, columns: list[str], title: str):
    """عرض Heatmap تفاعلية كبيرة وواضحة وليست صورة صغيرة."""
    if len(columns) < 2:
        st.info("اختر عمودين رقميين على الأقل لعرض Heatmap.")
        return
    correlation = df[columns].apply(pd.to_numeric, errors="coerce").corr().round(2)
    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=title,
    )
    fig.update_layout(
        height=max(720, 70 * len(columns)),
        width=None,
        template="plotly_dark",
        margin=dict(l=150, r=80, t=100, b=180),
        font=dict(size=16),
        coloraxis_colorbar=dict(title="Correlation", len=0.75),
    )
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=14), automargin=True)
    fig.update_yaxes(tickfont=dict(size=14), automargin=True)
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=15, color="white")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
    st.dataframe(correlation, use_container_width=True)


def get_estimator(model):
    """إذا كان الملف GridSearchCV نأخذ best_estimator_، وإلا نستخدمه مباشرة."""
    return getattr(model, "best_estimator_", model)


def get_model_features(model):
    """استخراج أسماء الأعمدة وترتيبها كما تدرب عليها الموديل."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    estimator = get_estimator(model)
    if hasattr(estimator, "feature_names_in_"):
        return list(estimator.feature_names_in_)
    if hasattr(estimator, "named_steps"):
        for step in estimator.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return [
        "year", "inflation", "gdp_growth", "unemployment", "fed_funds_rate",
        "interest_rate", "exchange_rate", "foreign_reserves",
        "current_account_gdp", "inflation_change", "gdp_growth_change",
    ]


def get_classifier(model):
    """استخراج XGBoost classifier لاستخدام feature importance."""
    estimator = get_estimator(model)
    if hasattr(estimator, "named_steps"):
        return estimator.named_steps.get("clf", estimator)
    return estimator

# =============================================================================
# 4) تجهيز البيانات الجديدة بنفس تجهيز التدريب
# =============================================================================
def prepare_for_model(
    df: pd.DataFrame,
    model_features: list[str],
    for_prediction: bool = False,
    use_original_file_rules: bool = False,
):
    """تجهيز مرن لأي CSV مع الحفاظ على قواعد Features الخاصة بالموديل.

    كل ملف جديد يُرتب ويُحوّل ويُضاف له diff عند توفر أعمدة الدولة/السنة،
    ثم تُطابق Features الموديل. قواعد الملف الأصلي الخاصة بفلترة year > 2015
    لا تُفرض على الملفات الجديدة؛ تُفعّل فقط عند اختيار ملف economic_indicators الأصلي.
    """
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    group_col = None
    if "iso3" in data.columns:
        group_col = "iso3"
    elif "country" in data.columns:
        group_col = "country"

    # هذا الترتيب مهم جدًا حتى تكون diff محسوبة من السنة السابقة للدولة نفسها.
    if "year" in data.columns and group_col:
        data = data.sort_values([group_col, "year"]).reset_index(drop=True)
        if "inflation" in data.columns and "inflation_change" in model_features:
            data["inflation_change"] = data.groupby(group_col)["inflation"].diff()
        if "gdp_growth" in data.columns and "gdp_growth_change" in model_features:
            data["gdp_growth_change"] = data.groupby(group_col)["gdp_growth"].diff()
    elif "year" in data.columns:
        data = data.sort_values("year").reset_index(drop=True)
        if "inflation" in data.columns and "inflation_change" in model_features:
            data["inflation_change"] = data["inflation"].diff()
        if "gdp_growth" in data.columns and "gdp_growth_change" in model_features:
            data["gdp_growth_change"] = data["gdp_growth"].diff()

    # هذه الخطوات موجودة في crisis_prediction_pipeline_xgb_fixed_no_nulls.ipynb
    # وتُستخدم فقط عند تشغيل prediction، وليس أثناء حساب مقاييس التدريب.
    if for_prediction:
        if use_original_file_rules and "year" in data.columns:
            data = data[data["year"] > 2015].reset_index(drop=True)
        if use_original_file_rules and "government_debt_gdp" in data.columns:
            data = data.drop(columns=["government_debt_gdp"])
        if "iso3" in data.columns:
            data["iso3"] = data["iso3"].fillna("GRP")

    # نحافظ على نفس ترتيب أعمدة التدريب حرفيًا.
    for feature in model_features:
        if feature not in data.columns:
            data[feature] = np.nan
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    return data, data[model_features].copy()


def get_global_top_feature(model):
    """إرجاع أهم Feature عامة، للاستخدام في رسم Feature Importance فقط."""
    clf = get_classifier(model)
    features = get_model_features(model)
    if hasattr(clf, "feature_importances_"):
        scores = np.asarray(clf.feature_importances_)
        if len(scores) == len(features):
            return features[int(np.argmax(scores))]
    return "غير متاح"


def get_local_top_features(model, imputed_X: np.ndarray) -> list[str]:
    """حساب العمود المؤثر لكل صف باستخدام مساهمات XGBoost المحلية.

    feature_importances_ تعطي أهمية عامة واحدة للموديل، لذلك لا نستخدمها
    لوصف كل صف. pred_contribs يعيد مساهمة كل Feature في قرار كل حالة،
    ونختار أكبر مساهمة بالقيمة المطلقة مع استبعاد bias.
    """
    features = get_model_features(model)
    clf = get_classifier(model)
    try:
        booster = clf.get_booster()
        matrix = xgb.DMatrix(imputed_X, feature_names=features)
        contributions = np.asarray(booster.predict(matrix, pred_contribs=True))
        if contributions.ndim == 2 and contributions.shape[1] >= len(features):
            local_scores = contributions[:, :len(features)]
            indices = np.argmax(np.abs(local_scores), axis=1)
            return [features[int(index)] for index in indices]
    except Exception:
        pass
    return ["غير متاح"] * len(imputed_X)


def predict_new_data(
    raw_df: pd.DataFrame,
    model: object,
    threshold: float,
    use_original_file_rules: bool = False,
):
    """تنفيذ التنبؤ بنفس Pipeline، مع حفظ القيم بعد imputer في الناتج."""
    features = get_model_features(model)
    prepared, X = prepare_for_model(
        raw_df,
        features,
        for_prediction=True,
        use_original_file_rules=use_original_file_rules,
    )
    probability = model.predict_proba(X)[:, 1]

    result = prepared.copy()
    imputed_input = X.to_numpy(dtype=float)

    # predict_proba عالج Nulls داخليًا، لكن DataFrame الناتج قد يحتفظ بالقيم الأصلية.
    # notebook يكتب هنا القيم بعد fitted imputer حتى لا يظهر Null في أعمدة Features.
    estimator = get_estimator(model)
    imputer = getattr(estimator, "named_steps", {}).get("imputer") if hasattr(estimator, "named_steps") else None
    if imputer is not None and hasattr(imputer, "transform"):
        try:
            imputed_values = imputer.transform(X)
            imputed_input = np.asarray(imputed_values, dtype=float)
            # نحول أعمدة Features الموجودة إلى float قبل الكتابة حتى يقبل Pandas قيم imputer العشرية.
            result = result.astype({feature: "float64" for feature in features})
            result.loc[:, features] = pd.DataFrame(imputed_input, index=result.index, columns=features)
        except Exception as exc:
            st.warning(f"تعذر عرض القيم بعد Imputer في جدول النتائج، مع استمرار التنبؤ: {exc}")
    result["crisis_probability"] = np.round(probability * 100, 2)
    result["crisis_next_12m_prediction"] = (probability >= threshold).astype(int)
    result["top_influencing_feature"] = get_local_top_features(model, imputed_input)
    return result

# =============================================================================
# 5) رسومات التحليل العامة
# =============================================================================
def render_selected_chart(df: pd.DataFrame, x_col: str, y_col: str | None, chart_type: str, color_col: str | None):
    """رسم واحد بناءً على نوع الرسم والأعمدة التي اختارها المستخدم."""
    plot_df = df.copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    if y_col:
        plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col] + ([y_col] if y_col else []))

    if chart_type == "Histogram":
        fig = px.histogram(plot_df, x=x_col, color=color_col, nbins=30, marginal="box", title=f"Histogram: {x_col}")
    elif chart_type == "Scatter":
        if not y_col:
            st.warning("Scatter يحتاج اختيار عمود X وعمود Y.")
            return
        fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, hover_data=[c for c in ["country", "iso3", "year"] if c in plot_df.columns], title=f"Scatter: {x_col} vs {y_col}")
    elif chart_type == "Box plot":
        fig = px.box(plot_df, y=x_col, color=color_col, points="outliers", title=f"Box plot: {x_col}")
    else:  # Line chart
        if not y_col:
            st.warning("Line chart يحتاج اختيار عمود Y.")
            return
        fig = px.line(plot_df.sort_values(x_col), x=x_col, y=y_col, color=color_col, title=f"Line: {y_col} by {x_col}")
    fig.update_layout(height=480, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def show_prediction_status(result: pd.DataFrame, threshold: float):
    """عرض ألوان وحالة مختصرة للتنبؤات بدل عرض أرقام فقط."""
    if "crisis_next_12m_prediction" not in result.columns:
        return
    count_crisis = int((result["crisis_next_12m_prediction"] == 1).sum())
    total = len(result)
    ratio = count_crisis / total if total else 0
    if ratio >= 0.50:
        st.markdown(f'<div class="status-crisis">تنبيه قوي: {count_crisis:,} من {total:,} حالة مصنفة Crisis ({ratio:.1%})</div>', unsafe_allow_html=True)
    elif count_crisis > 0:
        st.markdown(f'<div class="status-warning">تنبيه متوسط: {count_crisis:,} من {total:,} حالة مصنفة Crisis ({ratio:.1%})</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-safe">لا توجد حالات Crisis حسب Threshold = {threshold:.0%}</div>', unsafe_allow_html=True)

# =============================================================================
# 6) كل تحليلات global_criss_anylasis_code.ipynb
# =============================================================================
def render_complete_training_analysis(raw_df: pd.DataFrame):
    """تنفيذ كل الرسومات والجداول الموجودة في notebook التحليل دون حذف أي تحليل."""
    data = raw_df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    target = TARGET if TARGET in data.columns else None

    # يطابق notebook: إزالة الصفوف المتكررة قبل حساب الارتباطات.
    duplicate_count = int(data.duplicated().sum())
    st.subheader("1) Dataset overview")
    st.write(f"Duplicate rows before cleaning: **{duplicate_count:,}**")
    if duplicate_count:
        data = data.drop_duplicates().reset_index(drop=True)
        st.write(f"Duplicate rows removed: **{duplicate_count:,}**")

    # في notebook، correlation_before يشمل الـ Target ويستبعد الأعمدة النصية فقط.
    numeric_before = data.select_dtypes(include="number").columns.tolist()
    text_columns = data.select_dtypes(exclude="number").columns.tolist()

    overview = pd.DataFrame({
        "Column": data.columns,
        "Data type": [str(data[c].dtype) for c in data.columns],
        "Missing values": [int(data[c].isna().sum()) for c in data.columns],
        "Missing %": [round(float(data[c].isna().mean() * 100), 2) for c in data.columns],
        "Unique values": [int(data[c].nunique(dropna=True)) for c in data.columns],
    })
    st.dataframe(overview, use_container_width=True, hide_index=True)
    st.write(f"Shape: **{data.shape[0]:,} rows × {data.shape[1]} columns**")
    if target:
        st.write(f"Target column: **{target}**")

    st.subheader("2) Numeric correlation before feature engineering")
    if len(numeric_before) >= 2:
        st.write("**Heatmap 1 — Before preprocessing / البيانات الخام**")
        before_columns = st.multiselect("اختيار أعمدة Heatmap قبل التجهيز", numeric_before, default=numeric_before, key="before_heatmap_columns")
        render_heatmap(data, before_columns, "Correlation Heatmap — Before Preprocessing")
        st.markdown("### Highly correlated pairs before preprocessing")
        before_high_corr = get_high_correlation_pairs(data, before_columns, 0.90)
        if before_high_corr.empty:
            st.info("لا توجد أزواج ارتباط مطلقها ≥ 0.90 قبل تجهيز البيانات.")
        else:
            st.dataframe(before_high_corr, use_container_width=True, hide_index=True)
            st.caption("يتم عرض الأزواج التي تحقق |correlation| ≥ 0.90 قبل Feature Engineering.")

    # نفس feature engineering الموجود في notebook، مع الحفاظ على raw data للعرض.
    engineered = data.copy()
    # نفس Feature Engineering الموجود حرفيًا في notebook: iso3 ثم year، ثم diff لكل دولة.
    if "iso3" in engineered.columns and "year" in engineered.columns:
        engineered = engineered.sort_values(["iso3", "year"]).reset_index(drop=True)
        if "inflation" in engineered.columns:
            engineered["inflation_change"] = engineered.groupby("iso3")["inflation"].diff()
        if "gdp_growth" in engineered.columns:
            engineered["gdp_growth_change"] = engineered.groupby("iso3")["gdp_growth"].diff()

    # نفس العمودين اللذين يحذفهما notebook بعد Feature Engineering.
    dropped_after_engineering = [c for c in ["government_debt_gdp", "real_interest_rate_10y"] if c in engineered.columns]
    if dropped_after_engineering:
        engineered = engineered.drop(columns=dropped_after_engineering)

    # Heatmap بعد التجهيز تشمل Target، أما الرسومات الأصلية فتستخدم Features فقط.
    numeric_after = engineered.select_dtypes(include="number").columns.tolist()
    numeric_plot_after = [c for c in numeric_after if c != target]

    st.subheader("3) Feature engineering result")
    st.write("تمت إضافة `inflation_change` و`gdp_growth_change` بعد الترتيب حسب الدولة والسنة، كما في notebook.")
    st.dataframe(engineered.head(20), use_container_width=True, hide_index=True)

    st.subheader("4) Numeric correlation after feature engineering")
    if len(numeric_after) >= 2:
        st.write("**Heatmap 2 — After preprocessing / بعد تجهيز البيانات**")
        after_columns = st.multiselect("اختيار أعمدة Heatmap بعد التجهيز", numeric_after, default=numeric_after, key="after_heatmap_columns")
        render_heatmap(engineered, after_columns, "Correlation Heatmap — After Preprocessing and Feature Engineering")
        st.markdown("### Highly correlated pairs after preprocessing")
        after_high_corr = get_high_correlation_pairs(engineered, after_columns, 0.90)
        if after_high_corr.empty:
            st.info("لا توجد أزواج ارتباط مطلقها ≥ 0.90 بعد تجهيز البيانات.")
        else:
            st.dataframe(after_high_corr, use_container_width=True, hide_index=True)
            st.caption("يتم عرض الأزواج التي تحقق |correlation| ≥ 0.90 بعد Feature Engineering.")

    if target:
        st.subheader("5) Target distribution")
        target_counts = engineered[target].value_counts().sort_index()
        # تحويل آمن لفئات الهدف: لا نستخدم Index.map(...).fillna(...) لتفادي خطأ Pandas.
        target_labels = [
            {0: "Class 0 - No Crisis", 1: "Class 1 - Crisis"}.get(value, str(value))
            for value in target_counts.index
        ]
        st.plotly_chart(px.bar(x=target_labels, y=target_counts.values, labels={"x":"Target Class", "y":"Rows"}, title="Target Distribution: crisis_next_12m"), use_container_width=True)
        st.dataframe(target_counts.rename("Rows"), use_container_width=True)

    st.subheader("6) Histograms of all numeric features")
    hist_features = st.multiselect("اختيار أعمدة histogram كما في notebook", numeric_plot_after, default=numeric_plot_after, key="all_hist_features")
    if hist_features:
        hist_long = engineered[hist_features].melt(var_name="Feature", value_name="Value").dropna()
        hist_fig = px.histogram(
            hist_long,
            x="Value",
            color="Feature",
            facet_row="Feature",
            nbins=35,
            marginal="box",
            opacity=0.82,
            facet_row_spacing=0.045,
            title="Interactive Histograms of Numeric Features",
            labels={"Value": "Feature value", "count": "Number of rows"},
        )
        hist_fig.update_layout(
            height=max(650, 330 * len(hist_features)),
            template="plotly_white",
            bargap=0.08,
            hovermode="x unified",
            margin=dict(l=80, r=50, t=90, b=80),
            font=dict(size=14),
            showlegend=False,
        )
        hist_fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB", automargin=True)
        hist_fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB", automargin=True)
        st.plotly_chart(
            hist_fig,
            use_container_width=True,
            config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
        )
        st.caption("يمكنك تمرير المؤشر لرؤية القيم، واستخدام التكبير والتحريك وإخفاء/إظهار السلاسل من شريط الأدوات.")

    if target:
        st.subheader("7) Box plots of all numeric features by target class")
        box_features = st.multiselect("اختيار أعمدة boxplot كما في notebook", numeric_plot_after, default=numeric_plot_after, key="all_box_features")
        if box_features:
            box_long = engineered[box_features + [target]].melt(id_vars=[target], var_name="Feature", value_name="Value").dropna()
            box_long["Target Class"] = box_long[target].map({0: "Class 0 - No Crisis", 1: "Class 1 - Crisis"})
            box_fig = px.box(
                box_long,
                x="Feature",
                y="Value",
                color="Target Class",
                points="outliers",
                boxmode="group",
                notched=False,
                title="Interactive Boxplots by Target Class",
                labels={"Value": "Feature value", "Feature": "Numeric feature"},
                color_discrete_map={
                    "Class 0 - No Crisis": "#2E86DE",
                    "Class 1 - Crisis": "#E74C3C",
                },
            )
            box_fig.update_layout(
                height=680,
                template="plotly_white",
                hovermode="closest",
                margin=dict(l=80, r=50, t=90, b=170),
                font=dict(size=14),
                boxgap=0.35,
                boxgroupgap=0.18,
                legend_title_text="Target class",
            )
            box_fig.update_xaxes(tickangle=-45, showgrid=False, automargin=True)
            box_fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB", automargin=True, zeroline=True)
            st.plotly_chart(
                box_fig,
                use_container_width=True,
                config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
            )
            st.caption("يمكنك تمرير المؤشر على كل Box لرؤية الوسيط والربيعات والقيم الشاذة، مع التكبير والتحريك والتصفية من الـLegend.")

    if "country" in engineered.columns:
        st.subheader("8) Top 15 countries by number of rows")
        country_counts = engineered["country"].value_counts().head(15).sort_values()
        st.plotly_chart(px.bar(x=country_counts.values, y=country_counts.index, orientation="h", labels={"x":"Number of rows", "y":"Country"}, title="Top 15 Countries by Number of Rows"), use_container_width=True)
        st.dataframe(country_counts.rename("Rows"), use_container_width=True)

    st.subheader("9) Outlier summary for every numeric feature")
    outlier_rows = []
    for feature in numeric_after:
        values = pd.to_numeric(engineered[feature], errors="coerce").dropna()
        if len(values) == 0:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (values < low) | (values > high)
        outlier_rows.append({"Feature": feature, "Q1": q1, "Q3": q3, "IQR": iqr, "Lower bound": low, "Upper bound": high, "Outlier count": int(mask.sum()), "Outlier %": round(float(mask.mean() * 100), 2)})
    st.dataframe(pd.DataFrame(outlier_rows).sort_values("Outlier count", ascending=False), use_container_width=True, hide_index=True)

    if target:
        st.subheader("10) Feature overlap by target class — pair plot")
        default_pair = [c for c in ["inflation", "gdp_growth", "unemployment", "fed_funds_rate", "interest_rate"] if c in numeric_plot_after]
        pair_features = st.multiselect("اختيار أعمدة pair plot كما في notebook", numeric_plot_after, default=default_pair, max_selections=6, key="pair_features")
        pair_df = engineered[pair_features + [target]].dropna().copy() if pair_features else pd.DataFrame()
        if len(pair_features) >= 2 and not pair_df.empty:
            pair_df["Target Class"] = pair_df[target].map({0: "Class 0 - No Crisis", 1: "Class 1 - Crisis"})
            fig = px.scatter_matrix(pair_df, dimensions=pair_features, color="Target Class", title="Feature Overlap by Target Class", height=850)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("اختر عمودين على الأقل لعرض pair plot.")

# =============================================================================
# 7) مقاييس train/test
# =============================================================================
def calculate_metrics(model, train_df: pd.DataFrame, threshold: float):
    """حساب Accuracy وPrecision وRecall وF1 على train/test."""
    if TARGET not in train_df.columns:
        return None
    features = get_model_features(model)
    cleaned, X = prepare_for_model(train_df, features)
    y = pd.to_numeric(cleaned[TARGET], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def evaluate(X_part, y_part):
        p = model.predict_proba(X_part)[:, 1]
        pred = (p >= threshold).astype(int)
        report = classification_report(y_part, pred, labels=[0, 1], output_dict=True, zero_division=0)
        return {
            "accuracy": accuracy_score(y_part, pred),
            "precision_class_0": report["0"]["precision"],
            "recall_class_0": report["0"]["recall"],
            "f1_class_0": report["0"]["f1-score"],
            "precision_class_1": report["1"]["precision"],
            "recall_class_1": report["1"]["recall"],
            "f1_class_1": report["1"]["f1-score"],
            "confusion_matrix": confusion_matrix(y_part, pred, labels=[0, 1]),
        }
    return evaluate(X_train, y_train), evaluate(X_test, y_test)

# =============================================================================
# 7) بدء التطبيق
# =============================================================================
st.markdown('<div class="main-title">Early Alarm Warning System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Global Economic Crisis Prediction and Analysis Dashboard</div>', unsafe_allow_html=True)

try:
    model = load_model()
    model_features = get_model_features(model)
except Exception as exc:
    st.error(f"تعذر تحميل النموذج: {exc}")
    st.stop()

st.sidebar.header("System Settings")
threshold = st.sidebar.slider("Crisis Threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.01)

# لا يتم تحميل ملف التدريب أو prediction إلا عند فتح القسم الخاص به.
section = st.sidebar.radio(
    "اختر وظيفة واحدة فقط",
    [
        "1 - Training Data Analysis",
        "2 - New Data Prediction",
        "3 - Ready Prediction Viewer",
        "4 - Model Performance",
    ],
)

# =============================================================================
# القسم الأول: تحليل بيانات التدريب فقط
# =============================================================================
if section == "1 - Training Data Analysis":
    st.header("Training Data Analysis")
    st.write("هذا القسم مخصص للتحليل فقط. يمكنك رفع ملف Notebook أو Python للتحليل، ثم اختيار البيانات التي سيُطبق عليها التحليل. لا يتم استخدامه لإنتاج توقعات.")

    # الخيار الأساسي: رفع كود التحليل، والخيار الاحتياطي: تشغيل التحليل المدمج المطابق للـ notebook.
    analysis_mode = st.radio(
        "طريقة تشغيل التحليل",
        ["Use uploaded analysis code", "Use built-in notebook-compatible analysis (fallback)"],
        horizontal=True,
        key="analysis_mode",
    )
    analysis_code_upload = None
    if analysis_mode == "Use uploaded analysis code":
        analysis_code_upload = st.file_uploader(
            "ارفع ملف كود التحليل (.ipynb أو .py)",
            type=["ipynb", "py"],
            key="analysis_code_upload",
        )
        if analysis_code_upload is None:
            st.info("ارفع ملف التحليل، أو اختر الخيار الاحتياطي لتشغيل التحليل المدمج.")
        else:
            analysis_code_info = extract_analysis_code(analysis_code_upload.name, analysis_code_upload.getvalue())
            st.success(f"تمت قراءة كود التحليل: {analysis_code_upload.name}")
            detected = analysis_code_info["detected"]
            st.write("**التحليلات والخطوات المكتشفة من الملف:**")
            st.write(" → ".join(detected) if detected else "لم يتم التعرف آليًا على أسماء خطوات؛ ستظل كل التحليلات العامة متاحة.")
            with st.expander("عرض كود التحليل المرفوع — للمرجعية فقط"):
                st.code(analysis_code_info["source"], language="python")

    use_uploaded_analysis = st.checkbox("رفع CSV آخر للتحليل بدل global_crisis_new.csv")

    if use_uploaded_analysis:
        analysis_upload = st.file_uploader("ارفع ملف CSV للتحليل", type=["csv"], key="analysis_upload")
        if analysis_upload is None:
            st.info("ارفع ملف CSV حتى تظهر اختيارات التحليل.")
            st.stop()
        analysis_df = load_uploaded_csv(analysis_upload.getvalue())
        analysis_source = "Uploaded analysis CSV"
    else:
        if not TRAIN_DATA_PATH.exists():
            st.error(f"ملف التدريب غير موجود: {TRAIN_DATA_PATH}")
            st.stop()
        analysis_df = load_csv_from_path(str(TRAIN_DATA_PATH))
        analysis_source = "global_crisis_new.csv"

    st.success(f"تم تحميل مصدر التحليل: {analysis_source} — عدد الصفوف: {len(analysis_df):,}")

    # اختيار الأعمدة مستقل داخل قسم التحليل.
    # هذا الاستدعاء يعرض كل تحليلات notebook دون استثناء.
    render_complete_training_analysis(analysis_df)

# =============================================================================
# القسم الثاني: التنبؤ على بيانات جديدة فقط
# =============================================================================
elif section == "2 - New Data Prediction":
    st.header("New Data Prediction")
    st.write("ارفع CSV جديدًا لتطبيق الـ Pipeline المرن عليه، أو أدخل حالة واحدة يدويًا.")

    prediction_mode = st.radio(
        "اختيار طريقة إدخال البيانات",
        ["CSV prediction", "Manual single prediction"],
        horizontal=True,
        key="prediction_mode",
    )

    if prediction_mode == "Manual single prediction":
        st.info("أدخل قيم Features الخاصة بالموديل. سيتم تطبيق نفس ترتيب الأعمدة وتجهيز البيانات قبل التوقع.")
        manual_values = {}
        manual_cols = st.columns(2)
        for index, feature in enumerate(model_features):
            with manual_cols[index % 2]:
                manual_values[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.6f",
                    key=f"manual_{feature}",
                )
        manual_country = st.text_input("اسم البلد — اختياري", value="Manual Country")
        manual_iso3 = st.text_input("ISO3 — اختياري", value="MAN")
        manual_year = st.number_input("السنة — اختياري", min_value=1900, max_value=2100, value=2025, step=1)
        if st.button("Run manual prediction", type="primary"):
            manual_row = pd.DataFrame([manual_values])
            manual_row["country"] = manual_country
            manual_row["iso3"] = manual_iso3
            manual_row["year"] = int(manual_year)
            with st.spinner("Applying the trained pipeline to the manual row..."):
                manual_result = predict_new_data(manual_row, model, threshold)
            st.success("تم تنفيذ التوقع اليدوي بنجاح.")
            show_prediction_status(manual_result, threshold)
            st.dataframe(manual_result, use_container_width=True)
            st.download_button("Download manual prediction", manual_result.to_csv(index=False).encode("utf-8"), "manual_prediction.csv", "text/csv")
    else:
        new_upload = st.file_uploader("ارفع CSV جديد للتنبؤ", type=["csv"], key="new_prediction_upload")
        if new_upload is None:
            st.info("ارفع ملف CSV حتى يبدأ التنبؤ.")
            st.stop()
        # يتم تمرير bytes إلى دالة عليها cache_data حتى لا يعاد تحليل نفس الملف عند كل تفاعل.
        new_df = load_uploaded_csv(new_upload.getvalue())

        st.write("عدد الصفوف المدخلة:", f"{len(new_df):,}")
        st.write("أعمدة النموذج المطلوبة:", model_features)
        missing = [c for c in model_features if c not in new_df.columns]
        if missing:
            st.warning(f"أعمدة غير موجودة وسيتم تجهيزها كقيم مفقودة: {missing}")

        if st.button("Run prediction on CSV", type="primary"):
            with st.spinner("Preparing data and running the trained pipeline..."):
                prediction_result = predict_new_data(
                    new_df,
                    model,
                    threshold,
                    use_original_file_rules=False,
                )
            st.success("تم تجهيز البيانات والتنبؤ بنجاح.")
            show_prediction_status(prediction_result, threshold)
            st.dataframe(prediction_result, use_container_width=True, height=520)
            st.download_button("Download prediction result", prediction_result.to_csv(index=False).encode("utf-8"), "new_data_predictions.csv", "text/csv")

# =============================================================================
# القسم الثالث: عرض ملف prediction الجاهز فقط
# =============================================================================
elif section == "3 - Ready Prediction Viewer":
    st.header("Ready Prediction Viewer")
    st.write("هذا القسم يعرض ملف التوقعات الجاهز crisis_predictions_full.csv، وليس ملف التدريب الخام.")

    use_uploaded_prediction = st.checkbox("رفع prediction CSV آخر بدل الملف المحدد في المسار")
    if use_uploaded_prediction:
        ready_upload = st.file_uploader("ارفع ملف prediction جاهز", type=["csv"], key="ready_prediction_upload")
        if ready_upload is None:
            st.info("ارفع ملف prediction جاهز.")
            st.stop()
        ready_df = load_uploaded_csv(ready_upload.getvalue())
    else:
        if not PREDICTIONS_PATH.exists():
            st.error(f"ملف prediction غير موجود: {PREDICTIONS_PATH}")
            st.stop()
        ready_df = load_csv_from_path(str(PREDICTIONS_PATH))

    st.success(f"تم تحميل ملف prediction الجاهز — عدد الصفوف: {len(ready_df):,}")
    if "crisis_next_12m" in ready_df.columns:
        positive_col = "crisis_next_12m"
    elif "crisis_next_12m_prediction" in ready_df.columns:
        positive_col = "crisis_next_12m_prediction"
    else:
        positive_col = None

    if positive_col:
        count_positive = int(pd.to_numeric(ready_df[positive_col], errors="coerce").fillna(0).sum())
        total_rows = len(ready_df)
        ratio = count_positive / total_rows if total_rows else 0
        if ratio >= 0.5:
            st.markdown(f'<div class="status-crisis">الملف يحتوي على {count_positive:,} حالة Crisis من {total_rows:,}</div>', unsafe_allow_html=True)
        elif count_positive > 0:
            st.markdown(f'<div class="status-warning">الملف يحتوي على {count_positive:,} حالة Crisis من {total_rows:,}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-safe">لا توجد حالات Crisis في الملف الجاهز</div>', unsafe_allow_html=True)

    # عرض الجدول الجاهز فقط، بدون إعادة predict أو استخدام raw training data.
    st.dataframe(ready_df, use_container_width=True, height=560)
    st.download_button("Download displayed prediction file", ready_df.to_csv(index=False).encode("utf-8"), "displayed_predictions.csv", "text/csv")

    # في ملف prediction نعرض خيارين فقط كما طلبت:
    # 1) Heatmap للارتباط بين الأعمدة الرقمية الموجودة في ملف النتائج.
    # 2) Feature Importance للموديل الذي أنشأ التوقعات.
    st.subheader("Prediction file analysis")
    prediction_analysis_option = st.radio(
        "اختيار نوع العرض",
        ["Heatmap", "Feature Importance"],
        horizontal=True,
        key="prediction_analysis_option",
    )
    prediction_numeric = ready_df.select_dtypes(include="number").columns.tolist()

    if prediction_analysis_option == "Heatmap":
        heatmap_cols = st.multiselect(
            "اختيار أعمدة Heatmap من ملف prediction",
            prediction_numeric,
            default=prediction_numeric,
            key="prediction_heatmap_columns",
        )
        if len(heatmap_cols) >= 2:
            prediction_corr = ready_df[heatmap_cols].corr().round(2)
            heatmap_fig = px.imshow(
                prediction_corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation Heatmap — Prediction Data",
            )
            heatmap_fig.update_layout(height=650, template="plotly_white")
            st.plotly_chart(heatmap_fig, use_container_width=True)
            st.dataframe(prediction_corr, use_container_width=True)
        else:
            st.info("اختر عمودين رقميين على الأقل لعرض Heatmap.")

    else:
        # feature_importances_ مأخوذة من نفس classifier داخل Pipeline المحفوظة.
        classifier = get_classifier(model)
        model_importance = getattr(classifier, "feature_importances_", None)
        if model_importance is None:
            st.warning("Feature Importance غير متاحة في نوع الموديل المحفوظ.")
        else:
            importance_df = pd.DataFrame({
                "Feature": model_features,
                "Importance": np.asarray(model_importance),
            }).sort_values("Importance", ascending=True)
            importance_df["Importance_%"] = (importance_df["Importance"] * 100).round(2)
            importance_fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                text="Importance_%",
                title="Feature Importance — XGBoost Prediction Model",
                color="Importance",
                color_continuous_scale="Blues",
            )
            importance_fig.update_layout(height=650, template="plotly_white")
            st.plotly_chart(importance_fig, use_container_width=True)
            st.dataframe(importance_df.sort_values("Importance", ascending=False), use_container_width=True, hide_index=True)

# =============================================================================
# القسم الرابع: الأداء والمقاييس
# =============================================================================
else:
    st.header("Model Performance")
    st.write("المقاييس تحسب على global_crisis_new.csv فقط، لأنه ملف التدريب الذي يحتوي على target الحقيقي.")
    if not TRAIN_DATA_PATH.exists():
        st.error(f"ملف التدريب غير موجود: {TRAIN_DATA_PATH}")
        st.stop()
    train_for_metrics = load_csv_from_path(str(TRAIN_DATA_PATH))
    metrics = calculate_metrics(model, train_for_metrics, threshold)
    if metrics is None:
        st.error(f"لا يوجد عمود {TARGET} في ملف التدريب.")
    else:
        train_m, test_m = metrics
        metric_keys = [
            "accuracy", "precision_class_0", "recall_class_0", "f1_class_0",
            "precision_class_1", "recall_class_1", "f1_class_1",
        ]
        table = pd.DataFrame({"Train": {k: train_m[k] for k in metric_keys}, "Test": {k: test_m[k] for k in metric_keys}})
        table["Overfitting gap"] = table["Train"] - table["Test"]
        st.dataframe(table.style.format("{:.4f}"), use_container_width=True)
        st.write("Precision Class 1 يوضح دقة الإنذارات التي قال عنها النموذج إنها أزمات. Recall Class 1 يوضح نسبة الأزمات الحقيقية التي اكتشفها النموذج. فجوة Train-Test الكبيرة قد تشير إلى overfitting.")
        a, b = st.columns(2)
        a.write("Train confusion matrix")
        a.dataframe(pd.DataFrame(train_m["confusion_matrix"], index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))
        b.write("Test confusion matrix")
        b.dataframe(pd.DataFrame(test_m["confusion_matrix"], index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))
        if test_m["recall_class_1"] >= 0.60:
            st.success(f"Recall Class 1 على Test جيد نسبيًا: {test_m['recall_class_1']:.3f}")
        else:
            st.warning(f"Recall Class 1 على Test يحتاج مراجعة: {test_m['recall_class_1']:.3f}")

st.divider()
st.caption("Early Alarm Warning System | XGBoost | Cached data loading | Training analysis / New prediction / Ready prediction viewer")
