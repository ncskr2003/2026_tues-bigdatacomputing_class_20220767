
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ── 저장된 파일 로드 ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "Linear": joblib.load('linear_model.pkl'),
        "Poly":   joblib.load('poly_model.pkl'),
        "Ridge":  joblib.load('ridge_model.pkl'),
    }

@st.cache_data
def load_meta():
    stats    = joblib.load('data_stats.pkl')
    features = joblib.load('features.pkl')
    return stats, features

@st.cache_data
def get_test_data():
    url = "https://github.com/dongupak/DataML/raw/main/csv/life_expectancy.csv"
    df  = pd.read_csv(url).dropna()
    df.columns = df.columns.str.strip()
    FEATURES = joblib.load('features.pkl')
    X   = df[FEATURES].values
    y   = df['Life expectancy'].values
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    np.random.seed(42)
    idx     = np.random.choice(len(X_tr_full), 50, replace=False)
    X_train = X_tr_full[idx]
    y_train = y_tr_full[idx]
    return X_train, X_test, y_train, y_test

models              = load_models()
stats, FEATURES     = load_meta()
X_train, X_test, y_train, y_test = get_test_data()

#======================================================
#============ 대시보드 설계 ============================
st.set_page_config(page_title="기대수명 예측", layout="wide")
st.title("🏥 AI 기대수명 예측 및 모델 비교 서비스")
st.caption(f"독립변수: {', '.join(FEATURES)}")

# ── [조건 4] 사이드바 슬라이더 ──────────────────────────────────
st.sidebar.header("📋 입력값 설정")

user_vals = {}
for feat in FEATURES:
    mn  = float(stats[feat]['min'])
    mx  = float(stats[feat]['max'])
    avg = float(stats[feat]['mean'])

    if feat == 'GDP':  # GDP만 별도 처리
        user_vals[feat] = st.sidebar.slider(
            f"{feat} (USD)",
            min_value=int(mn),
            max_value=int(mx),
            value=int(avg),
            step=1000           # 1000달러 단위
        )
    else:
        user_vals[feat] = st.sidebar.slider(
            f"{feat}",
            min_value=round(mn, 1),
            max_value=round(mx, 1),
            value=round(avg, 1),
            step=round((mx - mn) / 100, 2)
        )
# 모델 선택 드롭다운
selected = st.sidebar.selectbox("🤖 모델 선택", ["Linear", "Poly", "Ridge"])

# ── [조건 4] 실시간 예측 ─────────────────────────────────────────
input_arr  = np.array([[user_vals[f] for f in FEATURES]])
prediction = models[selected].predict(input_arr)[0]

st.subheader("🎯 예측 결과")
col_pred, col_info = st.columns([1, 2])
with col_pred:
    st.metric(
        label=f"[{selected} 모델] 예측 기대수명",
        value=f"{prediction:.2f} 세"
    )
with col_info:
    st.info(
        f"선택 모델: **{selected}**\n\n"
        + "\n".join([f"- {f}: {v}" for f, v in user_vals.items()])
    )

st.divider()

# ── [조건 3] 성능 비교 섹션 ──────────────────────────────────────
st.subheader("📊 모델 성능 비교")

rows = []
for name, model in models.items():
    complexity = model.named_steps['poly'].n_output_features_
    rows.append({
        'Model':      name,
        'Complexity': complexity,
        'Train R²':   round(model.score(X_train, y_train), 4),
        'Test R²':    round(model.score(X_test,  y_test),  4),
        'Train MSE':  round(mean_squared_error(y_train, model.predict(X_train)), 2),
        'Test MSE':   round(mean_squared_error(y_test,  model.predict(X_test)),  2),
    })
perf_df = pd.DataFrame(rows)

st.dataframe(perf_df.set_index('Model'), use_container_width=True)

# Test R² 막대그래프
fig, ax = plt.subplots(figsize=(6, 4))
colors = ['red' if m == selected else 'steelblue' for m in perf_df['Model']]
bars = ax.bar(perf_df['Model'], perf_df['Test R²'], color=colors, alpha=0.8, edgecolor='black')
ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=11, fontweight='bold')
ax.set_title("Test R² Score Comparison", fontsize=14)
ax.set_xlabel("Model")
ax.set_ylabel("Test R²")
y_min = min(perf_df['Test R²'].min() - 0.1, -0.1)
ax.set_ylim(y_min, 1.05)
ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
ax.grid(True, linestyle=':', alpha=0.5, axis='y')
plt.tight_layout()
st.pyplot(fig)

st.caption("🔴 빨간 막대 = 현재 선택된 모델")
