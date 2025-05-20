import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import dill

# ========== 1. 加载模型和预处理器 ==========
stacking_regressor    = joblib.load("stacking_model.pkl")

qt_lcd                = joblib.load("qt_lcd.pkl")
qt_gsa                = joblib.load("qt_GSA.pkl")
qt_density            = joblib.load("qt_Density.pkl")
qt_ktol               = joblib.load("qt_Ktoluene.pkl")       # 如果你也用 QuantileTransformer 存过 Ktoluene，请改为对应文件名
boxcox_kt_transformer = joblib.load("lambda_Ktoluene.pkl")  # 你的 FixedBoxCoxTransformer 对象
boxcox_lambda_vf      = float(joblib.load("lambda_vf.pkl"))  # 直接保存的 λ 值

qt_TSN                = joblib.load("qt_TSN.pkl")

# ========== 2. Streamlit 页面配置 ==========
st.set_page_config(layout="wide", page_title="Stacking 模型预测与 SHAP 可视化", page_icon="📊")
st.title("📊 Stacking 模型预测与 SHAP 可视化分析")
st.write("""
通过输入特征值进行模型预测，并结合 SHAP 分析结果，了解特征对模型预测的贡献。
""")

# ========== 3. 侧边栏用户输入（不限制小数位数） ==========
st.sidebar.header("特征输入区域")
st.sidebar.write("请输入特征值：")
LCD      = st.sidebar.number_input(
    "特征 LCD (范围: 6.03338–39.1106)", min_value=6.03338, max_value=39.1106,
    value=8.33119, format="%g"
)
Vf       = st.sidebar.number_input(
    "特征 Vf (范围: 0.2574–0.9182)", min_value=0.2574, max_value=0.9182,
    value=0.5726, format="%g"
)
GSA      = st.sidebar.number_input(
    "特征 GSA (范围: 204.912–7061.42)", min_value=204.912, max_value=7061.42,
    value=701.884, format="%g"
)
Density  = st.sidebar.number_input(
    "特征 Density (范围: 0.237838–2.86501)", min_value=0.237838, max_value=2.86501,
    value=1.51454, format="%g"
)
Ktoluene = st.sidebar.number_input(
    "特征 Ktoluene (范围: 0.000027383–28527.4)", min_value=0.000027383,
    max_value=28527.4, value=0.013545, format="%g"
)

predict_button = st.sidebar.button("进行预测")

# ========== 4. 预测逻辑（带预处理 + 显示转换 + 反变换） ==========
if predict_button:
    st.header("预测结果")
    try:
        # ---- 4.1 特征逐项转换 ----
        lcd_q     = qt_lcd.transform([[LCD]])[0, 0]
        gsa_q     = qt_gsa.transform([[GSA]])[0, 0]
        density_q = qt_density.transform([[Density]])[0, 0]

        # Vf: 单纯用 Box–Cox λ 值
        vf_arr = np.array([[float(Vf)]], dtype=float)
        # 由于只存了 λ，我们用 scipy.stats.boxcox
        from scipy.stats import boxcox
        vf_bc = boxcox(vf_arr.flatten(), lmbda=boxcox_lambda_vf)[0]

        # Ktoluene: 用自定义 FixedBoxCoxTransformer
        ktol_arr = np.array([[float(Ktoluene)]], dtype=float)
        ktol_bc  = boxcox_kt_transformer.transform(ktol_arr)[0, 0]

        # ---- 4.2 显示输入特征原始 vs 转换后 ----
        df_trans = pd.DataFrame({
            "特征":      ["LCD", "Vf", "GSA", "Density", "Ktoluene"],
            "原始值":    [LCD, Vf, GSA, Density, Ktoluene],
            "转换后值": [lcd_q, vf_bc, gsa_q, density_q, ktol_bc]
        })
        st.subheader("🔄 特征值转换对比")
        st.table(df_trans)

        # ---- 4.3 构造模型输入，顺序与训练一致 ----
        X_user = df_trans["转换后值"].to_numpy().reshape(1, -1)
        st.write("▶ 用于模型的 X_user：", X_user)

        # ---- 4.4 模型预测（变换后 TSN） ----
        pred_trans = stacking_regressor.predict(X_user)[0]
        st.subheader("📈 模型输出 (TSN_transformed)")
        st.write(f"{pred_trans:.6f}")

        # ---- 4.5 反变换回原始 TSN ----
        pred_orig = qt_TSN.inverse_transform([[pred_trans]])[0, 0]
        st.subheader("🏷️ 预测原始 TSN")
        st.success(f"{pred_orig:.6f}")

    except Exception as e:
        st.error(f"预测时发生错误：{e}")

# ========== 5. SHAP 可视化部分（保持原样） ==========
st.header("SHAP 可视化分析")
st.write("""
以下图表展示了模型的 SHAP 分析结果，包括第一层基学习器、第二层元学习器以及整个 Stacking 模型的特征贡献。
""")

# 第一层基学习器 SHAP 可视化
st.subheader("1. 第一层基学习器")
st.write("基学习器（RandomForest、XGB、LGBM 等）的特征贡献分析。")
first_layer_img = "summary_plot.png"
try:
    img1 = Image.open(first_layer_img)
    st.image(img1, caption="第一层基学习器的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到第一层基学习器的 SHAP 图像文件。")

# 第二层元学习器 SHAP 可视化
st.subheader("2. 第二层元学习器")
st.write("元学习器（Linear Regression）的输入特征贡献分析。")
meta_layer_img = "SHAP Contribution Analysis for the Meta-Learner in the Second Layer of Stacking Regressor.png"
try:
    img2 = Image.open(meta_layer_img)
    st.image(img2, caption="第二层元学习器的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到第二层元学习器的 SHAP 图像文件。")

# 整体 Stacking 模型 SHAP 可视化
st.subheader("3. 整体 Stacking 模型")
st.write("整个 Stacking 模型的特征贡献分析。")
overall_img = "Based on the overall feature contribution analysis of SHAP to the stacking model.png"
try:
    img3 = Image.open(overall_img)
    st.image(img3, caption="整体 Stacking 模型的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到整体 Stacking 模型的 SHAP 图像文件。")

# 页脚
st.markdown("---")
st.header("总结")
st.write("""
通过本页面，您可以：
1. 使用输入特征值进行实时预测。
2. 直观地理解第一层基学习器、第二层元学习器以及整体 Stacking 模型的特征贡献情况。
这些分析有助于深入理解模型的预测逻辑和特征的重要性。
""")
