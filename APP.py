import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from scipy.stats import boxcox

# ========== 1. 加载模型和预处理器 ==========
model_path        = "stacking_model.pkl"
qt_lcd_path       = "qt_lcd.pkl"
qt_gsa_path       = "qt_GSA.pkl"
qt_density_path   = "qt_Density.pkl"
qt_ktol_path      = "qt_Ktoluene.pkl"
lambda_vf_path    = "lambda_vf.pkl"

stacking_regressor = joblib.load(model_path)
qt_lcd             = joblib.load(qt_lcd_path)
qt_gsa             = joblib.load(qt_gsa_path)
qt_density         = joblib.load(qt_density_path)
qt_ktol            = joblib.load(qt_ktol_path)
boxcox_lambda_vf   = joblib.load(lambda_vf_path)

# ========== 2. Streamlit 页面配置 ==========
st.set_page_config(layout="wide", page_title="Stacking 模型预测与 SHAP 可视化", page_icon="📊")
st.title("📊 Stacking 模型预测与 SHAP 可视化分析")
st.write("""
通过输入特征值进行模型预测，并结合 SHAP 分析结果，了解特征对模型预测的贡献。
""")

# ========== 3. 侧边栏用户输入 ==========
st.sidebar.header("特征输入区域")
st.sidebar.write("请输入特征值：")
LCD      = st.sidebar.number_input("特征 LCD (范围: 6.03338-39.1106)", min_value=6.03,  max_value=39.11, value=8.33)
Vf       = st.sidebar.number_input("特征 Vf (范围: 0.2574-0.9182)",   min_value=0.26,  max_value=0.92, value=0.57)
GSA      = st.sidebar.number_input("特征 GSA (范围: 204.912-7061.42)", min_value=204.9, max_value=7061.4, value=701.88)
Density  = st.sidebar.number_input("特征 Density (范围: 0.237838-2.86501)", min_value=0.24, max_value=2.87, value=1.51)
Ktoluene = st.sidebar.number_input("特征 Ktoluene (范围: 0.0000274-28527.4)", min_value=0.00003, max_value=28527.4, value=0.0135)

predict_button = st.sidebar.button("进行预测")

# ========== 4. 预测逻辑（带预处理） ==========
if predict_button:
    st.header("预测结果")
    try:
        # ---- 4.1 按训练时同样方式做转换 ----
        # QuantileTransformer 接受 2D array，形状 (n_samples,1)
        lcd_q     = qt_lcd.transform(     np.array([[LCD]]) )
        gsa_q     = qt_gsa.transform(     np.array([[GSA]]) )
        density_q = qt_density.transform( np.array([[Density]]) )
        ktol_q    = qt_ktol.transform(    np.array([[Ktoluene]]) )

        # Box–Cox 变换
        vf_bc, _  = boxcox(np.array([Vf]), lmbda=boxcox_lambda_vf)

        # ---- 4.2 拼接成模型输入，顺序必须与训练时一致 ----
        # 假设训练时特征顺序是 [LCD, Vf, GSA, Density, Ktoluene]
        X_user = np.concatenate(
            [lcd_q, vf_bc.reshape(-1,1), gsa_q, density_q, ktol_q],
            axis=1
        )

        # ---- 4.3 预测并展示 ----
        prediction = stacking_regressor.predict(X_user)[0]
        st.success(f"预测结果：{prediction:.4f}")

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
