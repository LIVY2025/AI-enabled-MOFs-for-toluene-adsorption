import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from scipy.stats import boxcox
import dill

# ========== 2. 页面配置与简介 ==========
st.set_page_config(layout="wide", page_title="Stacking 模型预测与 SHAP 可视化", page_icon="📊")
st.title("📊 Stacking 模型预测与 SHAP 可视化分析")

# ========== 0. 少量 CSS 美化 ==========
st.markdown("""
<style>
/* 背景色 */
.reportview-container {
    background-color: #F9FAFB;
}
/* 标题下划线 */
h1, h2, h3 {
    border-bottom: 2px solid #E0E0E0;
    padding-bottom: 4px;
}
/* 卡片样式 */
.card {
    background: white;
    padding: 16px;
    margin: 12px 0;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ========== 1. 加载模型和预处理器 ==========
stacking_regressor    = joblib.load("stacking_model.pkl")

qt_lcd                = joblib.load("qt_lcd.pkl")
qt_gsa                = joblib.load("qt_GSA.pkl")
qt_density            = joblib.load("qt_Density.pkl")
boxcox_lambda_kt      = joblib.load("lambda_Ktoluene.pkl")
boxcox_lambda_vf      = joblib.load("lambda_vf.pkl")

qt_TSN                = joblib.load("qt_TSN.pkl")

st.markdown("""
欢迎使用 **MOF 材料甲苯吸附能力（TSN）** 预测与可解释性分析平台。  
- **模型说明**：基于 8 个基学习器和 MLP 元学习器的 Stacking 回归，测试集 R² 可达 0.882。  
- **功能**：  
  1. 在线输入 5 个原始特征，实时获取预测结果。  
  2. 显示数据预处理（分位数/Box–Cox）前后对比。  
  3. 转换至原始 TSN，并展示 SHAP 特征贡献图。  
- **使用步骤**：点击侧边栏填写特征 → “进行预测” → 查看结果与可解释性图示。
""")

with st.expander("🔧 查看预测流程"):
    st.markdown("""
1. 用户输入原始特征  
2. 分位数/Box–Cox 预处理  
3. Stacking 模型预测 (TSN_transformed)  
4. 转换至原始 TSN  
5. SHAP 分析各层模型特征贡献  
    """)

# ========== 3. 侧边栏：特征输入 ==========
st.sidebar.header("特征输入区域")
LCD      = st.sidebar.number_input(
    "特征 LCD (6.03338–39.1106)", min_value=6.03338, max_value=39.1106,
    value=8.33119, format="%g"
)
Vf       = st.sidebar.number_input(
    "特征 Vf (0.2574–0.9182)", min_value=0.2574, max_value=0.9182,
    value=0.5726, format="%g"
)
GSA      = st.sidebar.number_input(
    "特征 GSA (204.912–7061.42)", min_value=204.912, max_value=7061.42,
    value=701.884, format="%g"
)
Density  = st.sidebar.number_input(
    "特征 Density (0.237838–2.86501)", min_value=0.237838, max_value=2.86501,
    value=1.51454, format="%g"
)
Ktoluene = st.sidebar.number_input(
    "特征 Ktoluene (0.000027383–28527.4)", min_value=0.000027383,
    max_value=28527.4, value=0.013545, format="%g"
)

predict_button = st.sidebar.button("进行预测")

# ========== 4. 预测逻辑（带预处理 + 显示转换 + 反变换） ==========
if predict_button:
    # 4.1 特征转换
    lcd_q     = qt_lcd.transform([[LCD]])[0, 0]
    gsa_q     = qt_gsa.transform([[GSA]])[0, 0]
    density_q = qt_density.transform([[Density]])[0, 0]

    # Box–Cox 变换（scipy.stats.boxcox 仅返回 transformed array）
    vf_bc   = boxcox(np.array([Vf]),      lmbda=boxcox_lambda_vf)
    ktol_bc = boxcox(np.array([Ktoluene]), lmbda=boxcox_lambda_kt)

    # 4.2 显示特征原始 vs 转换后
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔄 特征数据预处理")
    df_trans = pd.DataFrame({
        "特征":      ["LCD", "Vf", "GSA", "Density", "Ktoluene"],
        "原始值":    [LCD, Vf, GSA, Density, Ktoluene],
        "转换值": [lcd_q, vf_bc, gsa_q, density_q, ktol_bc]
    })
    st.table(df_trans)
    st.markdown('</div>', unsafe_allow_html=True)

    # 4.3 构造模型输入并显示
    X_user = df_trans["转换值"].to_numpy().reshape(1, -1)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👉 用于模型的 X_user")
    st.write(X_user)
    st.markdown('</div>', unsafe_allow_html=True)

    # 4.4 模型预测与反变换
    pred_trans = stacking_regressor.predict(X_user)[0]
    pred_orig  = qt_TSN.inverse_transform([[pred_trans]])[0, 0]
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 预测结果")
    st.markdown(f"- **TSN_transformed**: {pred_trans:.6f}  \n- **原始 TSN**: {pred_orig:.6f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ========== 5. SHAP 可视化 ==========
st.header("🔍 SHAP 可视化分析")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("基学习器")
    try:
        st.image(Image.open("summary_plot.png"), use_column_width=True)
    except FileNotFoundError:
        st.warning("未找到第一层基学习器的 SHAP 图像文件。")
with col2:
    st.subheader("元学习器")
    try:
        st.image(
            Image.open("SHAP Contribution Analysis for the Meta-Learner in the Second Layer of Stacking Regressor.png"),
            use_column_width=True
        )
    except FileNotFoundError:
        st.warning("未找到第二层元学习器的 SHAP 图像文件。")
with col3:
    st.subheader("整体模型")
    try:
        st.image(
            Image.open("Based on the overall feature contribution analysis of SHAP to the stacking model.png"),
            use_column_width=True
        )
    except FileNotFoundError:
        st.warning("未找到整体 Stacking 模型的 SHAP 图像文件。")

# ========== 6. 总结与联系 ==========
st.markdown("---")
st.header("🎯 总结与参考")
st.write("""
- 本页面实时预测 MOF 材料对甲苯的吸附能力，并通过 SHAP 提供可解释性分析。   
- 有疑问或建议，请联系：m202311485@xs.ustb.edu.cn
""")
