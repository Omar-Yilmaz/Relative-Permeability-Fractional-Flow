import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(page_title="Reservoir Rock Properties", layout="wide")

def calculate_corey(swir, sor, krwe, kroe, nw, no, muw, muo, steps=50):
    sw = np.linspace(swir, 1 - sor, steps)
    swe = (sw - swir) / (1 - swir - sor)

    krw = krwe * (swe ** nw)
    kro = kroe * ((1 - swe) ** no)

    krw = np.clip(krw, 0, krwe)
    kro = np.clip(kro, 0, kroe)

    fw = np.zeros_like(sw)
    mask = (krw > 0) | (kro > 0)
    mobility_w = krw[mask] / muw
    mobility_o = kro[mask] / muo
    fw[mask] = mobility_w / (mobility_w + mobility_o)

    return pd.DataFrame({'Sw': sw, 'Krw': krw, 'Kro': kro, 'Fw': fw})

def draw_core_schematic():
    fig, ax = plt.subplots(figsize=(8, 3))

    # Core shape (2D representation of a cylinder)
    core = patches.Rectangle((2, 0), 6, 3, edgecolor='black', facecolor='#d1d5db', lw=2)
    ax.add_patch(core)

    # Flow arrows
    ax.annotate('', xy=(1, 1.5), xytext=(0, 1.5), arrowprops=dict(facecolor='#3b82f6', width=3, headwidth=10))
    ax.text(0.5, 1.7, '$q_{in}$ (Water/Oil)', ha='center', fontsize=10)

    ax.annotate('', xy=(9.5, 1.5), xytext=(8.5, 1.5), arrowprops=dict(facecolor='#3b82f6', width=3, headwidth=10))
    ax.text(9, 1.7, '$q_{out}$', ha='center', fontsize=10)

    # Rock grains and pores representation
    np.random.seed(42)
    for _ in range(30):
        x = np.random.uniform(2.2, 7.8)
        y = np.random.uniform(0.2, 2.8)
        r = np.random.uniform(0.1, 0.3)
        circle = patches.Circle((x, y), r, facecolor='#9ca3af', edgecolor='black')
        ax.add_patch(circle)

    ax.text(5, 3.2, 'Porous Rock Medium', ha='center', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 4)
    ax.axis('off')
    return fig

def draw_layer_schematics(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    total_h = df['Thickness (h)'].sum()
    if total_h == 0:
        return fig

    # 1. Parallel Flow (Arithmetic)
    y_current = total_h
    for _, row in df.iterrows():
        h = row['Thickness (h)']
        k = row['Permeability (k)']
        y_current -= h
        rect = patches.Rectangle((2, y_current), 6, h, edgecolor='black', facecolor='#e2e8f0')
        ax1.add_patch(rect)
        ax1.text(5, y_current + h/2, f'k = {k} mD\nh = {h} ft', ha='center', va='center')

    ax1.annotate('', xy=(9, total_h/2), xytext=(1, total_h/2),
                 arrowprops=dict(facecolor='#3b82f6', width=2, headwidth=8, alpha=0.5))
    ax1.text(5, total_h + total_h*0.05, 'Parallel Flow ($k_{arithmetic}$)\nFlow parallel to bedding', ha='center', fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, total_h * 1.1)
    ax1.axis('off')

    # 2. Series Flow (Harmonic)
    x_current = 2
    for _, row in df.iterrows():
        h = row['Thickness (h)'] # represents length in flow direction for series
        k = row['Permeability (k)']
        width = h / total_h * 6 # normalize width for display
        rect = patches.Rectangle((x_current, 0), width, 4, edgecolor='black', facecolor='#e2e8f0')
        ax2.add_patch(rect)
        ax2.text(x_current + width/2, 2, f'k={k}\nL={h}', ha='center', va='center', rotation=90)
        x_current += width

    ax2.annotate('', xy=(9, 2), xytext=(1, 2),
                 arrowprops=dict(facecolor='#10b981', width=2, headwidth=8, alpha=0.5))
    ax2.text(5, 4.2, 'Series Flow ($k_{harmonic}$)\nFlow perpendicular to bedding', ha='center', fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5)
    ax2.axis('off')

    return fig

# Sidebar Navigation
st.sidebar.title("ResEng Core Dashboard")
tab = st.sidebar.radio("Navigation", ["Relative Permeability & Flow", "Permeability Averaging"])

if tab == "Relative Permeability & Flow":
    st.header("Relative Permeability & Fractional Flow")

    st.subheader("Rock Shape & Flow Direction")
    st.pyplot(draw_core_schematic())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        swir = st.number_input("Swir (Irreducible Water)", value=0.20, step=0.01)
        nw = st.number_input("Nw (Water Exponent)", value=2.5, step=0.1)
    with col2:
        sor = st.number_input("Sor (Residual Oil)", value=0.25, step=0.01)
        no = st.number_input("No (Oil Exponent)", value=2.0, step=0.1)
    with col3:
        krwe = st.number_input("Krw@Sor", value=0.30, step=0.01)
        muw = st.number_input("Water Viscosity (cP)", value=1.0, step=0.1)
    with col4:
        kroe = st.number_input("Kro@Swir", value=0.80, step=0.01)
        muo = st.number_input("Oil Viscosity (cP)", value=2.5, step=0.1)

    df_corey = calculate_corey(swir, sor, krwe, kroe, nw, no, muw, muo)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Relative Permeability Curve")
        fig_kr = go.Figure()
        fig_kr.add_trace(go.Scatter(x=df_corey['Sw'], y=df_corey['Krw'], mode='lines', name='Krw (Water)', line=dict(color='blue')))
        fig_kr.add_trace(go.Scatter(x=df_corey['Sw'], y=df_corey['Kro'], mode='lines', name='Kro (Oil)', line=dict(color='green')))
        fig_kr.update_layout(xaxis_title="Water Saturation (Sw)", yaxis_title="Relative Permeability", height=400)
        st.plotly_chart(fig_kr, use_container_width=True)

    with col_chart2:
        st.subheader("Fractional Flow Curve")
        fig_fw = go.Figure()
        fig_fw.add_trace(go.Scatter(x=df_corey['Sw'], y=df_corey['Fw'], mode='lines', name='Fw', line=dict(color='purple')))
        fig_fw.update_layout(xaxis_title="Water Saturation (Sw)", yaxis_title="Fractional Flow (Fw)", height=400)
        st.plotly_chart(fig_fw, use_container_width=True)

elif tab == "Permeability Averaging":
    st.header("Permeability Averaging in Layered Systems")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Layer Properties")
        initial_data = pd.DataFrame({
            "Layer": [1, 2, 3],
            "Thickness (h)": [10.0, 15.0, 8.0],
            "Permeability (k)": [50.0, 120.0, 10.0]
        })

        edited_df = st.data_editor(initial_data, num_rows="dynamic", use_container_width=True)

        sum_h = edited_df["Thickness (h)"].sum()
        sum_kh = (edited_df["Thickness (h)"] * edited_df["Permeability (k)"]).sum()
        sum_h_over_k = (edited_df["Thickness (h)"] / edited_df["Permeability (k)"]).sum()

        # Avoid log of zero/negative and division by zero
        valid_k = edited_df[edited_df["Permeability (k)"] > 0]
        sum_h_ln_k = (valid_k["Thickness (h)"] * np.log(valid_k["Permeability (k)"])).sum()
        sum_h_valid = valid_k["Thickness (h)"].sum()

        k_arithmetic = sum_kh / sum_h if sum_h > 0 else 0
        k_harmonic = sum_h / sum_h_over_k if sum_h_over_k > 0 else 0
        k_geometric = np.exp(sum_h_ln_k / sum_h_valid) if sum_h_valid > 0 else 0

    with col2:
        st.subheader("Calculated Averages")
        st.metric("Arithmetic Average (Parallel Flow)", f"{k_arithmetic:.2f} mD")
        st.metric("Harmonic Average (Series Flow)", f"{k_harmonic:.2f} mD")
        st.metric("Geometric Average (Random Flow)", f"{k_geometric:.2f} mD")

    st.subheader("Flow Direction & Bedding Configuration")
    st.pyplot(draw_layer_schematics(edited_df))