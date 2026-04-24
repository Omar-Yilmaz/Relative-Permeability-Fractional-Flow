import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

# System Configuration
st.set_page_config(
    page_title="Reservoir Rock Properties", 
    page_icon="🛢️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS Injection
st.markdown("""
<style>
    .reportview-container { background: #0f172a; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stMetric label { color: #94a3b8 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #f8fafc; font-weight: 600; }
    div.stButton > button { background-color: #3b82f6; color: white; border-radius: 8px; border: none; padding: 10px 24px; font-weight: 600; transition: all 0.2s; width: 100%; }
    div.stButton > button:hover { background-color: #2563eb; box-shadow: 0 4px 12px rgba(37,99,235,0.4); border: none; }
    div.stDownloadButton > button { background-color: #10b981; color: white; border-radius: 8px; font-weight: 600; width: 100%; }
    div.stDownloadButton > button:hover { background-color: #059669; box-shadow: 0 4px 12px rgba(16,185,129,0.4); }
    .section-header { font-size: 24px; font-weight: 700; color: #60a5fa; margin-top: 10px; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #334155; }
    .error-text { color: #ef4444; font-weight: bold; padding: 10px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border: 1px solid #ef4444; }
</style>
""", unsafe_allow_html=True)

# Analytical Functions
def calculate_multiphase_properties(swir, sor, krwe, kroe, nw, no, muw, muo, pe, lam, steps=100):
    sw = np.linspace(swir, 1 - sor, steps)
    
    if swir + sor >= 1.0:
        return None
        
    swe = (sw - swir) / (1 - swir - sor)
    
    # Relative Permeability
    krw = krwe * (swe ** nw)
    kro = kroe * ((1 - swe) ** no)
    
    krw = np.clip(krw, 0, krwe)
    kro = np.clip(kro, 0, kroe)
    
    # Fractional Flow
    fw = np.zeros_like(sw)
    mask = (krw > 0) | (kro > 0)
    mobility_w = krw[mask] / muw
    mobility_o = kro[mask] / muo
    fw[mask] = mobility_w / (mobility_w + mobility_o)
    
    # Capillary Pressure (Brooks-Corey)
    swe_pc = np.maximum(swe, 1e-4) # Prevent division by zero at Sw = Swir
    pc = pe * (swe_pc ** (-1/lam))
    pc = np.clip(pc, 0, pe * 100) # Cap graphical infinity
    
    return pd.DataFrame({
        'Sw': np.round(sw, 4), 
        'Krw': np.round(krw, 4), 
        'Kro': np.round(kro, 4), 
        'Fw': np.round(fw, 4),
        'Pc (psi)': np.round(pc, 2)
    })

def generate_eclipse_format(df):
    output = io.StringIO()
    output.write("-- Generated relative permeability table for reservoir simulation\n")
    output.write("-- SW    KRW      KRO      PC\n")
    output.write("SWOF\n")
    for _, row in df.iterrows():
        output.write(f"  {row['Sw']:.4f}  {row['Krw']:.4f}  {row['Kro']:.4f}  {row['Pc (psi)']:.2f}\n")
    output.write("/\n")
    return output.getvalue()

# Graphic Generators
def draw_core_schematic():
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    core = patches.Rectangle((2, 0), 6, 3, edgecolor='#60a5fa', facecolor='#1e293b', lw=2)
    ax.add_patch(core)
    
    ax.annotate('', xy=(1.5, 1.5), xytext=(0, 1.5), arrowprops=dict(facecolor='#3b82f6', edgecolor='none', width=4, headwidth=12))
    ax.text(0.5, 1.8, '$q_{in}$', ha='center', color='#f8fafc', fontsize=12, fontweight='bold')
    
    ax.annotate('', xy=(10, 1.5), xytext=(8.5, 1.5), arrowprops=dict(facecolor='#3b82f6', edgecolor='none', width=4, headwidth=12))
    ax.text(9.5, 1.8, '$q_{out}$', ha='center', color='#f8fafc', fontsize=12, fontweight='bold')
    
    np.random.seed(42)
    for _ in range(50):
        x = np.random.uniform(2.2, 7.8)
        y = np.random.uniform(0.2, 2.8)
        r = np.random.uniform(0.05, 0.2)
        circle = patches.Circle((x, y), r, facecolor='#475569', edgecolor='#94a3b8')
        ax.add_patch(circle)
        
    ax.text(5, 3.4, 'Porous Medium Flow Schematic', ha='center', color='#94a3b8', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    return fig

def draw_layer_schematics(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_alpha(0.0)
    ax1.patch.set_alpha(0.0)
    ax2.patch.set_alpha(0.0)
    
    total_h = df['Thickness (h)'].sum()
    if total_h == 0:
        return fig
        
    y_current = total_h
    colors = ['#1e293b', '#334155', '#475569', '#64748b']
    
    # Parallel
    for idx, row in df.iterrows():
        h = row['Thickness (h)']
        k = row['Permeability (k)']
        y_current -= h
        color = colors[idx % len(colors)]
        rect = patches.Rectangle((2, y_current), 6, h, edgecolor='#60a5fa', facecolor=color)
        ax1.add_patch(rect)
        ax1.text(5, y_current + h/2, f'k = {k} mD | h = {h}', ha='center', va='center', color='#f8fafc')
        
    ax1.annotate('', xy=(9, total_h/2), xytext=(1, total_h/2), arrowprops=dict(facecolor='#3b82f6', edgecolor='none', width=3, headwidth=10, alpha=0.7))
    ax1.text(5, total_h + total_h*0.08, 'Parallel Flow ($k_{arithmetic}$)', ha='center', color='#94a3b8', fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, total_h * 1.2)
    ax1.axis('off')
    
    # Series
    x_current = 2
    for idx, row in df.iterrows():
        h = row['Thickness (h)']
        k = row['Permeability (k)']
        width = h / total_h * 6
        color = colors[idx % len(colors)]
        rect = patches.Rectangle((x_current, 0), width, 4, edgecolor='#10b981', facecolor=color)
        ax2.add_patch(rect)
        ax2.text(x_current + width/2, 2, f'k={k}', ha='center', va='center', rotation=90, color='#f8fafc')
        x_current += width
        
    ax2.annotate('', xy=(9, 2), xytext=(1, 2), arrowprops=dict(facecolor='#10b981', edgecolor='none', width=3, headwidth=10, alpha=0.7))
    ax2.text(5, 4.5, 'Series Flow ($k_{harmonic}$)', ha='center', color='#94a3b8', fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5.5)
    ax2.axis('off')
    
    return fig

def apply_plotly_theme(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(30, 41, 59, 0.8)', bordercolor='#475569', borderwidth=1),
        margin=dict(l=50, r=30, t=30, b=50),
        xaxis=dict(gridcolor='#334155', zerolinecolor='#475569'),
        yaxis=dict(gridcolor='#334155', zerolinecolor='#475569')
    )
    return fig

# UI Layout
st.sidebar.title("🛢️ ResEng Dashboard")
st.sidebar.markdown("---")
tab = st.sidebar.radio("Analysis Modules", ["Multiphase Properties (Kr, Fw, Pc)", "Permeability Averaging"])

if tab == "Multiphase Properties (Kr, Fw, Pc)":
    st.markdown('<div class="section-header">Brooks-Corey Multiphase Flow & Capillary Pressure</div>', unsafe_allow_html=True)
    
    st.pyplot(draw_core_schematic(), use_container_width=True)
    
    with st.expander("⚙️ Fluid & Rock Parameters", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            swir = st.number_input("Swir", value=0.20, step=0.01, format="%.2f", help="Irreducible Water Saturation")
            nw = st.number_input("Nw", value=2.5, step=0.1, format="%.1f", help="Corey Exponent (Water)")
        with col2:
            sor = st.number_input("Sor", value=0.25, step=0.01, format="%.2f", help="Residual Oil Saturation")
            no = st.number_input("No", value=2.0, step=0.1, format="%.1f", help="Corey Exponent (Oil)")
        with col3:
            krwe = st.number_input("Krw@Sor", value=0.30, step=0.01, format="%.2f", help="Endpoint Relative Permeability (Water)")
            kroe = st.number_input("Kro@Swir", value=0.80, step=0.01, format="%.2f", help="Endpoint Relative Permeability (Oil)")
        with col4:
            muw = st.number_input("μw (cP)", value=1.0, step=0.1, format="%.1f", help="Water Viscosity")
            muo = st.number_input("μo (cP)", value=2.5, step=0.1, format="%.1f", help="Oil Viscosity")
        with col5:
            pe = st.number_input("Pe (psi)", value=2.0, step=0.5, format="%.1f", help="Capillary Entry Pressure")
            lam = st.number_input("λ (Lambda)", value=2.0, step=0.1, format="%.1f", help="Pore Size Distribution Index")

    df_data = calculate_multiphase_properties(swir, sor, krwe, kroe, nw, no, muw, muo, pe, lam)

    if df_data is None:
        st.markdown('<div class="error-text">Mathematical Boundary Error: Sum of Swir and Sor must be strictly less than 1.0. Adjustment required.</div>', unsafe_allow_html=True)
    else:
        col_chart1, col_chart2, col_chart3 = st.columns(3)
        
        with col_chart1:
            fig_kr = go.Figure()
            fig_kr.add_trace(go.Scatter(x=df_data['Sw'], y=df_data['Krw'], mode='lines', name='Krw', line=dict(color='#3b82f6', width=3)))
            fig_kr.add_trace(go.Scatter(x=df_data['Sw'], y=df_data['Kro'], mode='lines', name='Kro', line=dict(color='#10b981', width=3)))
            fig_kr.update_layout(title="Relative Permeability", xaxis_title="Water Saturation (Sw)", yaxis_title="Relative Permeability", height=400)
            st.plotly_chart(apply_plotly_theme(fig_kr), use_container_width=True)

        with col_chart2:
            fig_fw = go.Figure()
            fig_fw.add_trace(go.Scatter(x=df_data['Sw'], y=df_data['Fw'], mode='lines', name='Fw', line=dict(color='#8b5cf6', width=3), fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.1)'))
            fig_fw.update_layout(title="Fractional Flow", xaxis_title="Water Saturation (Sw)", yaxis_title="Fractional Flow (Fw)", height=400)
            st.plotly_chart(apply_plotly_theme(fig_fw), use_container_width=True)
            
        with col_chart3:
            fig_pc = go.Figure()
            fig_pc.add_trace(go.Scatter(x=df_data['Sw'], y=df_data['Pc (psi)'], mode='lines', name='Pc', line=dict(color='#f59e0b', width=3)))
            fig_pc.update_layout(title="Capillary Pressure", xaxis_title="Water Saturation (Sw)", yaxis_title="Capillary Pressure (psi)", height=400, yaxis_type="log")
            st.plotly_chart(apply_plotly_theme(fig_pc), use_container_width=True)

        with st.expander("📊 Computed Dataset & Export Definitions", expanded=False):
            col_tbl, col_dl = st.columns([3, 1])
            with col_tbl:
                st.dataframe(df_data, height=250, use_container_width=True)
            with col_dl:
                csv = df_data.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download CSV Matrix", data=csv, file_name='multiphase_properties.csv', mime='text/csv')
                
                eclipse_data = generate_eclipse_format(df_data).encode('utf-8')
                st.download_button(label="Export ECLIPSE (SWOF)", data=eclipse_data, file_name='swof_table.inc', mime='text/plain')

        with st.expander("📚 Source Equations", expanded=False):
            col_eq1, col_eq2 = st.columns(2)
            with col_eq1:
                st.markdown("**Effective Water Saturation:**")
                st.latex(r"S_{we} = \frac{S_w - S_{wir}}{1 - S_{wir} - S_{or}}")
                st.markdown("**Brooks-Corey Relative Permeability:**")
                st.latex(r"k_{rw} = k_{rw}^o (S_{we})^{n_w} \quad \text{and} \quad k_{ro} = k_{ro}^o (1 - S_{we})^{n_o}")
            with col_eq2:
                st.markdown("**Fractional Flow (Gravity/Capillary Neglected):**")
                st.latex(r"f_w = \frac{\lambda_w}{\lambda_w + \lambda_o} = \frac{\frac{k_{rw}}{\mu_w}}{\frac{k_{rw}}{\mu_w} + \frac{k_{ro}}{\mu_o}}")
                st.markdown("**Brooks-Corey Capillary Pressure:**")
                st.latex(r"P_c = P_e (S_{we})^{-1/\lambda}")

elif tab == "Permeability Averaging":
    st.markdown('<div class="section-header">Layered System Permeability Averaging</div>', unsafe_allow_html=True)
    
    st.pyplot(draw_layer_schematics(pd.DataFrame({"Thickness (h)": [10.0, 15.0, 8.0], "Permeability (k)": [50.0, 120.0, 10.0]})), use_container_width=True)
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("**Layer Configuration Data**")
        initial_data = pd.DataFrame({
            "Layer": [1, 2, 3],
            "Thickness (h)": [10.0, 15.0, 8.0],
            "Permeability (k)": [50.0, 120.0, 10.0]
        })
        
        edited_df = st.data_editor(
            initial_data, 
            num_rows="dynamic", 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Layer": st.column_config.NumberColumn("Layer ID", disabled=True),
                "Thickness (h)": st.column_config.NumberColumn("Thickness (h)", min_value=0.1, step=0.1, format="%.1f"),
                "Permeability (k)": st.column_config.NumberColumn("Permeability (k)", min_value=0.1, step=0.1, format="%.1f")
            }
        )
        
        sum_h = edited_df["Thickness (h)"].sum()
        sum_kh = (edited_df["Thickness (h)"] * edited_df["Permeability (k)"]).sum()
        sum_h_over_k = (edited_df["Thickness (h)"] / edited_df["Permeability (k)"]).sum()
        
        valid_k = edited_df[edited_df["Permeability (k)"] > 0]
        sum_h_ln_k = (valid_k["Thickness (h)"] * np.log(valid_k["Permeability (k)"])).sum()
        sum_h_valid = valid_k["Thickness (h)"].sum()

        k_arithmetic = sum_kh / sum_h if sum_h > 0 else 0
        k_harmonic = sum_h / sum_h_over_k if sum_h_over_k > 0 else 0
        k_geometric = np.exp(sum_h_ln_k / sum_h_valid) if sum_h_valid > 0 else 0

        with st.expander("📚 Source Equations", expanded=False):
            st.markdown("**Arithmetic Average (Parallel Flow):**")
            st.latex(r"k_{arithmetic} = \frac{\sum_{i=1}^{n} k_i h_i}{\sum_{i=1}^{n} h_i}")
            st.markdown("**Harmonic Average (Series Flow):**")
            st.latex(r"k_{harmonic} = \frac{\sum_{i=1}^{n} h_i}{\sum_{i=1}^{n} \frac{h_i}{k_i}}")
            st.markdown("**Geometric Average (Random Flow):**")
            st.latex(r"k_{geometric} = \exp \left( \frac{\sum_{i=1}^{n} h_i \ln(k_i)}{\sum_{i=1}^{n} h_i} \right)")

    with col2:
        st.markdown("**Calculated Flow Parameters**")
        st.metric("Arithmetic Average (Parallel Flow)", f"{k_arithmetic:.2f} mD")
        st.metric("Harmonic Average (Series Flow)", f"{k_harmonic:.2f} mD")
        st.metric("Geometric Average (Random Flow)", f"{k_geometric:.2f} mD")
