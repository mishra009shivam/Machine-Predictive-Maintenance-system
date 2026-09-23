"""
app/app.py
──────────────────────────────────────────────────────────────────────────────
Streamlit Web Dashboard for SC08 — Machine Predictive Maintenance & Fault Priority System.

Features:
  - Input controls for raw machine operating conditions
  - Real-time prediction via pre-trained NumPy ANN and Mamdani Fuzzy Logic System
  - Visual severity gauge & color-coded priority badges
  - Rule activation & explainability breakdown
  - Preset operational scenario selector for quick viva testing
  - Compact expandable Project Information & System Pipeline panels
"""

import os
import sys

# Ensure project root is in sys.path for direct streamlit execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.prediction import PredictiveMaintenancePipeline


# ==============================================================================
# Page Configuration & Custom CSS Styling (Dark Industrial Theme)
# ==============================================================================

st.set_page_config(
    page_title="SC08 Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark industrial theme with clean typography and rounded cards
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .assessment-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .priority-badge-LOW {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.25rem;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .priority-badge-MEDIUM {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.25rem;
        display: inline-block;
        border: 1px solid #FDE047;
    }
    .priority-badge-HIGH {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.25rem;
        display: inline-block;
        border: 1px solid #FDBA74;
    }
    .priority-badge-URGENT {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.25rem;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
    .recommendation-box {
        background-color: #F1F5F9;
        border-left: 5px solid #0EA5E9;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-top: 1rem;
        font-size: 0.98rem;
        color: #1E293B;
    }
    .flow-step-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        padding: 10px 16px;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        color: #1E293B;
        font-size: 0.95rem;
        margin: 4px 0;
    }
    .flow-arrow {
        text-align: center;
        color: #0EA5E9;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 2px 0;
    }
    .footer-text {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        padding: 1.5rem 0 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# Cached Pipeline Loader
# ==============================================================================

@st.cache_resource
def load_pipeline():
    """Instantiate and cache the PredictiveMaintenancePipeline."""
    return PredictiveMaintenancePipeline()


# ==============================================================================
# Benchmark Test Scenarios
# ==============================================================================

PRESET_SCENARIOS = {
    "Custom User Input": None,
    "Scenario 1: Normal Machine (Low Risk)": {
        "product_type": "L", "air_temp_K": 300.0, "process_temp_K": 310.0,
        "rotational_speed_rpm": 1500.0, "torque_Nm": 40.0, "tool_wear_min": 20.0
    },
    "Scenario 2: Moderate Risk Machine": {
        "product_type": "M", "air_temp_K": 301.0, "process_temp_K": 311.0,
        "rotational_speed_rpm": 1400.0, "torque_Nm": 48.0, "tool_wear_min": 130.0
    },
    "Scenario 3: High ANN Probability (Elevated Risk)": {
        "product_type": "H", "air_temp_K": 303.0, "process_temp_K": 313.0,
        "rotational_speed_rpm": 1200.0, "torque_Nm": 65.0, "tool_wear_min": 50.0
    },
    "Scenario 4: High Tool Wear Machine": {
        "product_type": "L", "air_temp_K": 299.0, "process_temp_K": 309.0,
        "rotational_speed_rpm": 1550.0, "torque_Nm": 38.0, "tool_wear_min": 225.0
    },
    "Scenario 5: Critical Emergency Condition": {
        "product_type": "M", "air_temp_K": 304.0, "process_temp_K": 314.0,
        "rotational_speed_rpm": 1180.0, "torque_Nm": 68.0, "tool_wear_min": 235.0
    },
}


# ==============================================================================
# Main Streamlit Application
# ==============================================================================

def main():
    # Title & Subtitle Header
    st.markdown('<div class="main-header">SC08 — Machine Predictive Maintenance & Fault Priority System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Hybrid Artificial Neural Network (ANN) + Mamdani Fuzzy Logic Decision System</div>', unsafe_allow_html=True)

    # Load Pipeline
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Failed to load prediction pipeline: {e}")
        st.stop()

    # Sidebar: Input Form & Presets
    st.sidebar.header("⚙️ Machine Operating Parameters")

    preset_choice = st.sidebar.selectbox(
        "Load Test Scenario Preset",
        options=list(PRESET_SCENARIOS.keys()),
        index=0
    )

    # Fill defaults from preset if selected
    preset_data = PRESET_SCENARIOS[preset_choice]

    with st.sidebar.form("input_form"):
        st.subheader("Raw Sensor Readings")

        product_type = st.selectbox(
            "Product Type (Quality Variant)",
            options=["L", "M", "H"],
            index=["L", "M", "H"].index(preset_data["product_type"]) if preset_data else 0,
            help="L = Low quality variant, M = Medium quality, H = High quality variant"
        )

        air_temp = st.number_input(
            "Air Temperature [K]",
            min_value=250.0, max_value=350.0,
            value=preset_data["air_temp_K"] if preset_data else 300.0,
            step=0.5,
            help="Surrounding ambient temperature in Kelvin"
        )

        process_temp = st.number_input(
            "Process Temperature [K]",
            min_value=250.0, max_value=350.0,
            value=preset_data["process_temp_K"] if preset_data else 310.0,
            step=0.5,
            help="Operating machine process temperature in Kelvin"
        )

        speed_rpm = st.number_input(
            "Rotational Speed [rpm]",
            min_value=0.0, max_value=5000.0,
            value=preset_data["rotational_speed_rpm"] if preset_data else 1500.0,
            step=10.0,
            help="Spindle rotational speed in revolutions per minute"
        )

        torque_nm = st.number_input(
            "Torque [Nm]",
            min_value=0.0, max_value=150.0,
            value=preset_data["torque_Nm"] if preset_data else 40.0,
            step=1.0,
            help="Spindle load torque in Newton-metres (NOT vibration)"
        )

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0.0, max_value=300.0,
            value=preset_data["tool_wear_min"] if preset_data else 20.0,
            step=1.0,
            help="Cumulative cutting tool wear duration in minutes"
        )

        submit_btn = st.form_submit_button("🔍 Run Prediction Pipeline", type="primary", use_container_width=True)

    # ── LIVE MACHINE ASSESSMENT SECTION ─────────────────────────────────────
    st.subheader("📊 Live Machine Assessment")

    inputs_to_run = {
        "product_type": product_type,
        "air_temp_K": air_temp,
        "process_temp_K": process_temp,
        "rotational_speed_rpm": speed_rpm,
        "torque_Nm": torque_nm,
        "tool_wear_min": tool_wear,
    }

    # Execute Prediction
    result = None
    try:
        result = pipeline.predict(**inputs_to_run)
    except (ValueError, TypeError) as e:
        st.error(f"Input Validation Error: {e}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")

    if result is not None:
        # Core Assessment Metrics Cards
        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            st.metric(
                label="ANN Fault Probability P(Failure)",
                value=f"{result['fault_probability'] * 100:.2f}%",
                delta=f"Prob: {result['fault_probability']:.4f}"
            )

        with col_m2:
            st.metric(
                label="Fuzzy Severity Index",
                value=f"{result['severity_index']:.2f} / 100",
                delta=f"Level: {result['priority']}"
            )

        with col_m3:
            priority = result["priority"]
            st.write("**Maintenance Priority:**")
            st.markdown(f"<span class='priority-badge-{priority}'>{priority}</span>", unsafe_allow_html=True)

        # Severity Gauge / Progress Indicator
        st.write("**Severity Visual Indicator:**")
        sev_pct = min(1.0, max(0.0, result["severity_index"] / 100.0))
        st.progress(sev_pct)

        # Action Recommendation Box
        st.markdown(f"""
            <div class="recommendation-box">
                <strong>🛠️ Recommended Action:</strong><br>
                {result['recommendation']}
            </div>
        """, unsafe_allow_html=True)

        # Explainability Section
        st.markdown("---")
        st.subheader("💡 Why Did the System Make This Decision?")
        st.info(result["fuzzy_explanation"])

        # Fired Rules Detail
        st.write("**Fired Fuzzy Rules:**")
        if result["fired_rules"]:
            for r in result["fired_rules"]:
                st.markdown(f"- **Rule R{r['rule']}** `[{r['consequent']} @ strength {r['strength']:.4f}]`: {r['description']}")
        else:
            st.write("*(No specific rules fired — safe fallback severity applied)*")
    else:
        st.warning("Please correct invalid inputs in the sidebar to view prediction results.")

    # ── PROJECT INFORMATION SECTION (COMPACT EXPANDERS AT BOTTOM) ────────────
    st.markdown("---")
    st.subheader("📁 Project Information")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        with st.expander("🔄 How the System Works", expanded=False):
            st.markdown("""
            <div style="display: flex; flex-direction: column; gap: 4px; padding: 4px 0;">
                <div class="flow-step-box">Machine Sensor Data</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step-box">Preprocessing & Min-Max Normalisation</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step-box">ANN (7 → 16 → 8 → 1)</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step-box">Fault Probability P(Failure) ∈ [0, 1]</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step-box">Mamdani Fuzzy Logic Inference (NumPy)</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step-box">Severity Index [0, 100]</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step-box">Maintenance Priority (LOW / MEDIUM / HIGH / URGENT)</div>
            </div>
            """, unsafe_allow_html=True)

    with exp_col2:
        with st.expander("ℹ️ About This Project", expanded=False):
            st.markdown("""
            - **Dataset**: UCI AI4I 2020 Predictive Maintenance Dataset
            - **Dataset Size**: 10,000 operational machine records
            - **ANN Architecture**: 7 → 16 → 8 → 1
            - **ANN Implementation**: Implemented from scratch using pure **NumPy**
            - **Fuzzy Logic Implementation**: Implemented from scratch using pure **NumPy**
            - **Membership Functions**: Triangular and trapezoidal membership functions
            - **Defuzzification**: Centre of Gravity (CoG) centroid defuzzification
            - **System Architecture**: Hybrid ANN + Mamdani Fuzzy Logic architecture
            - **Torque Fidelity**: Torque is spindle load torque [Nm], not vibration
            - **Project Purpose**: University Soft Computing project & viva defense
            """)

    # ── FOOTER ───────────────────────────────────────────────────────────────
    st.markdown("""
        <div class="footer-text">
            SC08 | Machine Predictive Maintenance & Fault Priority System &bull; Hybrid ANN + Mamdani Fuzzy Logic
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
