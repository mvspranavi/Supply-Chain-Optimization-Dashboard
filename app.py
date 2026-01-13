# app.py
import streamlit as st
import pandas as pd
import os
from optimizer_agent import run_pipeline
from dotenv import load_dotenv
from io import StringIO

load_dotenv()

st.set_page_config(page_title="SC Cost Optimizer", layout="wide", page_icon="💡")

# --- Styles (keeps your previous look; adjust if you want) ---
st.markdown("""
    <style>
        .stApp { background: linear-gradient(120deg, #dbeafe 0%, #bfdbfe 100%); font-family: 'Poppins', sans-serif; color:#0f172a; }
        h1 { text-align: center; color: #0d1b2a; font-size: 2.4em; font-weight: 700; }
        .streamlit-expanderHeader { background-color: #0d6efd !important; color: white !important; font-weight: 600 !important; font-size: 15px !important; border-radius: 8px !important; padding: 6px 10px !important; }
        .data-card { background: #ffffffcc; border-radius: 12px; padding: 14px; margin-bottom: 14px; box-shadow: 0px 4px 8px rgba(0,0,0,0.08); }
        div.stButton > button { background: linear-gradient(90deg, #0d6efd, #00b4d8); color: white; border-radius: 10px; padding: 8px 14px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

st.title("💡 Supply Chain Cost Optimization — AI-Assisted")

# ---------------- Sidebar / Controls ----------------
with st.sidebar:
    st.header("⚙️ Configuration")
    data_dir = st.text_input("Local Data Directory", value="./data")
    prompt_extra = st.text_area("Additional LLM Prompt (optional)", value="")
    if st.button("🚀 Run Optimization"):
        st.session_state["run_requested"] = True

# ---------------- Helpers: validation ----------------
def check_files_exist(data_dir):
    required = [
        "suppliers.csv",
        "plants.csv",
        "warehouses.csv",
        "demand.csv",
        "transport_costs.csv"
    ]
    missing = [f for f in required if not os.path.isfile(os.path.join(data_dir, f))]
    return missing

def validate_data_and_report(data_dir):
    """Return (ok: bool, messages: list) where ok False if critical issues found."""
    messages = []
    missing = check_files_exist(data_dir)
    if missing:
        messages.append(("error", f"Missing files: {', '.join(missing)}"))
        return False, messages

    # Load small pieces
    try:
        transport_df = pd.read_csv(os.path.join(data_dir, "transport_costs.csv"))
        demand_df = pd.read_csv(os.path.join(data_dir, "demand.csv"))
        suppliers_df = pd.read_csv(os.path.join(data_dir, "suppliers.csv"))
    except Exception as e:
        messages.append(("error", f"Failed to read CSVs: {e}"))
        return False, messages

    # Check required columns in transport
    for col in ["from_id", "to_id", "cost_per_unit"]:
        if col not in transport_df.columns:
            messages.append(("error", f"transport_costs.csv missing column: {col}"))
            return False, messages

    # Check inbound arcs for each customer
    customer_ids = demand_df.iloc[:, 0].astype(str).tolist()  # first column assumed customer_id
    inbound_map = transport_df["to_id"].astype(str).unique().tolist()
    customers_no_inbound = [c for c in customer_ids if c not in inbound_map]
    if customers_no_inbound:
        messages.append(("error", f"Customers with NO inbound arcs in transport_costs.csv: {customers_no_inbound}"))
        # Critical — the optimizer will be infeasible. Stop here.
        return False, messages

    # Supply vs demand capacity check
    # Attempt to find 'capacity' and 'demand' columns
    try:
        supply_sum = suppliers_df["capacity"].astype(float).sum()
        # demand file must have a 'demand' column (or second column)
        if "demand" in demand_df.columns:
            demand_sum = demand_df["demand"].astype(float).sum()
        else:
            # fallback: take second column if named differently
            demand_sum = demand_df.iloc[:, 1].astype(float).sum()
        if supply_sum < demand_sum:
            messages.append(("warning", f"Total supply capacity ({supply_sum}) < total demand ({demand_sum}). Solver may be infeasible or unsatisfied demand will occur."))
    except Exception:
        # If parsing failed, report a warning but allow run
        messages.append(("warning", "Could not validate numeric supply/demand sums (check CSV column names/types)."))

    # Basic sanity: transport costs non-negative?
    if (transport_df["cost_per_unit"].astype(float) < 0).any():
        messages.append(("warning", "Negative transport cost found — please verify data."))

    # All checks passed (or only warnings)
    return True, messages

# ---------------- Data preview ----------------
st.markdown("## 📊 Preview Data Files")
def show_csv(file_name):
    try:
        df = pd.read_csv(os.path.join(data_dir, file_name))
        with st.expander(f"📘 {file_name}", expanded=False):
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Could not load {file_name}: {e}")

for fn in ["suppliers.csv", "plants.csv", "warehouses.csv", "demand.csv", "transport_costs.csv"]:
    show_csv(fn)

# ---------------- Run pipeline with validation and improved results ----------------
if st.session_state.get("run_requested", False):
    # Reset run_requested so subsequent edits require clicking again
    st.session_state["run_requested"] = False

    ok, messages = validate_data_and_report(data_dir)
    # show messages
    for level, msg in messages:
        if level == "error":
            st.error(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.info(msg)

    if not ok:
        st.error("Critical data issues detected. Fix the CSVs and run again.")
    else:
        # Run the pipeline
        with st.spinner("🔄 Running optimization pipeline..."):
            try:
                # call backend with data_dir so it reads CSVs
                res = run_pipeline(prompt_extra, data_dir)
            except Exception as e:
                st.exception(f"Pipeline failed: {e}")
                res = None

        if res:
            st.success("✅ Run complete!", icon="🎯")
            st.markdown("---")

            # Unpack results (keeps your variable names)
            opt = res["optimization"]               # dict with status, shipments, breakdown
            total_cost = res.get("total_cost", None)
            ai_explanation = res.get("ai_response", {})

            status = opt.get("status", "Unknown")
            breakdown = opt.get("breakdown", {})
            shipments = opt.get("shipments", [])

            # Status banner
            if status.lower() != "optimal" and status.lower() != "optimal":
                st.warning(f"Solver status: {status}")

            # Cost breakdown cards
            st.subheader("📉 Cost Breakdown")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Transport Cost", f"${breakdown.get('transport_cost', 0):,.2f}")
            col2.metric("Supplier Cost", f"${breakdown.get('supplier_cost', 0):,.2f}")
            col3.metric("Plant Production Cost", f"${breakdown.get('plant_production_cost', 0):,.2f}")
            col4.metric("Warehouse Storage Cost", f"${breakdown.get('warehouse_storage_cost', 0):,.2f}")

            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.write("### 💰 Total Optimized Cost")
            if total_cost is not None:
                st.write(f"**${total_cost:,.2f}**")
            else:
                st.write("N/A")
            st.markdown("</div>", unsafe_allow_html=True)

            # Shipments table
            st.markdown("### 🚚 Shipments (non-zero)")
            if shipments:
                df_ship = pd.DataFrame(shipments)
                # Show columns nicely and add columns for per-route supplier/plant/warehouse cost if possible
                # (we keep original columns: from,to,units,transport_unit_cost,transport_cost)
                st.dataframe(df_ship, use_container_width=True)

                # Provide CSV download
                csv_buf = df_ship.to_csv(index=False)
                st.download_button("📥 Download Shipments CSV", csv_buf, file_name="shipments_results.csv", mime="text/csv")
            else:
                st.info("No shipments returned (check data and solver status).")

            # AI explanation / suggestions
            st.subheader("🧠 AI Explanation / Suggestions")

            if isinstance(ai_explanation, dict):
                parts = []

                if "executive_summary" in ai_explanation:
                    parts.append(f"### 🧭 Executive Summary\n{ai_explanation['executive_summary']}")

                if "suggestions" in ai_explanation:
                    suggestions_md = "\n".join([f"- {s}" for s in ai_explanation["suggestions"]])
                    parts.append(f"### 💡 Top Suggestions\n{suggestions_md}")

                if "sensitivity" in ai_explanation:
                    sensitivity_md = "\n".join([f"- {s}" for s in ai_explanation["sensitivity"]])
                    parts.append(f"### 📊 Sensitivity Ideas\n{sensitivity_md}")

                if "raw" in ai_explanation and len(ai_explanation) == 1:
                    parts.append(ai_explanation["raw"])

                st.markdown("\n\n".join(parts), unsafe_allow_html=True)
            else:
                st.markdown(str(ai_explanation), unsafe_allow_html=True)


            # Optional: show raw optimization JSON for debugging
            with st.expander("Show raw optimization JSON"):
                st.json(opt)
#source venv/bin/activate
#export GROQ_API_KEY="Your_api_key"
#streamlit run app.py