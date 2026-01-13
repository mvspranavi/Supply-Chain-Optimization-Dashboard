# optimizer_agent.py
import os
import json
import pandas as pd
import pulp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence

# --------------------------
# Multi-Echelon Supply Chain Optimizer (Suppliers → Plants → Warehouses → Customers)
# --------------------------

def load_csv_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def load_all_data(data_dir="./data"):
    """
    Expected CSV files in data_dir:
    - suppliers.csv (supplier_id,location,capacity,cost_per_unit)
    - plants.csv (plant_id,location,capacity,prod_cost_per_unit)
    - warehouses.csv (warehouse_id,location,storage_cost_per_unit,capacity)
    - demand.csv (customer_id,location,demand)
    - transport_costs.csv (from_id,to_id,cost_per_unit,lead_time_days)
    """
    suppliers_df = load_csv_safe(os.path.join(data_dir, "suppliers.csv"))
    plants_df = load_csv_safe(os.path.join(data_dir, "plants.csv"))
    warehouses_df = load_csv_safe(os.path.join(data_dir, "warehouses.csv"))
    demand_df = load_csv_safe(os.path.join(data_dir, "demand.csv"))
    transport_df = load_csv_safe(os.path.join(data_dir, "transport_costs.csv"))
    return suppliers_df, plants_df, warehouses_df, demand_df, transport_df

def build_data_structures(suppliers_df, plants_df, warehouses_df, demand_df, transport_df):
    # Maps and helpers
    suppliers = {row["supplier_id"]: {"capacity": float(row["capacity"]), "cost_per_unit": float(row["cost_per_unit"])}
                for _, row in suppliers_df.iterrows()}
    plants = {row["plant_id"]: {"capacity": float(row["capacity"]), "prod_cost_per_unit": float(row["prod_cost_per_unit"])}
            for _, row in plants_df.iterrows()}
    warehouses = {row["warehouse_id"]: {"capacity": float(row["capacity"]), "storage_cost_per_unit": float(row["storage_cost_per_unit"])}
                for _, row in warehouses_df.iterrows()}
    demand = {row["customer_id"]: float(row["demand"]) for _, row in demand_df.iterrows()}

    # Build transport arcs dict: (from -> {to: transport_cost})
    transport = {}
    for _, row in transport_df.iterrows():
        src = row["from_id"]
        dst = row["to_id"]
        cost = float(row["cost_per_unit"])
        if src not in transport:
            transport[src] = {}
        transport[src][dst] = cost

    return suppliers, plants, warehouses, demand, transport

def optimize_multi_echelon(suppliers, plants, warehouses, demand, transport):
    """
    Build LP:
    - Variables: x_{i,j} >= 0 for each arc in transport (from i to j)
    - Objective: minimize sum(transport_cost * x) + supplier_cost * outflow_from_supplier
                 + plant_prod_cost * outflow_from_plant + warehouse_storage_cost * inflow_to_warehouse
    - Constraints:
      * Supplier outflow <= supplier capacity
      * Plant outflow <= plant capacity
      * Warehouse inflow <= warehouse capacity
      * Customer inflow >= demand
      * Flow conservation-ish: for plants and warehouses, inflow >= outflow (can't produce/ship more than received)
        (This is a simple material-balance assumption; can be tightened/modified later)
    """

    # create LP
    prob = pulp.LpProblem("MultiEchelonSCCostMin", pulp.LpMinimize)

    # create decision vars for each available arc in transport
    x = {}
    for src, dests in transport.items():
        for dst in dests:
            var_name = f"x_{src}_{dst}"
            x[(src, dst)] = pulp.LpVariable(var_name, lowBound=0, cat="Continuous")

    # Objective parts
    # 1) transport cost
    transport_cost_term = pulp.lpSum([transport[(a)][b] * x[(a, b)] for (a, b) in x])

    # 2) supplier cost: for arcs where src is a supplier, supplier cost_per_unit * outflow
    supplier_cost_term = pulp.lpSum([
        suppliers[src]["cost_per_unit"] * x[(src, dst)]
        for (src, dst) in x if src in suppliers
    ])

    # 3) plant production cost: apply prod_cost_per_unit to outflow from plant
    plant_cost_term = pulp.lpSum([
        plants[src]["prod_cost_per_unit"] * x[(src, dst)]
        for (src, dst) in x if src in plants
    ])

    # 4) warehouse storage cost: apply storage cost to inflow to warehouse
    warehouse_storage_term = pulp.lpSum([
        warehouses[dst]["storage_cost_per_unit"] * x[(src, dst)]
        for (src, dst) in x if dst in warehouses
    ])

    prob += transport_cost_term + supplier_cost_term + plant_cost_term + warehouse_storage_term, "Total_Cost"

    # Constraints
    # Supplier capacity
    for s in suppliers:
        out_vars = [x[(a, b)] for (a, b) in x if a == s]
        if out_vars:
            prob += pulp.lpSum(out_vars) <= suppliers[s]["capacity"], f"SupCap_{s}"

    # Plant capacity and flow balance (inflow >= outflow and outflow <= plant capacity)
    for p in plants:
        in_vars = [x[(a, b)] for (a, b) in x if b == p]
        out_vars = [x[(a, b)] for (a, b) in x if a == p]
        if out_vars:
            prob += pulp.lpSum(out_vars) <= plants[p]["capacity"], f"PlantCap_{p}"
        if in_vars and out_vars:
            # can't ship out more than what arrived (simple material balance)
            prob += pulp.lpSum(out_vars) <= pulp.lpSum(in_vars), f"PlantFlowBal_{p}"

    # Warehouse capacity (inflow <= capacity) and flow balance (inflow >= outflow)
    for w in warehouses:
        in_vars = [x[(a, b)] for (a, b) in x if b == w]
        out_vars = [x[(a, b)] for (a, b) in x if a == w]
        if in_vars:
            prob += pulp.lpSum(in_vars) <= warehouses[w]["capacity"], f"WareCap_{w}"
        if in_vars and out_vars:
            prob += pulp.lpSum(in_vars) >= pulp.lpSum(out_vars), f"WareFlowBal_{w}"

    # Demand satisfaction: inflow to customer >= demand
    for c in demand:
        in_vars = [x[(a, b)] for (a, b) in x if b == c]
        if in_vars:
            prob += pulp.lpSum(in_vars) >= demand[c], f"Demand_{c}"
        else:
            # No inbound arcs to this customer -> infeasible in reality; add 0>=demand to cause infeasibility
            prob += 0 >= demand[c], f"Demand_no_arcs_{c}"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Gather solution
    shipments = []
    for (a, b), var in x.items():
        val = var.varValue if var.varValue is not None else 0.0
        if val > 1e-9:
            shipments.append({
                "from": a,
                "to": b,
                "units": float(val),
                "transport_unit_cost": float(transport[a][b]),
                "transport_cost": float(val * transport[a][b])
            })

    # Compute cost breakdown explicitly (redundant with objective but clearer)
    transport_cost_total = sum(s["transport_cost"] for s in shipments)
    supplier_cost_total = sum(
        s["units"] * suppliers[s["from"]]["cost_per_unit"] for s in shipments if s["from"] in suppliers
    )
    plant_prod_cost_total = sum(
        s["units"] * plants[s["from"]]["prod_cost_per_unit"] for s in shipments if s["from"] in plants
    )
    warehouse_storage_total = sum(
        s["units"] * warehouses[s["to"]]["storage_cost_per_unit"] for s in shipments if s["to"] in warehouses
    )

    total_cost = float(pulp.value(prob.objective)) if pulp.value(prob.objective) is not None else (
        transport_cost_total + supplier_cost_total + plant_prod_cost_total + warehouse_storage_total
    )

    breakdown = {
        "transport_cost": transport_cost_total,
        "supplier_cost": supplier_cost_total,
        "plant_production_cost": plant_prod_cost_total,
        "warehouse_storage_cost": warehouse_storage_total,
        "total_cost": total_cost
    }

    status = pulp.LpStatus[prob.status] if hasattr(prob, "status") else "Unknown"

    return {
        "status": status,
        "shipments": shipments,
        "breakdown": breakdown
    }, total_cost

def explain_with_groq(run_summary: dict, prompt_extra=""):
    """
    Ask Groq LLM for a short executive summary and 3 action suggestions.
    If Groq is not available or errors, return a simple fallback explanation.
    """
    try:
        groq = ChatGroq(temperature=0.2, model="llama-3.1-8b-instant")
        template = """
You are a supply-chain cost analyst. Given the optimization summary (JSON), produce:
1) A 2-sentence executive summary of the cost outcome.
2) Top 3 concrete cost-reduction suggestions.
3) One-line sensitivity checks to try next.
Return a JSON object with keys: executive_summary, suggestions (list), sensitivity (list).
OptimizationSummary:
{run_summary}
{prompt_extra}
"""
        prompt = PromptTemplate.from_template(template)
        parser = StrOutputParser()
        chain = RunnableSequence(prompt | groq | parser)
        resp = chain.invoke({"run_summary": json.dumps(run_summary), "prompt_extra": prompt_extra})
        # try to parse JSON
        try:
            return json.loads(resp)
        except Exception:
            return {"raw": resp}
    except Exception as e:
        # fallback explanation
        return {
            "executive_summary": "AI explanation unavailable: " + str(e),
            "suggestions": [],
            "sensitivity": []
        }

def run_pipeline(user_input, data_dir="./data"):
    """
    Top-level pipeline: loads data, runs optimization, asks Groq to explain,
    and returns results dictionary used by Streamlit.
    """
    # Load CSVs
    suppliers_df, plants_df, warehouses_df, demand_df, transport_df = load_all_data(data_dir)
    suppliers, plants, warehouses, demand, transport = build_data_structures(
        suppliers_df, plants_df, warehouses_df, demand_df, transport_df
    )

    # Run optimization
    opt_res, total_cost = optimize_multi_echelon(suppliers, plants, warehouses, demand, transport)

    # Ask LLM for explanation (best-effort)
    explanation = explain_with_groq({
        "status": opt_res["status"],
        "breakdown": opt_res["breakdown"],
        "shipments": opt_res["shipments"]
    }, prompt_extra=user_input)

    # package result
    run_summary = {
        "ai_response": explanation,
        "optimization": opt_res,
        "total_cost": total_cost
    }
    return run_summary
