"""
steel_opt_with_mps.py - A version of the original code that saves the model in MPS format
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import csv
import math

# ============================================================
# PARAMETERS (EXACTLY MATCHING THE ORIGINAL CODE)
# ============================================================

MAX_TRIMLOSS_CUT = 5.5
MAX_TRIMLOSS_HRB = 0.5
MAX_TRIMLOSS_MILL = 0.5
MAX_TRIMLOSS_MILL_ALTERNATIVE = 6.5
MIN_TRIMLOSS_MILL_ALTERNATIVE = 1.5
MAX_CASTER_DECREASE = 3.23
MAX_EXTRA_INTERCASTER_WIDTH = 4.0
ALLOWABLE_INTERCASTER_WIDTH = 2.0

HEAVY_GAUGE = 0.4
MEDIUM_GAUGE = 0.123

MIN_GAUGE_ROLLER_CHANGE = 0.155
MAX_ROLLER_DEGRADATION = 100.0

COST_FOR_FORBIDDEN_GRADES = 70.0
MIN_HEAT_SIZE = 150.0
MIN_GRADE_CHANGE_PENALTY = 0.066666

COST_ROLLER_CHANGE = 13.33333
CHECK_TOLERANCE = 1e-4

PENALTY_WIDTH_DIFFERENCE = 6.66666
PENALTY_TRIM_LOSS = 1.0
PENALTY_ROLL_CHANGE = 13.33333
PENALTY_GAUGE_DECREASE = 166.66666

MAX_HEAT_WEIGHT = 175.0
MIN_HEAT_WEIGHT = 140.0

# ============================================================
# LOADING FUNCTIONS (EXACTLY MATCHING THE ORIGINAL CODE)
# ============================================================

def load_grade_costs():
    grade_cost = {}
    filename = 'grades_hash.csv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        return grade_cost
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            if not row:
                continue
            grade_to = row[0].strip()
            for i in range(1, len(header)):
                if i < len(row) and row[i] and row[i] != ".":
                    grade_cost[(header[i], grade_to)] = float(row[i])
    return grade_cost

def get_grade_cost(grade_from, grade_to, grade_cost_dict):
    if grade_from == grade_to:
        return 0.0
    cost = grade_cost_dict.get((grade_from, grade_to))
    if cost is None:
        return COST_FOR_FORBIDDEN_GRADES
    return cost * MIN_HEAT_SIZE + MIN_GRADE_CHANGE_PENALTY

class Coil:
    def __init__(self, coil_id, edge, grade, order_width, gauge, weight, length):
        self.id = coil_id
        self.edge = edge
        self.grade = grade
        self.orderWidth = order_width
        self.gauge = gauge
        self.weight = weight
        self.length = length
        self.degradation = max(1, -716.68*gauge**3 + 424.91*gauge**2 - 84.595*gauge + 6.7878)
    
    def getGaugeMin(self):
        if self.gauge >= HEAVY_GAUGE:
            return 0.5 * self.gauge
        elif self.gauge >= MEDIUM_GAUGE:
            return 0.75 * self.gauge
        return 0.9 * self.gauge
    
    def getWidthUB(self):
        if self.edge == 'M':
            return self.orderWidth + MAX_TRIMLOSS_MILL_ALTERNATIVE
        elif self.edge == 'HRB':
            return self.orderWidth + MAX_TRIMLOSS_HRB
        return self.orderWidth + MAX_TRIMLOSS_CUT

def load_instance_from_csv(filepath):
    coils = {}
    grade_cost_dict = load_grade_costs()
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            grade_str = row.get('Grade', '').strip()
            if grade_str.startswith('Grade_'):
                grade = grade_str
            else:
                try:
                    grade_num = int(grade_str)
                    grade = f"Grade_{grade_num}"
                except:
                    grade = "Grade_1"
            def parse_num(val):
                if not val:
                    return 0.0
                val = val.strip().strip('"')
                if ',' in val:
                    val = val.replace(',', '.')
                try:
                    return float(val)
                except:
                    return 0.0
            gauge = parse_num(row.get('Gauge', '0'))
            order_width = parse_num(row.get('OrderWidth', '0'))
            weight_raw = parse_num(row.get('Weight', '0'))
            if weight_raw > 100:
                weight = weight_raw
            else:
                weight = weight_raw
            length = parse_num(row.get('CoilLength', '1000'))
            edge = row.get('EdgeCode', 'C').strip().strip('"')
            coil = Coil(idx, edge, grade, order_width, gauge, weight, length)
            coils[idx] = coil
    return coils, grade_cost_dict

def build_model(coils, grade_cost_dict):
    """Build model exactly matching the original code"""
    num_casters = 2
    num_slots = len(coils) // num_casters
    casters = list(range(1, num_casters + 1))
    slots = list(range(1, num_slots + 1))
    coil_list = list(coils.keys())
    grades = list(set(coils[c].grade for c in coil_list))
    max_order_width = max(coils[c].orderWidth for c in coil_list)
    
    grade_changes = {}
    for g1 in grades:
        for g2 in grades:
            if g1 != g2:
                grade_changes[(g1, g2)] = get_grade_cost(g1, g2, grade_cost_dict)
    
    m = gp.Model("SteelccModel")
    m.setParam('Threads', 2)
    m.setParam('TimeLimit', 21600)
    m.setParam('MIPFocus', 2)
    m.setParam('Symmetry', 2)
    
    # Variables
    V = m.addVars(casters, slots, name="CastWidth")
    X = m.addVars(casters, slots, coil_list, vtype=GRB.BINARY, name="Assignation")
    P = m.addVars(slots, ub=MAX_EXTRA_INTERCASTER_WIDTH, name="WidthDiff")
    Ybar = m.addVars(slots, vtype=GRB.BINARY, name="RollerChange")
    CumDegrad = m.addVars(slots, ub=MAX_ROLLER_DEGRADATION, name="CummulativeDegradation")
    HeatChg = m.addVars(casters, slots, vtype=GRB.BINARY, name="HeatChange")
    CumHeat = m.addVars(casters, slots, lb=0, ub=MAX_HEAT_WEIGHT, name="cumtonHeat")
    GradeChange = m.addVars(casters, slots, vtype=GRB.BINARY, name="GradeChange")
    GradeChangeCost = m.addVars(casters, slots, lb=0, name="GradeChangeCost")
    GaugeJump = m.addVars(casters, slots, name="GaugeJump")
    
    # Constraints (same as original code)
    # Width reduction constraints
    for k in casters:
        for s in range(1, num_slots):
            m.addConstr(V[k, s] >= V[k, s+1], name=f"DecWidth_{k}_{s}")
            m.addConstr(V[k, s+1] >= V[k, s] - MAX_CASTER_DECREASE, name=f"CastDec_{k}_{s}")
    
    # Inter-caster width difference constraints
    for s in slots:
        m.addConstr(V[1, s] - V[2, s] <= ALLOWABLE_INTERCASTER_WIDTH + P[s], name=f"InterDiff1_{s}")
        m.addConstr(V[2, s] - V[1, s] <= ALLOWABLE_INTERCASTER_WIDTH + P[s], name=f"InterDiff2_{s}")
    
    # Width bounds
    for k in casters:
        for s in slots:
            m.addConstr(V[k, s] >= gp.quicksum(coils[c].orderWidth * X[k, s, c] for c in coil_list), name=f"WidthLB_{k}_{s}")
            m.addConstr(V[k, s] <= gp.quicksum(coils[c].getWidthUB() * X[k, s, c] for c in coil_list), name=f"WidthUB_{k}_{s}")
    
    # Roller constraints
    m.addConstr(Ybar[1] == 1, name="RollerFirst")
    for k in casters:
        for s in slots:
            m.addConstr(Ybar[s] <= gp.quicksum(X[k, s, c] for c in coil_list if coils[c].gauge >= MIN_GAUGE_ROLLER_CHANGE), name=f"RollerMin_{k}_{s}")
    
    m.addConstr(CumDegrad[1] == gp.quicksum(coils[c].degradation * X[k, 1, c] for k in casters for c in coil_list), name="DegInit")
    for s in slots:
        m.addConstr(CumDegrad[s] >= gp.quicksum(coils[c].degradation * X[k, s, c] for k in casters for c in coil_list), name=f"DegMin_{s}")
    for s in range(2, num_slots + 1):
        m.addConstr(CumDegrad[s] >= CumDegrad[s-1] + gp.quicksum(coils[c].degradation * X[k, s, c] for k in casters for c in coil_list) - MAX_ROLLER_DEGRADATION * Ybar[s], name=f"DegCum_{s}")
    
    n_rollers = math.ceil(sum(coils[c].degradation for c in coil_list) / MAX_ROLLER_DEGRADATION)
    m.addConstr(gp.quicksum(Ybar[s] for s in slots) >= n_rollers, name="MinRollers")
    
    # Assignment constraints
    for c in coil_list:
        m.addConstr(gp.quicksum(X[k, s, c] for k in casters for s in slots) == 1, name=f"Assign1_{c}")
    for k in casters:
        for s in slots:
            m.addConstr(gp.quicksum(X[k, s, c] for c in coil_list) == 1, name=f"Assign2_{k}_{s}")
    
    # Gauge jump constraints
    for k in casters:
        for s in range(1, num_slots):
            m.addConstr(GaugeJump[k, s] >= 25000 * (
                gp.quicksum(coils[c].getGaugeMin() * X[k, s, c] for c in coil_list) -
                gp.quicksum(coils[cc].gauge * X[k, s+1, cc] for cc in coil_list)
            ), name=f"GaugeJump_{k}_{s}")
    
    # Grade change constraints
    for k in casters:
        for s in range(1, num_slots):
            for g in grades:
                m.addConstr(GradeChange[k, s] >= (
                    gp.quicksum(X[k, s, c] for c in coil_list if coils[c].grade == g) +
                    gp.quicksum(X[k, s+1, cc] for cc in coil_list if coils[cc].grade != g) - 1
                ), name=f"GradeDet_{k}_{s}_{g}")
    
    for k in casters:
        for s in range(1, num_slots):
            for (g1, g2), cost in grade_changes.items():
                m.addConstr(GradeChangeCost[k, s] >= cost * (
                    gp.quicksum(X[k, s, c] for c in coil_list if coils[c].grade == g1) +
                    gp.quicksum(X[k, s+1, cc] for cc in coil_list if coils[cc].grade == g2) - 1
                ), name=f"GradeCost_{k}_{s}_{g1}_{g2}")
    
    # Heat constraints
    for k in casters:
        m.addConstr(CumHeat[k, 1] == gp.quicksum(coils[c].weight * (1/2000) * X[k, 1, c] for c in coil_list), name=f"HeatInit_{k}")
    for k in casters:
        for s in range(2, num_slots + 1):
            t_s = gp.quicksum(coils[c].weight * (1/2000) * X[k, s, c] for c in coil_list)
            m.addConstr(CumHeat[k, s] >= CumHeat[k, s-1] + t_s - MAX_HEAT_WEIGHT * HeatChg[k, s], name=f"HeatLow_{k}_{s}")
            m.addConstr(CumHeat[k, s] <= CumHeat[k, s-1] + t_s, name=f"HeatUp_{k}_{s}")
            m.addConstr(CumHeat[k, s] <= t_s + MAX_HEAT_WEIGHT * (1 - HeatChg[k, s]), name=f"HeatReset_{k}_{s}")
            m.addConstr(CumHeat[k, s] + (MAX_HEAT_WEIGHT - CumHeat[k, s-1]) >= t_s, name=f"HeatIntermix_{k}_{s}")
            m.addConstr(MIN_HEAT_WEIGHT * HeatChg[k, s] <= CumHeat[k, s-1], name=f"HeatMin_{k}_{s}")
            m.addConstr(CumHeat[k, s-1] <= MAX_HEAT_WEIGHT + CumHeat[k, s] - t_s, name=f"HeatMax_{k}_{s}")
    
    for k in casters:
        for s in range(1, num_slots):
            m.addConstr(GradeChange[k, s] <= HeatChg[k, s+1], name=f"GradeHeat_{k}_{s}")
            t_s1 = gp.quicksum(coils[c].weight * (1/2000) * X[k, s+1, c] for c in coil_list)
            m.addConstr(CumHeat[k, s+1] >= t_s1 - MAX_HEAT_WEIGHT * (1 - GradeChange[k, s]), name=f"GradeTon_{k}_{s}")
    
    # Objective function
    grade_obj = gp.quicksum(GradeChangeCost[k, s] for k in casters for s in slots)
    gauge_obj = gp.quicksum(GaugeJump[k, s] for k in casters for s in slots)
    width_obj = gp.quicksum(PENALTY_WIDTH_DIFFERENCE * P[s] for s in slots)
    avg_length = sum(coils[c].length for c in coil_list) / len(coil_list)
    trim_obj = gp.quicksum(
        PENALTY_TRIM_LOSS * (V[k, s] - gp.quicksum(coils[c].orderWidth * X[k, s, c] for c in coil_list)) *
        3.5433 * avg_length * 0.2817929 * (1/2000)
        for k in casters for s in slots
    )
    roller_obj = gp.quicksum(PENALTY_ROLL_CHANGE * Ybar[s] for s in slots)
    
    m.setObjective(grade_obj + gauge_obj + width_obj + trim_obj + roller_obj, GRB.MINIMIZE)
    m.update()
    
    return m, X, V, Ybar, HeatChg

# ============================================================
# MAIN FUNCTION WITH MPS SAVING
# ============================================================

def main():
    DATA_DIR = r"D:\GitProjects\OTH\COA\steel\TorresEtAl2023\Instances"
    INSTANCE = "small3"
    
    # Find instance file
    instance_path = None
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower() == f"{INSTANCE.lower()}.csv":
                instance_path = os.path.join(root, f)
                break
        if instance_path:
            break
    
    if not instance_path:
        print(f"Instance {INSTANCE} not found")
        return
    
    print(f"Loading: {instance_path}")
    coils, grade_cost_dict = load_instance_from_csv(instance_path)
    print(f"Loaded {len(coils)} coils")
    
    # Build model
    print("\nBuilding model...")
    model, X, V, Ybar, HeatChg = build_model(coils, grade_cost_dict)
    
    print(f"Model has {model.NumVars} variables, {model.NumConstrs} constraints")
    
    # ============================================================
    # Step 1: Save model to MPS file
    # ============================================================
    mps_filename = "steel_model.mps"
    print(f"\n📁 Saving model to MPS file: {mps_filename}")
    model.write(mps_filename)
    print(f"✅ Model saved successfully to {mps_filename}")
    
    # ============================================================
    # Step 2: Read back from MPS file (optional - for verification)
    # ============================================================
    print(f"\n📖 Reading model back from MPS file...")
    model_from_mps = gp.read(mps_filename)
    print(f"✅ Model read successfully from MPS file")
    print(f"   Read model has {model_from_mps.NumVars} variables, {model_from_mps.NumConstrs} constraints")
    
    # ============================================================
    # Step 3: Solve model (on CPU - Gurobi optimizes automatically)
    # ============================================================
    print(f"\n🚀 Solving model...")
    model_from_mps.optimize()
    
    if model_from_mps.Status == GRB.OPTIMAL:
        print(f"\n✅ Optimal solution found!")
        print(f"   Objective value: {model_from_mps.ObjVal:.2f}")
        print(f"   MIP gap: {model_from_mps.MIPGap*100:.2f}%")
        print(f"   Solve time: {model_from_mps.Runtime:.2f} seconds")
    elif model_from_mps.Status == GRB.INFEASIBLE:
        print("❌ Model is infeasible")
    else:
        print(f"❌ Solver status: {model_from_mps.Status}")
    
    # Display some basic information
    print("\n📊 Model statistics:")
    print(f"   Variable count: {model_from_mps.NumVars}")
    print(f"   Constraint count: {model_from_mps.NumConstrs}")
    print(f"   Non-zero coefficients: {model_from_mps.NumNZs}")

if __name__ == "__main__":
    main()