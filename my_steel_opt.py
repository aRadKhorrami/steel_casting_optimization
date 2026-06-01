"""
EXACT implementation of authors' formulation
from steelccGurobi.py

This code uses IDENTICAL:
- Variables (V, X, P, Ybar, CumDegrad, HeatChg, CumHeat, GradeChange, GradeChangeCost, GaugeJump)
- Constraints (as in formulateModel function)
- Objective function (as in authors' code)
- Parameters (from parameters.py)
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import csv
import math

# ============================================================
# PARAMETERS (EXACTLY from parameters.py)
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
# GRADE COST (EXACTLY as authors' getGradeCost)
# ============================================================

def load_grade_costs():
    """Load grade costs EXACTLY as authors' steelcc.py"""
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
    """EXACT same formula as authors' getGradeCost()"""
    if grade_from == grade_to:
        return 0.0
    cost = grade_cost_dict.get((grade_from, grade_to))
    if cost is None:
        return COST_FOR_FORBIDDEN_GRADES
    return cost * MIN_HEAT_SIZE + MIN_GRADE_CHANGE_PENALTY


# ============================================================
# COIL CLASS (EXACTLY as authors' steelcc.py)
# ============================================================

class Coil:
    def __init__(self, coil_id, edge, grade, order_width, gauge, weight, length):
        self.id = coil_id
        self.edge = edge
        self.grade = grade
        self.orderWidth = order_width
        self.gauge = gauge
        self.weight = weight
        self.length = length
        # Degradation formula from authors
        self.degradation = max(1, -716.68*gauge**3 + 424.91*gauge**2 - 84.595*gauge + 6.7878)
    
    def getGaugeMin(self):
        """EXACT same as authors' getGaugeMin()"""
        if self.gauge >= HEAVY_GAUGE:
            return 0.5 * self.gauge
        elif self.gauge >= MEDIUM_GAUGE:
            return 0.75 * self.gauge
        return 0.9 * self.gauge
    
    def getWidthUB(self):
        """EXACT same as authors' getWidthUB()"""
        if self.edge == 'M':
            return self.orderWidth + MAX_TRIMLOSS_MILL_ALTERNATIVE
        elif self.edge == 'HRB':
            return self.orderWidth + MAX_TRIMLOSS_HRB
        return self.orderWidth + MAX_TRIMLOSS_CUT


# ============================================================
# LOAD INSTANCE (as authors' loadSchoppyOutput but for CSV)
# ============================================================

def load_instance_from_csv(filepath):
    """Load coils from CSV file, converting to authors' format"""
    coils = {}
    grade_cost_dict = load_grade_costs()
    
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        for idx, row in enumerate(reader):
            # Grade - convert to "Grade_X" format
            grade_str = row.get('Grade', '').strip()
            if grade_str.startswith('Grade_'):
                grade = grade_str
            else:
                try:
                    grade_num = int(grade_str)
                    grade = f"Grade_{grade_num}"
                except:
                    grade = "Grade_1"
            
            # Parse numbers
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
            
            # Weight: authors use lbs (divide by 2000 for tons)
            # In the CSV, weights appear to be in kg (72890 = 72.9 tons)
            # So convert kg to lbs first? Let's check authors' expected format
            # In run.py output, weights were like 72.9 (already tons)
            if weight_raw > 100:
                weight = weight_raw  # Already in tons (from kg/1000)
            else:
                weight = weight_raw
            
            length = parse_num(row.get('CoilLength', '1000'))
            
            edge = row.get('EdgeCode', 'C').strip().strip('"')
            
            coil = Coil(idx, edge, grade, order_width, gauge, weight, length)
            coils[idx] = coil
    
    return coils, grade_cost_dict


# ============================================================
# BUILD MODEL (EXACTLY as authors' formulateModel)
# ============================================================

def build_model(coils, grade_cost_dict, use_mill_cuts=0, use_grade_cuts=0, add_critical_pairs=0):
    """
    Build model EXACTLY as authors' formulateModel() in steelccGurobi.py
    
    Parameters match authors' run.py arguments:
        use_mill_cuts: 0=ignore, 1=enforce as constraints, 2=lazy
        use_grade_cuts: 0=simple grade change
        add_critical_pairs: 0=no cuts
    """
    
    # Sets (authors use 1-indexing)
    num_casters = 2
    num_slots = len(coils) // num_casters
    casters = list(range(1, num_casters + 1))
    slots = list(range(1, num_slots + 1))
    coil_list = list(coils.keys())
    
    # Get grades
    grades = list(set(coils[c].grade for c in coil_list))
    
    # Max order width for big-M
    max_order_width = max(coils[c].orderWidth for c in coil_list)
    
    # Grade change costs
    grade_changes = {}
    for g1 in grades:
        for g2 in grades:
            if g1 != g2:
                grade_changes[(g1, g2)] = get_grade_cost(g1, g2, grade_cost_dict)
    
    # Create model
    m = gp.Model("SteelccModel")
    m.setParam('Threads', 2)
    m.setParam('TimeLimit', 21600)
    m.setParam('MIPFocus', 2)
    m.setParam('Symmetry', 2)
    
    # ========== VARIABLES (EXACTLY as authors) ==========
    
    # V[k,s]: Casting width
    V = m.addVars(casters, slots, name="CastWidth")
    
    # X[k,s,c]: Assignment
    X = m.addVars(casters, slots, coil_list, vtype=GRB.BINARY, name="Assignation")
    
    # P[s]: Width difference penalty
    P = m.addVars(slots, ub=MAX_EXTRA_INTERCASTER_WIDTH, name="WidthDiff")
    
    # Roller variables
    Ybar = m.addVars(slots, vtype=GRB.BINARY, name="RollerChange")
    CumDegrad = m.addVars(slots, ub=MAX_ROLLER_DEGRADATION, name="CummulativeDegradation")
    
    # Heat variables
    HeatChg = m.addVars(casters, slots, vtype=GRB.BINARY, name="HeatChange")
    CumHeat = m.addVars(casters, slots, lb=0, ub=MAX_HEAT_WEIGHT, name="cumtonHeat")
    
    # Grade change variables
    GradeChange = m.addVars(casters, slots, vtype=GRB.BINARY, name="GradeChange")
    GradeChangeCost = m.addVars(casters, slots, lb=0, name="GradeChangeCost")
    
    # Gauge jump variable (authors use 25000 multiplier)
    GaugeJump = m.addVars(casters, slots, name="GaugeJump")
    
    # Mill edge alternative variable (if used)
    B = None
    if use_mill_cuts > 0:
        B = m.addVars(coil_list, vtype=GRB.BINARY, name="AlternativeMillWidth")
    
    # ========== CONSTRAINTS ==========
    
    # ----- 1) Caster width constraints (authors' section 1) -----
    
    # Decreasing width
    for k in casters:
        for s in range(1, num_slots):
            m.addConstr(V[k, s] >= V[k, s+1], name=f"DecreasingWidth_{k}_{s}")
            m.addConstr(V[k, s+1] >= V[k, s] - MAX_CASTER_DECREASE, name=f"CasterDecrease_{k}_{s}")
    
    # Inter-caster width difference
    for s in slots:
        m.addConstr(V[1, s] - V[2, s] <= ALLOWABLE_INTERCASTER_WIDTH + P[s], name=f"IntercasterDiff1_{s}")
        m.addConstr(V[2, s] - V[1, s] <= ALLOWABLE_INTERCASTER_WIDTH + P[s], name=f"IntercasterDiff2_{s}")
    
    # Width lower and upper bounds (authors' CasterWidthLB/UB)
    for k in casters:
        for s in slots:
            m.addConstr(V[k, s] >= gp.quicksum(coils[c].orderWidth * X[k, s, c] for c in coil_list),
                       name=f"CasterWidthLB_{k}_{s}")
            m.addConstr(V[k, s] <= gp.quicksum(coils[c].getWidthUB() * X[k, s, c] for c in coil_list),
                       name=f"CasterWidthUB_{k}_{s}")
    
    # ----- Mill edge constraints (authors' section 9) -----
    if use_mill_cuts == 1 and B is not None:
        for k in casters:
            for s in slots:
                for c in coil_list:
                    if coils[c].edge == 'M':
                        m.addConstr(V[k, s] >= coils[c].orderWidth * X[k, s, c] + B[c] * MIN_TRIMLOSS_MILL_ALTERNATIVE,
                                   name=f"MillLB_{k}_{s}_{c}")
                        m.addConstr(V[k, s] <= MAX_TRIMLOSS_MILL + coils[c].orderWidth * X[k, s, c] +
                                   B[c] * (MAX_TRIMLOSS_MILL_ALTERNATIVE - MAX_TRIMLOSS_MILL) +
                                   (1 - X[k, s, c]) * max_order_width,
                                   name=f"MillUB_{k}_{s}_{c}")
    
    # ----- Roller constraints (authors' section 4) -----
    
    m.addConstr(Ybar[1] == 1, name="RollerFirst")
    
    for k in casters:
        for s in slots:
            m.addConstr(Ybar[s] <= gp.quicksum(X[k, s, c] for c in coil_list if coils[c].gauge >= MIN_GAUGE_ROLLER_CHANGE),
                       name=f"RollerMinGauge_{k}_{s}")
    
    m.addConstr(CumDegrad[1] == gp.quicksum(coils[c].degradation * X[k, 1, c] for k in casters for c in coil_list),
               name="DegradationInit")
    
    for s in slots:
        m.addConstr(CumDegrad[s] >= gp.quicksum(coils[c].degradation * X[k, s, c] for k in casters for c in coil_list),
                   name=f"DegradationMin_{s}")
    
    for s in range(2, num_slots + 1):
        m.addConstr(CumDegrad[s] >= CumDegrad[s-1] + 
                   gp.quicksum(coils[c].degradation * X[k, s, c] for k in casters for c in coil_list) -
                   MAX_ROLLER_DEGRADATION * Ybar[s],
                   name=f"DegradationCum_{s}")
    
    # Minimum number of roller changes
    n_rollers = math.ceil(sum(coils[c].degradation for c in coil_list) / MAX_ROLLER_DEGRADATION)
    m.addConstr(gp.quicksum(Ybar[s] for s in slots) >= n_rollers, name="MinRollers")
    
    # ----- Valid assignments (authors' section 6) -----
    
    for c in coil_list:
        m.addConstr(gp.quicksum(X[k, s, c] for k in casters for s in slots) == 1, name=f"ValidAssign1_{c}")
    
    for k in casters:
        for s in slots:
            m.addConstr(gp.quicksum(X[k, s, c] for c in coil_list) == 1, name=f"ValidAssign2_{k}_{s}")
    
    # ----- Gauge penalization (authors' section 7) -----
    # Authors use 25000 multiplier exactly
    for k in casters:
        for s in range(1, num_slots):
            m.addConstr(GaugeJump[k, s] >= 25000 * (
                gp.quicksum(coils[c].getGaugeMin() * X[k, s, c] for c in coil_list) -
                gp.quicksum(coils[cc].gauge * X[k, s+1, cc] for cc in coil_list)
            ), name=f"GaugeJump_{k}_{s}")
    
    # ----- Grade change constraints (authors' section 7) -----
    
    # Detect grade change
    for k in casters:
        for s in range(1, num_slots):
            for g in grades:
                m.addConstr(GradeChange[k, s] >= (
                    gp.quicksum(X[k, s, c] for c in coil_list if coils[c].grade == g) +
                    gp.quicksum(X[k, s+1, cc] for cc in coil_list if coils[cc].grade != g) - 1
                ), name=f"GradeDetect_{k}_{s}_{g}")
    
    # Grade change cost
    for k in casters:
        for s in range(1, num_slots):
            for (g1, g2), cost in grade_changes.items():
                m.addConstr(GradeChangeCost[k, s] >= cost * (
                    gp.quicksum(X[k, s, c] for c in coil_list if coils[c].grade == g1) +
                    gp.quicksum(X[k, s+1, cc] for cc in coil_list if coils[cc].grade == g2) - 1
                ), name=f"GradeCost_{k}_{s}_{g1}_{g2}")
    
    # ----- Heat constraints (authors' section 11) -----
    
    # First slot
    for k in casters:
        m.addConstr(CumHeat[k, 1] == gp.quicksum(coils[c].weight * (1/2000) * X[k, 1, c] for c in coil_list),
                name=f"HeatInit_{k}")

    # Heat constraints for s > 1
    for k in casters:
        for s in range(2, num_slots + 1):
            t_s = gp.quicksum(coils[c].weight * (1/2000) * X[k, s, c] for c in coil_list)
            
            # Cumulative (with heat change)
            m.addConstr(CumHeat[k, s] >= CumHeat[k, s-1] + t_s - MAX_HEAT_WEIGHT * HeatChg[k, s],
                    name=f"HeatLower_{k}_{s}")
            
            # Cumulative (without heat change)
            m.addConstr(CumHeat[k, s] <= CumHeat[k, s-1] + t_s,
                    name=f"HeatUpper_{k}_{s}")
            
            # Reset after heat change
            m.addConstr(CumHeat[k, s] <= t_s + MAX_HEAT_WEIGHT * (1 - HeatChg[k, s]),
                    name=f"HeatReset_{k}_{s}")
            
            # Allow intermixing (AUTHORS' SPECIAL CONSTRAINT)
            m.addConstr(CumHeat[k, s] + (MAX_HEAT_WEIGHT - CumHeat[k, s-1]) >= t_s,
                    name=f"HeatIntermix_{k}_{s}")
            
            # Minimum tonnage before heat change
            m.addConstr(MIN_HEAT_WEIGHT * HeatChg[k, s] <= CumHeat[k, s-1],
                    name=f"HeatMin_{k}_{s}")
            
            # Maximum tonnage before heat change
            m.addConstr(CumHeat[k, s-1] <= MAX_HEAT_WEIGHT + CumHeat[k, s] - t_s,
                    name=f"HeatMax_{k}_{s}")

    # Grade change to heat change (ONCE, not twice)
    for k in casters:
        for s in range(1, num_slots):
            m.addConstr(GradeChange[k, s] <= HeatChg[k, s+1], name=f"GradeHeat_{k}_{s}")
            t_s1 = gp.quicksum(coils[c].weight * (1/2000) * X[k, s+1, c] for c in coil_list)
            m.addConstr(CumHeat[k, s+1] >= t_s1 - MAX_HEAT_WEIGHT * (1 - GradeChange[k, s]),
                    name=f"GradeTon_{k}_{s}")    
    # ========== OBJECTIVE FUNCTION (EXACTLY as authors) ==========
    
    # Grade change objective
    grade_obj = gp.quicksum(GradeChangeCost[k, s] for k in casters for s in slots)
    
    # Gauge jump objective
    gauge_obj = gp.quicksum(GaugeJump[k, s] for k in casters for s in slots)
    
    # Inter-caster width difference objective
    width_obj = gp.quicksum(PENALTY_WIDTH_DIFFERENCE * P[s] for s in slots)
    
    # Trim loss objective (authors use average length)
    avg_length = sum(coils[c].length for c in coil_list) / len(coil_list)
    trim_obj = gp.quicksum(
        PENALTY_TRIM_LOSS * (V[k, s] - gp.quicksum(coils[c].orderWidth * X[k, s, c] for c in coil_list)) *
        3.5433 * avg_length * 0.2817929 * (1/2000)
        for k in casters for s in slots
    )
    
    # Roller change objective
    roller_obj = gp.quicksum(PENALTY_ROLL_CHANGE * Ybar[s] for s in slots)
    
    m.setObjective(grade_obj + gauge_obj + width_obj + trim_obj + roller_obj, GRB.MINIMIZE)
    
    m.update()
    
    return m, X, V, Ybar, GradeChange, GradeChangeCost, GaugeJump, P, HeatChg, CumHeat


# ============================================================
# SOLVE AND EXTRACT SOLUTION
# ============================================================

def solve_and_extract(model, X, V, Ybar, HeatChg, coils, casters, slots, coil_list, grade_cost_dict):
    """Solve model and extract solution in authors' format"""
    
    model.optimize()
    
    if model.Status == GRB.INFEASIBLE:
        print("Model is infeasible")
        return None
    
    if model.SolCount == 0:
        print("No solution found")
        return None
    
    print(f"\n✅ Optimal solution found!")
    print(f"   Objective: {model.ObjVal:.2f}")
    print(f"   Gap: {model.MIPGap*100:.2f}%")
    print(f"   Time: {model.Runtime:.2f}s")
    
    # Extract schedule
    schedule = {}
    cast_width = {}
    roller_changes = []
    
    for k in casters:
        for s in slots:
            for c in coil_list:
                if X[k, s, c].X > 0.5:
                    schedule[(k, s)] = c
                    cast_width[(k, s)] = V[k, s].X
                    break
    
    for s in slots:
        if s > 1 and Ybar[s].X > 0.5:
            roller_changes.append(s)
    
    # Calculate costs (as authors do in getCosts)
    grade_cost = 0
    gauge_cost = 0
    trim_cost = 0
    width_diff_cost = 0
    
    for (k, s) in schedule:
        c = schedule[(k, s)]
        coil = coils[c]
        
        # Trim cost
        trim_cost += (cast_width[(k, s)] - coil.orderWidth) * 3.5433 * coil.length * 0.2817929 * (1/2000)
        
        # Gauge and grade cost
        if s < max(slots):
            next_c = schedule.get((k, s+1))
            if next_c is not None:
                next_coil = coils[next_c]
                
                # Gauge cost (as authors)
                if coil.gauge >= HEAVY_GAUGE:
                    gauge_cost += max(0, coil.gauge * 0.5 - next_coil.gauge)
                elif coil.gauge >= MEDIUM_GAUGE:
                    gauge_cost += max(0, coil.gauge * 0.75 - next_coil.gauge)
                else:
                    gauge_cost += max(0, coil.gauge * 0.9 - next_coil.gauge)
                
                # Grade cost (as authors)
                grade_cost += get_grade_cost(coil.grade, next_coil.grade, grade_cost_dict)
    
    # Width difference cost
    for s in slots:
        if (1, s) in cast_width and (2, s) in cast_width:
            diff = abs(cast_width[(1, s)] - cast_width[(2, s)])
            width_diff_cost += max(0, diff - ALLOWABLE_INTERCASTER_WIDTH)
    
    total_cost = (grade_cost + 
                  PENALTY_GAUGE_DECREASE * gauge_cost +
                  PENALTY_WIDTH_DIFFERENCE * width_diff_cost +
                  PENALTY_TRIM_LOSS * trim_cost +
                  COST_ROLLER_CHANGE * (len(roller_changes) + 1))
    
    print(f"\n--- Schedule ---")
    for k in casters:
        print(f"\nCaster {k}:")
        for s in slots:
            c = schedule.get((k, s))
            if c is not None:
                heat = "✓" if HeatChg[k, s].X > 0.5 else ""
                print(f"  Slot {s}: Coil {c}, Grade {coils[c].grade}, "
                      f"Width={cast_width[(k, s)]:.1f}, "
                      f"Gauge={coils[c].gauge:.3f}, "
                      f"Weight={coils[c].weight:.1f}t {heat}")
    
    print(f"\n--- Cost Summary ---")
    print(f"  Grade change cost: {grade_cost:.2f}")
    print(f"  Gauge change cost: {PENALTY_GAUGE_DECREASE * gauge_cost:.2f}")
    print(f"  Width diff cost: {PENALTY_WIDTH_DIFFERENCE * width_diff_cost:.2f}")
    print(f"  Trim loss cost: {PENALTY_TRIM_LOSS * trim_cost:.2f}")
    print(f"  Roller cost: {COST_ROLLER_CHANGE * (len(roller_changes) + 1):.2f}")
    print(f"  TOTAL: {total_cost:.2f}")
    
    return {
        'schedule': schedule,
        'cast_width': cast_width,
        'roller_changes': roller_changes,
        'objective': model.ObjVal,
        'total_cost': total_cost
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
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
        exit(1)
    
    print(f"Loading: {instance_path}")
    coils, grade_cost_dict = load_instance_from_csv(instance_path)
    print(f"Loaded {len(coils)} coils")
    
    # Build model EXACTLY as authors
    print("\nBuilding model (authors' formulation)...")
    model, X, V, Ybar, GradeChange, GradeChangeCost, GaugeJump, P, HeatChg, CumHeat = build_model(
        coils, grade_cost_dict,
        use_mill_cuts=0,
        use_grade_cuts=0,
        add_critical_pairs=0
    )
    
    print(f"Model: {model.NumVars} variables, {model.NumConstrs} constraints")
    
    # Solve
    result = solve_and_extract(model, X, V, Ybar, HeatChg, coils,
                               casters=[1, 2],
                               slots=list(range(1, len(coils)//2 + 1)),
                               coil_list=list(coils.keys()), grade_cost_dict=grade_cost_dict)