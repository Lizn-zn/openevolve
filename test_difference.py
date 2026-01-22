#!/usr/bin/env python3
"""测试 difference 计算"""

import json
from pathlib import Path

# 读取 checkpoint 数据
checkpoint_path = Path("problems/verina_advanced_1/openevolve_output/checkpoints/checkpoint_90")
programs_dir = checkpoint_path / "programs"

print(f"Reading programs from: {programs_dir}")
print()

# 获取有 consts_jsons 的程序
programs_with_consts = []
for prog_file in programs_dir.glob("*.json"):
    with open(prog_file) as f:
        data = json.load(f)
    
    artifacts = data.get("artifacts_json") or {}
    consts_jsons = artifacts.get("_consts_jsons", [])
    
    if consts_jsons:
        programs_with_consts.append({
            "id": data["id"][:8],
            "consts_jsons": consts_jsons,
            "num_sorries": len(consts_jsons),
        })

print(f"Total programs with consts_jsons: {len(programs_with_consts)}")
print()

# 显示前几个程序的 consts_jsons 结构
print("=" * 60)
print("Sample consts_jsons:")
print("=" * 60)
for i, prog in enumerate(programs_with_consts[:3]):
    print(f"\nProgram {prog['id']} ({prog['num_sorries']} sorries):")
    for j, sorry_consts in enumerate(prog["consts_jsons"]):
        print(f"  Sorry {j}: {len(sorry_consts)} consts")
        # 显示前 5 个常量
        print(f"    Sample: {sorry_consts[:3]}...")

# Chamfer Distance 计算
def jaccard_distance(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    if not set_a or not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - (intersection / union if union > 0 else 0.0)


def chamfer_distance(current: list, reference: list) -> float:
    if not current and not reference:
        return 0.0
    if not current or not reference:
        return 1.0

    current_sets = [set(consts) for consts in current]
    reference_sets = [set(consts) for consts in reference]

    # Direction 1: current -> reference
    sum_current_to_ref = 0.0
    for c_set in current_sets:
        min_dist = min(jaccard_distance(c_set, r_set) for r_set in reference_sets)
        sum_current_to_ref += min_dist
    avg_current_to_ref = sum_current_to_ref / len(current_sets)

    # Direction 2: reference -> current
    sum_ref_to_current = 0.0
    for r_set in reference_sets:
        min_dist = min(jaccard_distance(r_set, c_set) for c_set in current_sets)
        sum_ref_to_current += min_dist
    avg_ref_to_current = sum_ref_to_current / len(reference_sets)

    return (avg_current_to_ref + avg_ref_to_current) / 2.0


# 计算两两之间的 Chamfer Distance
print("\n" + "=" * 60)
print("Pairwise Chamfer Distances:")
print("=" * 60)

if len(programs_with_consts) >= 2:
    # 取前 5 个程序计算
    sample_progs = programs_with_consts[:6]
    
    all_distances = []
    for i, prog1 in enumerate(sample_progs):
        for j, prog2 in enumerate(sample_progs):
            if i < j:
                dist = chamfer_distance(prog1["consts_jsons"], prog2["consts_jsons"])
                all_distances.append(dist)
                print(f"{prog1['id']} vs {prog2['id']}: {dist:.4f}")
    
    print(f"\nMin distance: {min(all_distances):.4f}")
    print(f"Max distance: {max(all_distances):.4f}")
    print(f"Avg distance: {sum(all_distances)/len(all_distances):.4f}")

# 看看常量集合的重叠情况
print("\n" + "=" * 60)
print("Constants overlap analysis:")
print("=" * 60)

if len(programs_with_consts) >= 2:
    # 把每个程序的所有常量合并成一个 set
    all_const_sets = []
    for prog in programs_with_consts[:6]:
        all_consts = set()
        for sorry_consts in prog["consts_jsons"]:
            all_consts.update(sorry_consts)
        all_const_sets.append({
            "id": prog["id"],
            "consts": all_consts,
            "count": len(all_consts)
        })
    
    for cs in all_const_sets:
        print(f"{cs['id']}: {cs['count']} unique consts")
    
    print("\nPairwise Jaccard (flattened) - this is the OLD broken method:")
    for i, cs1 in enumerate(all_const_sets):
        for j, cs2 in enumerate(all_const_sets):
            if i < j:
                dist = jaccard_distance(cs1["consts"], cs2["consts"])
                intersection = len(cs1["consts"] & cs2["consts"])
                union = len(cs1["consts"] | cs2["consts"])
                print(f"{cs1['id']} vs {cs2['id']}: Jaccard={dist:.4f} (∩={intersection}, ∪={union})")

# 分析每个 sorry 的 Jaccard 距离
print("\n" + "=" * 60)
print("Per-sorry Jaccard analysis (detailed):")
print("=" * 60)

if len(programs_with_consts) >= 2:
    prog1 = programs_with_consts[0]
    prog2 = programs_with_consts[1]
    
    print(f"\nComparing {prog1['id']} ({prog1['num_sorries']} sorries) vs {prog2['id']} ({prog2['num_sorries']} sorries)")
    
    sets1 = [set(c) for c in prog1["consts_jsons"]]
    sets2 = [set(c) for c in prog2["consts_jsons"]]
    
    print("\nJaccard distance matrix:")
    print("       ", end="")
    for j in range(len(sets2)):
        print(f"S2_{j:2d}  ", end="")
    print()
    
    for i, s1 in enumerate(sets1):
        print(f"S1_{i}: ", end="")
        for j, s2 in enumerate(sets2):
            dist = jaccard_distance(s1, s2)
            print(f"{dist:.3f}  ", end="")
        print()
    
    print(f"\nChamfer distance for this pair: {chamfer_distance(prog1['consts_jsons'], prog2['consts_jsons']):.4f}")
