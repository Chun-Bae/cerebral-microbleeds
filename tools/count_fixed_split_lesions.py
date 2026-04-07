#!/usr/bin/env python3
"""
고정 분할(fixed_split)에 속한 환자들의 병변 수를 집계하는 스크립트.

출력 내용:
  1. 각 split(train/valid/test)별 환자 목록과 병변 수
  2. 각 split별 stratum 분포 및 병변 합계
  3. 전체 요약
"""

import os
from collections import defaultdict

# ── 경로 설정 ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LESION_TSV = os.path.join(PROJECT_ROOT, "data", "splits", "lesion_counts.tsv")
FIXED_SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "fixed_split")

SPLITS = ["train", "valid", "test"]

# stratum 표시 순서
STRATUM_ORDER = ["none", "very low", "low", "medium", "high", "extreme"]


def load_lesion_counts(path):
    """lesion_counts.tsv → {patient_id: (lesion_count, stratum)}"""
    data = {}
    with open(path, "r") as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            pid, count, stratum = parts[0], int(parts[1]), parts[2]
            data[pid] = (count, stratum)
    return data


def load_split(path):
    """split txt → [patient_id, ...]"""
    patients = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                patients.append(line)
    return patients


def print_separator(char="─", width=72):
    print(char * width)


def main():
    lesion_data = load_lesion_counts(LESION_TSV)

    grand_total_patients = 0
    grand_total_lesions = 0
    grand_stratum_counts = defaultdict(lambda: {"patients": 0, "lesions": 0})

    for split in SPLITS:
        split_path = os.path.join(FIXED_SPLIT_DIR, f"{split}.txt")
        patients = load_split(split_path)

        print()
        print_separator("═")
        print(f"  📂  {split.upper()}  ({len(patients)}명)")
        print_separator("═")

        # ── 환자별 병변 수 ──
        print(f"  {'환자ID':<10} {'병변수':>6}  {'Stratum':<10}")
        print_separator("─")

        split_total = 0
        stratum_agg = defaultdict(lambda: {"patients": 0, "lesions": 0})
        missing = []

        for pid in patients:
            if pid in lesion_data:
                count, stratum = lesion_data[pid]
                print(f"  {pid:<10} {count:>6}  {stratum:<10}")
                split_total += count
                stratum_agg[stratum]["patients"] += 1
                stratum_agg[stratum]["lesions"] += count
                grand_stratum_counts[stratum]["patients"] += 1
                grand_stratum_counts[stratum]["lesions"] += count
            else:
                missing.append(pid)
                print(f"  {pid:<10} {'N/A':>6}  {'(없음)':<10}")

        print_separator("─")
        print(f"  합계: {split_total}개 병변  |  환자 {len(patients)}명")

        if missing:
            print(f"  ⚠  TSV에 없는 환자: {', '.join(missing)}")

        # ── Stratum별 집계 ──
        print()
        print(f"  ▸ Stratum별 집계 ({split})")
        print(f"  {'Stratum':<12} {'환자수':>6} {'병변합':>8} {'비율':>8}")
        print_separator("·")

        for s in STRATUM_ORDER:
            if s in stratum_agg:
                p = stratum_agg[s]["patients"]
                l = stratum_agg[s]["lesions"]
                pct = (l / split_total * 100) if split_total > 0 else 0
                print(f"  {s:<12} {p:>6} {l:>8} {pct:>7.1f}%")

        print_separator("·")
        print(f"  {'전체':<12} {len(patients) - len(missing):>6} {split_total:>8} {'100.0%':>8}")

        grand_total_patients += len(patients)
        grand_total_lesions += split_total

    # ── 전체 요약 ──
    print()
    print_separator("━")
    print("  📊  전체 요약 (Fixed Split)")
    print_separator("━")
    print(f"  총 환자 수: {grand_total_patients}명")
    print(f"  총 병변 수: {grand_total_lesions}개")
    print()
    print(f"  {'Stratum':<12} {'환자수':>6} {'병변합':>8} {'비율':>8}")
    print_separator("·")

    for s in STRATUM_ORDER:
        if s in grand_stratum_counts:
            p = grand_stratum_counts[s]["patients"]
            l = grand_stratum_counts[s]["lesions"]
            pct = (l / grand_total_lesions * 100) if grand_total_lesions > 0 else 0
            print(f"  {s:<12} {p:>6} {l:>8} {pct:>7.1f}%")

    print_separator("·")
    total_p = sum(v["patients"] for v in grand_stratum_counts.values())
    print(f"  {'전체':<12} {total_p:>6} {grand_total_lesions:>8} {'100.0%':>8}")
    print()


if __name__ == "__main__":
    main()
