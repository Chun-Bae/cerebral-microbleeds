import os
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 데이터 파싱 및 정렬 (0p01 및 0.00001 대응)
# ==========================================
base_dir = "results/shallow_layer_v9_nms_brute_force/"
output_csv = "nms_brute_force_results.csv"

if not os.path.exists(base_dir):
    print(f"⚠️ 디렉토리를 찾을 수 없습니다: {base_dir}")
    valid_folders = []
else:
    all_items = os.listdir(base_dir)
    valid_folders = []

    for item in all_items:
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path):
            # 폴더명에서 숫자 추출 (0p01 형태 또는 일반 숫자 형태)
            match = re.search(r'(\d+)p(\d+)', item)
            if match:
                # 0p01 -> 0.01 변환
                nms_val = float(f"{match.group(1)}.{match.group(2)}")
                valid_folders.append((nms_val, item))
            else:
                try:
                    # 일반 숫자 형태 (0.1, 0.05 등)
                    nms_val = float(item)
                    valid_folders.append((nms_val, item))
                except ValueError:
                    continue

    # [중요] 문자열이 아닌 '실수(float)' 기준으로 정렬하여 0.00001이 가장 먼저 오도록 함
    valid_folders.sort(key=lambda x: x[0])
    print(f"📂 총 {len(valid_folders)}개의 폴더를 수치 순서로 정렬했습니다.")

results = []
for nms_val, folder_name in valid_folders:
    folder_path = os.path.join(base_dir, folder_name)
    log_candidates = [
        os.path.join(folder_path, "eval_log.txt"),
        os.path.join(folder_path, "fixed_split", "eval_log.txt")
    ]

    log_path = next((c for c in log_candidates if os.path.exists(c)), None)
    if not log_path:
        continue

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

        def get_val(pattern, text):
            match = re.search(pattern, text)
            return match.group(1) if match else ""

        results.append({
            "NMS_Thresh": nms_val,  # 시각화용 숫자
            "Precision": get_val(r"Precision\s+:\s+([\d.]+)", content),
            "Recall": get_val(r"Recall\s+:\s+([\d.]+)", content),
            "F1": get_val(r"F1\s+:\s+([\d.]+)", content),
            "AP": get_val(r"AP@[\d.]+\s+\(Target\)\s+:\s+([\d.]+)", content)
        })

if results:
    df = pd.DataFrame(results).apply(pd.to_numeric)
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV 저장 완료: '{output_csv}'")
else:
    print("❌ 파싱된 데이터가 없습니다.")
    exit()

# ==========================================
# 2. 하이브리드 라벨링 시각화 (웜톤 세미컬러 테마)
# ==========================================
output_image = "nms_performance_graph.png"

# 스타일 설정
sns.set_theme(style="whitegrid")
plt.figure(figsize=(22, 12))

metric_names = ['Precision', 'Recall', 'F1', 'AP']
# 요청하신 웜톤 세미컬러 팔레트
palette = ["#4C72B0", "#DD8452", "#C44E52", "#CCB974"]

# 메인 그래프 선 그리기
x_positions = range(len(df))
x_labels = [f"{x:g}" for x in df['NMS_Thresh']]

for i, metric in enumerate(metric_names):
    plt.plot(x_positions, df[metric],
             label=metric,
             color=palette[i],
             marker='o',
             markersize=9,
             linewidth=3,
             zorder=3)

# --- 라벨링 로직 (사용자 조정 수치 적용) ---
stack_y_positions = {
    'Precision': 1.15,
    'F1': 1.11,
    'Recall': 1.07,
    'AP': 1.03
}

for idx, row in df.iterrows():
    nms_val = row['NMS_Thresh']
    x_val = idx

    for m_idx, metric in enumerate(metric_names):
        score = row[metric]
        if pd.isna(score):
            continue

        color = palette[m_idx]

        # 지정하신 스택 조건
        should_stack = False
        if metric in ['Recall', 'AP']:
            if nms_val <= 0.65:
                should_stack = True
        else:  # Precision, F1
            if nms_val <= 0.40:
                should_stack = True

        if should_stack:
            target_y = stack_y_positions[metric]
            # 가이드 라인 (연한 점선)
            plt.plot([x_val, x_val], [score, target_y],
                     color=color, linestyle='--', linewidth=0.8, alpha=0.3, zorder=1)
            # 가로 숫자 표시 (테두리 없음)
            plt.text(x_val, target_y, f"{score:.3f}",
                     color=color, fontsize=9.5, ha='center', va='center',
                     fontweight='bold', zorder=5,
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.5))
        else:
            # 원상태 구간 (사용자 조정: nms-0.01, score-0.02, va='top')
            plt.text(x_val - 0.2, score - 0.02, f"{score:.3f}",
                     color=color, fontsize=9.5, ha='center', va='top',
                     fontweight='bold', zorder=5)

# 그래프 세부 디자인
plt.title('SSD_FE_V9 Model Performance Analysis',
          fontsize=22, fontweight='bold', pad=40)
plt.xlabel('NMS Threshold', fontsize=15)
plt.ylabel('Score', fontsize=15)

# X축 눈금을 실제 값들에 맞게 정렬하여 표시
plt.xticks(x_positions, x_labels, rotation=45, fontsize=11)
plt.yticks(fontsize=11)

# 범위 조정
current_min = df[metric_names].values.min()
plt.ylim(max(0, current_min - 0.12), 1.20)

plt.legend(title='Performance Metrics', bbox_to_anchor=(1.01, 1), loc='upper left',
           fontsize=13, title_fontsize=14, frameon=True)

plt.tight_layout()
plt.savefig(output_image, dpi=300)
print(f"✅ 최종 그래프 저장 완료: '{output_image}'")
