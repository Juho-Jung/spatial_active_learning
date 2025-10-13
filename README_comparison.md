# Active Learning Comparison Scripts

베이스라인 실험 결과를 비교하고 시각화하는 스크립트들입니다.

## 1. 성능 비교 스크립트 (al_perf_comparison.py)

여러 베이스라인 결과를 비교하여 성능 메트릭을 시각화합니다.

### 사용법:
```bash
python al_perf_comparison.py \
    --result_paths \
        /team/team_pxi/workspace/juhojung/spatial_active_learning/al_results/pneumoperitoneum_sdc_ppm_train-0908_spatial_split_r_10_s_20_val_500/adaptive_base_20250919_172458/results.json \
        /team/team_pxi/workspace/juhojung/spatial_active_learning/al_results/pneumoperitoneum_sdc_ppm_train-0908_spatial_split_r_10_s_20_val_500/random_base_20250918_205515/results.json \
    --output_dir output_comparison/performance \
    --target_lesion pneumoperitoneum
```

### 생성되는 파일:
- `spatial_metrics_comparison.png`: 공간 메트릭 비교 (Worst Bin Dice, P10 Bin Dice, Bin Std, Spatial Consistency)
- `performance_coverage_comparison.png`: 성능 커버리지 비교
- `combined_metrics_comparison.png`: 통합 메트릭 비교 (Validation Dice, Loss, Average Bin Dice, Min/Max Range)
- `normalized_spatial_metrics_comparison.png`: 정규화된 공간 메트릭 비교 (Performance Coverage, Spatial Consistency)

## 2. 시각화 비교 스크립트 (al_visualization_comparison.py)

두 베이스라인 결과의 선택된 샘플 위치를 시각화하여 비교합니다.

### 사용법:
```bash
python al_visualization_comparison.py \
    --result1_path /team/team_pxi/workspace/juhojung/spatial_active_learning/al_results/pneumoperitoneum_sdc_ppm_train-0908_spatial_split_r_10_s_20_val_500/adaptive_base_20250919_172458/results.json \
    --result2_path /team/team_pxi/workspace/juhojung/spatial_active_learning/al_results/pneumoperitoneum_sdc_ppm_train-0908_spatial_split_r_10_s_20_val_500/random_base_20250918_205515/results.json \
    --output_dir output_comparison/visualization \
    --target_lesion pneumoperitoneum \
    --collection sdc_ppm_train-0908
```

### 생성되는 파일:
- `adaptive_base_random_base/comparison/round1_comparison.png`: 라운드별 비교 플롯
- `adaptive_base_random_base/adaptive_base/round1_adaptive_base.png`: 개별 모델 플롯
- `adaptive_base_random_base/random_base/round1_random_base.png`: 개별 모델 플롯
- `adaptive_base_random_base/cumulative_round1_adaptive_base.png`: 누적 선택 시각화
- `adaptive_base_random_base/cumulative_round1_random_base.png`: 누적 선택 시각화
- ... (각 라운드별로 생성)

## 3. 디렉토리 구조

```
output_comparison/
├── performance/
│   ├── spatial_metrics_comparison.png
│   ├── performance_coverage_comparison.png
│   ├── combined_metrics_comparison.png
│   └── normalized_spatial_metrics_comparison.png
└── visualization/
    └── adaptive_base_random_base/
        ├── comparison/
        │   ├── round1_comparison.png
        │   ├── round2_comparison.png
        │   └── ...
        ├── adaptive_base/
        │   ├── round1_adaptive_base.png
        │   ├── round2_adaptive_base.png
        │   └── ...
        ├── random_base/
        │   ├── round1_random_base.png
        │   ├── round2_random_base.png
        │   └── ...
        ├── cumulative_round1_adaptive_base.png
        ├── cumulative_round2_adaptive_base.png
        ├── cumulative_round1_random_base.png
        ├── cumulative_round2_random_base.png
        └── ...
```

## 4. 주요 기능

### 성능 비교 스크립트:
- 최대 8개 베이스라인 결과를 동시에 비교
- 공간 메트릭, 성능 커버리지, 통합 메트릭 시각화
- 정규화된 공간 메트릭 비교 (Performance Coverage, Spatial Consistency)
- 자동으로 모델 이름 추출 (경로에서 strategy_uncertainty 부분)
- 요약 통계 출력

### 시각화 비교 스크립트:
- Ground truth mask의 중심점 계산
- 512x512 그리드에서 샘플 선택 위치 시각화
- 두 모델 간 공간적 분포 비교
- 누적 선택 시각화 (모든 라운드의 선택을 색상으로 구분)
- 정리된 디렉토리 구조 (comparison/, model1/, model2/)
- 라운드별 통계 정보 출력

## 5. 사용 예시

### 간단한 실행:
```bash
# 예시 스크립트 실행 (경로 수정 필요)
./run_comparison_example.sh
```

### 개별 실행:
```bash
# 성능 비교만 실행
python al_perf_comparison.py \
    --result_paths result1.json result2.json result3.json \
    --output_dir output_comparison/performance \
    --target_lesion pneumoperitoneum

# 시각화 비교만 실행
python al_visualization_comparison.py \
    --result1_path result1.json \
    --result2_path result2.json \
    --output_dir output_comparison/visualization \
    --target_lesion pneumoperitoneum \
    --collection sdc_ppm_train-0908
```

## 6. 예시 결과

### 성능 비교:
- 각 라운드별 성능 메트릭의 변화 추이
- 베이스라인 간 성능 차이 시각화
- 최종 성능 요약

### 시각화 비교:
- 샘플 선택의 공간적 분포 패턴
- 한 곳에 몰려서 선택되는지 vs 고르게 분산되는지 비교
- 각 라운드별 선택 패턴의 변화

## 7. 주의사항

- 결과 파일 경로가 정확한지 확인하세요
- 데이터베이스 연결이 필요한 경우 적절한 환경 설정이 필요합니다
- 시각화 비교는 ground truth mask가 있는 샘플만 처리됩니다
