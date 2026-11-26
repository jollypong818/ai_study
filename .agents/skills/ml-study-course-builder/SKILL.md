---
name: ml-study-course-builder
description: >-
  Use this skill whenever the user provides a dataset (URL, UCI/Kaggle link, CSV file, or dataset name)
  or an ML study topic (e.g., 회귀, 분류, 시계열, 군집, 추천, NLP) and wants to generate a complete
  Google Colab self-study curriculum and 30-minute peer presentation package inside a numbered topic subfolder
  in this workspace.
---

# ML Study Course Builder (Spring 개발자 → AI 엔지니어 전환용)

사용자가 데이터셋(링크, 파일, 주제)만 던져주면, 이 워크스페이스 안에 **주제별 하위폴더**를 만들고 **Google Colab 자습용 노트북(Step 1~5) + 로드맵 + 30분 공유 발표 대본**까지 한 번에 생성하는 워크플로우다.

> **필독 레퍼런스:** 자료를 생성하기 전에 반드시 아래 두 파일을 읽고 규칙을 그대로 적용한다.
> - 교수법·문체·코드 설명 규칙: [pedagogy_and_style.md](./references/pedagogy_and_style.md)
> - Google Colab 노트북 제작 규칙: [colab_notebook_template.md](./references/colab_notebook_template.md)

---

## 실행 절차

### 1단계. 대상 데이터셋 파악 & 주제 분류
1. 사용자가 제공한 데이터셋(URL, 파일 경로, 데이터셋명)을 확인한다.
   - URL이나 외부 데이터셋이면 웹 검색 또는 파일 읽기로 **실제 컬럼명, 행 수, 타겟(정답) 컬럼, 공식 다운로드 URL, 데이터 특성**을 반드시 먼저 검증한다. (컬럼명 추측 금지)
   - 로컬 CSV 파일이면 해당 파일을 직접 읽어 `head`, 컬럼 목록, 결측치, 타겟 분포를 확인한다.
2. 학습 주제를 분류한다. (예: `regression`, `classification`, `timeseries`, `clustering`, `nlp`, `recsys`)
3. 워크스페이스 루트의 기존 주제 폴더를 확인해 하위폴더명을 결정한다.
   - 명명 규칙: `<영문주제>/` (예: `regression/`, `classification/`, `clustering/`, `timeseries/`, `nlp/`)
   - 같은 주제로 두 번째 데이터셋을 다룰 때만 `<영문주제>_<짧은데이터셋명>/` (예: `classification_credit_fraud/`)을 사용한다.
   - 모든 주제 폴더 안의 데이터 파일은 **`dataset/`** 하위폴더에 저장·로드한다 (`os.makedirs("dataset", exist_ok=True)`).

### 2단계. 이 데이터셋만의 '생산적 실패(함정)' 3개 설계
정답 코드부터 짜지 않는다. 이 데이터셋에서 입문자가 100% 밟을 만한 **실전 함정 2~3개**를 먼저 뽑는다.
- 예시는 [pedagogy_and_style.md](./references/pedagogy_and_style.md)의 "주제별 단골 함정 카탈로그"를 참고한다.
- 모든 단계는 **1단계 = 1변경 = 1평가지표 변화**로 이어져야 한다.

### 3단계. 하위폴더 안에 전체 학습 패키지 생성
하위폴더 안에 아래 파일을 모두 생성한다. (사용자가 "Step 1만 먼저"처럼 범위를 좁힌 경우에만 해당 파일부터 만든다.)

| 파일명 | 역할 |
| :--- | :--- |
| `README.md` | 주제별 커리큘럼, Spring↔ML 매핑, 자습(4시간) vs 발표(30분) 2-트랙 표, 빈 실험 벤치마크 기록표, Colab 실행법 |
| `01_eda_and_first_trap.ipynb` | **Step 1:** DB 까보듯 EDA + 문서/데이터 이상 탐지 + 핵심 그래프 1~2장 + 뒤 단계 함정 복선 심기 |
| `02_baseline_and_trap1.ipynb` | **Step 2:** 가장 멍청한 기준선(Baseline) 점수 측정 + **함정 1** 일부러 터뜨리고 수정 |
| `03_validation_and_trap2.ipynb` | **Step 3:** 검증 분할 설계 + **함정 2** 체험 + 진짜 실력 점수 확인 |
| `04_models_and_error_analysis.ipynb` | **Step 4:** 단순 모델 한계 확인 → 더 나은 모델 비교 + **가장 크게 틀린 데이터 5건 직접 뜯어보기(에러 분석)** |
| `05_feature_engineering_and_wrapup.ipynb` | **Step 5:** 파생변수(Feature Engineering) 2~3개로 오차 개선 + 변수 중요도 확인 + 전체 실험 요약 |
| `PRESENTATION_30MIN.md` | **30분 발표 대본:** 자습 4시간 분량을 30분으로 압축한 분 단위 타임라인 + 그대로 읽어도 자연스러운 구어체 대본 + 예상 Q&A 5개 |

### 4단계. 루트 인덱스 업데이트
워크스페이스 루트의 `README.md`에 새로 만든 주제 하위폴더 링크, 학습 주제, 데이터셋, 핵심 평가지표, 함정 요약을 한 줄 추가한다.

### 5단계. 생성 결과 검증 (필수)
1. 생성한 모든 `.ipynb` 파일이 유효한 JSON(`nbformat: 4.5`)인지 명령어로 검증한다.
   - Windows PowerShell 예시:
     ```powershell
     Get-ChildItem "<폴더경로>\*.ipynb" | ForEach-Object { $nb = Get-Content $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json; "$($_.Name): OK ($($nb.cells.Count) cells)" }
     ```
2. 모든 코드 셀 **내부 주석(줄별 동작·옵션·Java/SQL 비유)**과 코드 셀 아래 **`🔍 코드 뜯어보기` 해설**이 빠지지 않았는지 점검한다.
3. AI 금지 문체(`~는 ~를 전제로 하지 않는다`, `~라고 해서 반드시 ~인 것은 아니다` 등)가 들어가지 않았는지 점검한다.
4. 사용자에게 생성된 파일 목록(클릭 가능한 `file:///` 링크)과 Colab 업로드 방법을 안내한다.
