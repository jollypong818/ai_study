# Google Colab 노트북(`.ipynb`) 제작 규칙

## 1. 파일 포맷 및 메타데이터
- `nbformat: 4`, `nbformat_minor: 5`
- `metadata.colab`에 `"provenance": [], "toc_visible": true` 포함
- 모든 코드 셀의 `"execution_count": null`, `"outputs": []`로 초기화
- 각 셀에 의미 있는 고유 `"id"` 부여 (예: `step1-intro`, `code-load-data`)

## 2. 데이터 로드 규칙 (Colab + 로컬 공용)
- 학습자가 로컬과 Google Colab 양쪽에서 코드 수정 없이 실행되도록 **노트북 첫 코드 셀에서 `dataset/` 폴더에 자동 다운로드**한다.
  - 항상 `os.makedirs("dataset", exist_ok=True)`로 폴더를 먼저 만들고, `dataset/<파일명>` 경로로 저장·로드한다.
  - UCI / GitHub Raw URL: `urllib.request.urlretrieve` + `zipfile` (`zf.extractall("dataset")`) 또는 `pd.read_csv(url)`
  - `sklearn.datasets` / `seaborn` 내장 데이터: 바로 로드
  - Kaggle 전용(인증 필요) 데이터인 경우: `kagglehub` 또는 Colab 파일 업로드(`from google.colab import files`) 스니펫을 깔끔하게 제공
- 이미 `dataset/<파일명>`이 있으면 재다운로드를 건너뛰도록 `if not os.path.exists("dataset/<파일명>")` 가드를 넣는다.

## 3. Colab 호환성 체크리스트 (Python 3.12 / pandas 2.x / numpy 2.x 기준)
1. **한글 폰트 깨짐 방지:**
   - Colab 기본 `matplotlib`은 한글 폰트가 없어 `□□□`로 깨진다.
   - `apt-get` + 런타임 재시작으로 흐름을 끊지 말고, **그래프 축·제목·범례 라벨만 깔끔한 영어**로 작성하고 마크다운·주석·print 출력은 전부 한국어로 작성한다.
   - 첫 그래프 앞에 *"Colab 기본 폰트 깨짐을 막기 위해 그래프 라벨만 영어로 적는다"*고 1줄 안내한다.
2. **numpy 2.x 출력 깔끔하게:**
   - `sorted(x.unique())` 대신 `sorted(int(v) for v in x.unique())` 사용 (`np.int64(...)` 지저분한 출력 방지)
   - `bool((...).all())`로 감싸 `np.True_` 대신 깔끔한 `True` 출력
3. **pandas 2.x 경고 방지:**
   - `pd.cut` 결과를 `groupby`할 때 `observed=False` 명시

## 4. 모든 노트북 공통 섹션 구조
1. **도입 셀 (Markdown):** 예상 시간, 목표, 백엔드 상황극, POE 공부 규칙
2. **본문 실습 (Markdown + Code 반복):**
   - `[실행 전 예측]` 질문
   - 코드 셀 (**코드 블록 안에 인라인·상단 주석으로 줄별 동작, 옵션 의미, Java/SQL 비유를 바로 적어** 위아래로 시선이 왔다갔다하지 않고 코드만 읽어도 이해되게 작성)
   - **`🔍 코드 뜯어보기`** 셀 (코드 주석으로 다 담기 어려운 SQL 번역 블록, 핵심 원리·함정 보충 설명)
   - 결과 해석 + 다음 단계 복선 / 발표 팁
3. **직접 해보는 자습 미션 (10분):**
   - `# TODO`가 들어간 빈 코드 셀 1~2개
   - `<details><summary>정답 코드 보기</summary>...</details>` 접힘 블록 (정답 코드 + 문법 설명)
4. **30분 발표용 정리 (Feynman Checkpoint):**
   - 본인 말로 채우는 3줄 요약 빈칸
   - 이 Step을 발표에서 2~3분 안에 설명할 때 그대로 읽어도 되는 구어체 대본 샘플

## 5. JSON 검증 명령어 (생성 직후 필수 실행)
```powershell
$nb = Get-Content -Path "<노트북경로>" -Raw -Encoding UTF8 | ConvertFrom-Json
"Valid JSON! cells=$($nb.cells.Count)"
```
