"""자전거 대여량 회귀 실습: 노트북 01~05와 자습 미션을 한 번에 실행한다.

README.md와 PRESENTATION_30MIN.md의 발표용 수치는 출력에서 다시 계산한다.
1. 데이터 탐색 → 2. 기준선·정답 누수 → 3. 시간 검증 → 4. 모델·오차 분석
→ 5. 파생 피처·최종 모델·API 요청 모의 추론 순서로 읽으면 된다.
"""

import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

# 화면을 띄우지 않고 PNG만 쓰는 백엔드로 고정한다. 서버나 터미널에서도 실행 가능하다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 실행 위치가 아니라 이 파일을 기준으로 CSV와 결과 폴더를 찾는다.
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "hour.csv"
UCI_ARCHIVE = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
SPLIT_DATE = "2012-09-01"
SEED = 42
TARGET = "cnt"

# 모델 입력은 SQL SELECT처럼 필요한 컬럼만 명시한다. ID·날짜·정답 구성값은 제외한다.
BASE_FEATURES = [
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed",
]
LEAK_FEATURES = ["casual", "registered"]
CALENDAR_FEATURES = ["is_rush_hour", "hr_sin", "hr_cos"]
LAG_DAYS = {"cnt_prev_day": 1, "cnt_prev_week": 7, "cnt_prev_2week": 14}
FINAL_FEATURES = BASE_FEATURES + list(LAG_DAYS)


def load_data(path):
    """시간별 원본을 읽는다. 기본 CSV가 없을 때만 로컬 ZIP 또는 UCI를 사용한다."""
    if not path.exists():
        # 사용자 지정 경로 오타를 다운로드로 숨기지 않는다.
        if path != DATA_PATH:
            raise FileNotFoundError(path)
        archive = path.with_name("bike_sharing_dataset.zip")
        if archive.exists():
            source = archive
        else:
            # 전체 ZIP을 풀지 않고 아래에서 필요한 hour.csv만 추출한다.
            with urlopen(UCI_ARCHIVE, timeout=30) as response:
                source = BytesIO(response.read())
        with ZipFile(source) as zipped:
            content = zipped.read("hour.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    df = pd.read_csv(path, parse_dates=["dteday"])
    # 인덱스는 나중에 랜덤 분할에서 학습/테스트 소속을 추적할 때 쓰는 행 번호다.
    return df.sort_values(["dteday", "hr"]).reset_index(drop=True)


def save_plot(ax, output, filename, title, xlabel, ylabel):
    """그래프 스타일과 파일 저장을 한곳에서 처리한다."""
    # 운영체제별 기본 한글 폰트 차이로 글자가 깨지지 않게 그래프 라벨은 영어로 적는다.
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis="y", alpha=0.3)
    figure = ax.get_figure()
    figure.tight_layout()
    figure.savefig(output / filename, dpi=140)
    plt.close(figure)  # 다음 그래프에 축이 섞이거나 메모리에 남지 않게 닫는다.


def explore_data(df, output):
    """01. SQL 조회하듯 스키마·누락·그룹별 대여 패턴을 확인한다."""
    print("\n=== 01. 탐색적 데이터 분석 (EDA) ===")
    # head()는 SQL의 LIMIT 5, describe()는 대여량의 요약 통계다.
    print(f"행: {len(df):,} | 열: {df.shape[1]} | NULL: {df.isna().sum().sum()}")
    print(df.head().to_string(index=False))
    print("\n[컬럼 타입]\n", df.dtypes.to_string())
    print("\n[대여량 분포]\n", df[TARGET].describe().round(2).to_string())

    # NULL 개수만으로는 통째로 빠진 행을 못 찾는다. 기간 전체의 날짜×24와 비교한다.
    dates = pd.date_range(df["dteday"].min(), df["dteday"].max(), freq="D")
    hours_per_day = df.groupby("dteday")["hr"].nunique().reindex(dates, fill_value=0)
    print(f"\n기간: {dates[0]:%Y-%m-%d} ~ {dates[-1]:%Y-%m-%d} | 누락 시간: {(24 - hours_per_day).sum()}시간")
    print("[기록이 적은 날짜 TOP 5]\n", hours_per_day.nsmallest(5).to_string())

    # [실행 전 예측] Readme.txt의 'season=1은 봄'과 실제 월/기온이 일치할까?
    seasons = df.groupby("season").agg(
        months=("mnth", lambda values: sorted(int(value) for value in values.unique())),
        avg_temp=("temp", "mean"), avg_cnt=(TARGET, "mean"),
    )
    print("\n[계절 코드별 월과 평균 기온: season=1은 겨울 패턴]\n", seasons.round(2))

    # GROUP BY hr, workingday로 평균을 낸 뒤 unstack()으로 범례별 가로 열을 만든다.
    hourly = df.groupby(["hr", "workingday"])[TARGET].mean().unstack()
    hourly = hourly.rename(columns={0: "Weekend / Holiday", 1: "Working Day"})
    ax = hourly.plot(marker="o", figsize=(10, 4), xticks=range(24))
    save_plot(ax, output, "01_workingday_vs_weekend.png", "Average Rentals by Hour", "Hour", "Average Rentals")
    for name in hourly:
        print(f"\n[{name} 피크 TOP 3]\n", hourly[name].nlargest(3).round(1).to_string())

    # 극단적 날씨처럼 관측치가 몇 건 없는 그룹은 평균만 보면 해석하기 어렵다.
    weather = df.groupby("weathersit")[TARGET].agg(["count", "mean", "min", "max"])
    weather["ratio(%)"] = weather["count"] / len(df) * 100
    print("\n[날씨별 대여량]\n", weather.round(2))
    yearly = df.groupby("yr")[TARGET].agg(["count", "mean"])
    print("\n[연도 코드별 대여량: 0=2011, 1=2012]\n", yearly.round(2))
    if {0, 1}.issubset(yearly.index):
        growth = (yearly.loc[1, "mean"] / yearly.loc[0, "mean"] - 1) * 100
        print(f"2012년 시간당 평균 대여량: 전년 대비 {growth:+.1f}%")
    monthly = df.groupby(df["dteday"].dt.to_period("M"))[TARGET].mean()
    ax = monthly.plot(marker="o", figsize=(10, 4))
    save_plot(ax, output, "02_monthly_trend.png", "Monthly Rental Trend", "Year-Month", "Average Rentals")

    # 두 컬럼은 예측 시점에 알 수 없는 정답의 구성값이다. 다음 실습에서 일부러 넣어본다.
    totals = df[LEAK_FEATURES].sum(axis=1)
    print(f"\n모든 행에서 casual + registered = cnt? {bool(totals.eq(df[TARGET]).all())}")
    print(df[["dteday", "hr", *LEAK_FEATURES, TARGET]].head().to_string(index=False))

    # 자습 미션: 정규화된 기온을 다섯 구간으로 자른 뒤, 12개 월별 평균과 함께 비교한다.
    temp_bins = pd.cut(df["temp"], bins=5)
    temperature = df.groupby(temp_bins, observed=False)[TARGET].agg(["count", "mean"])
    print("\n[기온 구간별 대여량]\n", temperature.round(2))
    ax = temperature["mean"].plot.bar(figsize=(9, 4), rot=30)
    save_plot(ax, output, "03_temperature.png", "Rentals by Temperature Range", "Normalized Temperature", "Average Rentals")
    by_month = df.groupby("mnth")[TARGET].mean().sort_values(ascending=False)
    print("\n[월별 평균 순위]\n", by_month.round(1).to_string())
    ax = by_month.plot.bar(figsize=(9, 4), rot=0)
    save_plot(ax, output, "04_month_ranking.png", "Average Rentals by Month", "Month", "Average Rentals")


def add_score(benchmark, experiment, label, split, y_train, train_pred, y_test, test_pred):
    """같은 방식으로 점수를 계산해 콘솔과 최종 기록표에 남긴다."""
    # MAE는 평균 절대 오차(대), RMSE는 큰 오차에 더 민감, R²은 평균 대비 설명력이다.
    row = {
        "experiment": experiment, "model": label, "split": split,
        "train_mae": mean_absolute_error(y_train, train_pred),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
        "test_r2": r2_score(y_test, test_pred),
    }
    benchmark.append(row)
    print(f"{experiment:11} {label}: Train MAE {row['train_mae']:.2f} / "
          f"Test MAE {row['test_mae']:.2f}, RMSE {row['test_rmse']:.2f}, R² {row['test_r2']:.4f}")


def evaluate(model, train, test, features, benchmark, experiment, label, split):
    """fit은 학습(빌드), predict는 규칙 적용(런타임)이다."""
    # 컬럼 리스트로 고르면 2차원 DataFrame(X), 컬럼명 하나로 고르면 1차원 Series(y)다.
    x_train, y_train = train[features], train[TARGET]
    x_test, y_test = test[features], test[TARGET]
    model.fit(x_train, y_train)
    train_pred, test_pred = model.predict(x_train), model.predict(x_test)
    add_score(benchmark, experiment, label, split, y_train, train_pred, y_test, test_pred)
    return train_pred, test_pred


def baseline_and_leakage(train, test, benchmark):
    """02. 무조건 평균 찍기 → 정답 누수 재현 → 누수 제거 → 음수 보정."""
    print("\n=== 02. 기준선과 함정 1: 정답 누수 ===")
    # [실행 전 예측] 학습 평균만 찍는 모델과 정답 구성값을 받은 모델은 각각 얼마나 틀릴까?
    evaluate(DummyRegressor(strategy="mean"), train, test, BASE_FEATURES, benchmark,
             "Random base", "학습 평균 기준선", "random")
    leaked = LinearRegression()
    leak_cols = BASE_FEATURES + LEAK_FEATURES
    evaluate(leaked, train, test, leak_cols, benchmark, "Exp 1", "누수 포함 선형회귀 (잘못된 점수)", "random")
    print("[누수 모델 계수]\n", pd.Series(leaked.coef_, index=leak_cols).round(4).to_string())

    # 동일한 행으로 다시 평가한다. 바뀌는 것은 casual/registered 제외 여부뿐이다.
    train_pred, test_pred = evaluate(LinearRegression(), train, test, BASE_FEATURES, benchmark,
                                    "Random LR", "누수 제거 선형회귀", "random")
    sample = pd.DataFrame({"actual": test[TARGET].to_numpy(), "predicted": test_pred})
    sample["error"] = sample["actual"] - sample["predicted"]
    print("\n[실제값 vs 예측값 5건]\n", sample.head().round(1).to_string(index=False))

    # 자습 미션: 대여량은 음수일 수 없으므로 0으로 자른 뒤 같은 테스트에서 다시 채점한다.
    print(f"음수 예측: {(test_pred < 0).sum()}건 / {len(test)}건")
    add_score(benchmark, "Random clip", "선형회귀 + 음수 보정", "random",
              train[TARGET], np.clip(train_pred, 0, None), test[TARGET], np.clip(test_pred, 0, None))


def split_by_date(df, cutoff):
    """기준일 미만만 학습에 사용하고 기준일부터 미래 평가에 사용한다."""
    # SQL의 WHERE와 같은 불리언 필터. 컷오프 당일은 테스트에만 들어간다.
    train = df.loc[df["dteday"] < cutoff]
    test = df.loc[df["dteday"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError(f"{cutoff} 기준으로 나누려면 기준일 전후 데이터가 모두 필요합니다.")
    return train, test


def random_forest():
    """모든 RF 실험에서 같은 설정을 쓴다. 나무 100그루의 예측을 평균낸다."""
    return RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=2)


def validate_timeline(df, train_rand, test_rand, train, test, benchmark):
    """03. 같은 RF로 랜덤 분할과 날짜 분할을 비교할 준비를 한다."""
    print("\n=== 03. 함정 2: 시간 데이터를 랜덤 분할 ===")
    evaluate(random_forest(), train_rand, test_rand, BASE_FEATURES, benchmark,
             "Exp 2", "랜덤 분할 RandomForest (미래 예측 지표 아님)", "random")

    # 행 번호 ±1은 빠진 시간이 있을 때 틀릴 수 있어 실제 타임스탬프 ±1시간을 조회한다.
    timestamps = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    center = timestamps.loc[test_rand.index[0]]
    nearby = df.loc[timestamps.between(center - pd.Timedelta(hours=1), center + pd.Timedelta(hours=1)),
                    ["dteday", "hr", "temp", TARGET]].copy()
    nearby["split"] = np.where(nearby.index.isin(train_rand.index), "Train", "Test")
    print("\n[랜덤 테스트 행의 앞뒤 1시간 소속]\n", nearby.to_string(index=False))

    for name, subset in [("Train", train), ("Test", test)]:
        print(f"{name}: {subset['dteday'].min():%Y-%m-%d} ~ {subset['dteday'].max():%Y-%m-%d} "
              f"({len(subset):,}행, {len(subset) / len(df):.1%}) | 평균 {subset[TARGET].mean():.1f}대")
    evaluate(DummyRegressor(strategy="mean"), train, test, BASE_FEATURES, benchmark,
             "Exp 0", "학습 평균 기준선", SPLIT_DATE)

    # 자습 미션: 7월로 옮기면 학습 기간과 테스트 계절이 함께 바뀐다. 동일 조건의 벤치마크와 구분한다.
    july_train, july_test = split_by_date(df, "2012-07-01")
    july_model = random_forest().fit(july_train[BASE_FEATURES], july_train[TARGET])
    july_mae = mean_absolute_error(july_test[TARGET], july_model.predict(july_test[BASE_FEATURES]))
    print(f"[분할일 미션] 2012-07-01 RF Test MAE: {july_mae:.2f}대 ({len(july_test):,}행)")


def compare_models(train, test, benchmark, output):
    """04. 과소적합과 과적합을 비교하고 오차가 큰 시간의 문맥을 살핀다."""
    print("\n=== 04. 함정 3: 선형 모델의 한계, 결정트리 과적합, 오차 분석 ===")
    # [실행 전 예측] 결정트리가 기출문제(Train)를 외우면 새 시험(Test)도 잘 볼까?
    models = [
        ("Exp 3", "LinearRegression", LinearRegression()),
        ("Exp 4a", "DecisionTree", DecisionTreeRegressor(random_state=SEED)),
        ("Exp 4b", "RandomForest", random_forest()),
    ]
    predictions = {}
    for experiment, name, model in models:
        _, predictions[name] = evaluate(model, train, test, BASE_FEATURES, benchmark, experiment, name, SPLIT_DATE)
    random_mae = next(row["test_mae"] for row in benchmark if row["experiment"] == "Exp 2")
    time_mae = benchmark[-1]["test_mae"]
    print(f"[RF 비교] 랜덤 분할 {random_mae:.2f}대 → {SPLIT_DATE} 시계열 분할 {time_mae:.2f}대")

    # 예측 배열은 test와 행 순서가 같다. 그래프의 x축은 실제 시각을 사용한다.
    chart = pd.DataFrame({
        "timestamp": test["dteday"] + pd.to_timedelta(test["hr"], unit="h"),
        "Actual": test[TARGET], "LinearRegression": predictions["LinearRegression"],
        "RandomForest": predictions["RandomForest"],
    })
    window = chart.loc[chart["timestamp"].between("2012-09-04", "2012-09-07", inclusive="left")]
    ax = window.plot(x="timestamp", style=["k-", "r--", "b-"], figsize=(11, 4))
    save_plot(ax, output, "05_actual_vs_predicted.png", "Actual vs Predicted: Sep 4–6, 2012", "Timestamp", "Rentals")

    # 장애 로그처럼 정보를 모은다. 반올림은 출력에만 적용하고 순위는 원래 오차로 정한다.
    errors = test[["dteday", "hr", "weekday", "workingday", "holiday", "weathersit", "temp", TARGET]].copy()
    errors["pred"] = predictions["RandomForest"]
    errors["error"] = errors[TARGET] - errors["pred"]  # 음수면 수요를 과대 예측했다.
    errors["abs_error"] = errors["error"].abs()
    top_errors = errors.nlargest(10, "abs_error").copy()
    # 날짜 이벤트는 오차를 해석하는 단서일 뿐이다. 테스트를 보고 모델 입력에 하드코딩하지 않는다.
    events = {"2012-11-23": "블랙프라이데이", "2012-12-24": "크리스마스 이브", "2012-10-02": "날씨 변화 확인"}
    top_errors["context"] = top_errors["dteday"].dt.strftime("%Y-%m-%d").map(events).fillna("")
    display_errors = top_errors.round({"temp": 2, "pred": 2, "error": 2, "abs_error": 2})
    print("\n[RF 오차 TOP 10: 발표에는 상위 5행 사용]\n", display_errors.to_string(index=False))
    top_errors.to_csv(output / "top_errors.csv", index=False, encoding="utf-8-sig")

    # 자습 미션: 트리 깊이만 바꾸고 Train/Test 격차와 10월 2일의 시간별 로그를 확인한다.
    print("\n[결정트리 깊이 비교: 여기서 본 Test로 운영 파라미터를 고르지 않는다]")
    for depth in [5, 10, 15, None]:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=SEED).fit(train[BASE_FEATURES], train[TARGET])
        train_mae = mean_absolute_error(train[TARGET], tree.predict(train[BASE_FEATURES]))
        test_mae = mean_absolute_error(test[TARGET], tree.predict(test[BASE_FEATURES]))
        print(f"max_depth={str(depth):4} | Train MAE: {train_mae:6.2f} | Test MAE: {test_mae:6.2f}")
    day = errors.loc[errors["dteday"].eq(pd.Timestamp("2012-10-02")), ["hr", "weathersit", "temp", TARGET, "pred", "error"]]
    print("\n[2012-10-02 시간별 날씨·실제·예측]\n", day.round(2).to_string(index=False))


def add_calendar_features(df):
    """출퇴근 여부와 23시·0시가 가깝다는 원형 정보를 추가한다."""
    # 원본을 보존해야 달력 피처만 바꾼 비교가 가능하다. True/False를 1/0으로 변환한다.
    rush = df["workingday"].eq(1) & df["hr"].isin([7, 8, 9, 17, 18, 19])
    angle = 2 * np.pi * df["hr"] / 24  # 24시간을 원의 0~2π 각도로 옮긴다.
    return df.assign(is_rush_hour=rush.astype(int), hr_sin=np.sin(angle), hr_cos=np.cos(angle))


def add_lag_features(df):
    """이전 1·7·14일 같은 시간의 대여량을 SQL LEFT JOIN처럼 붙인다."""
    # shift(24)는 누락 행 때문에 다른 시간을 참조할 수 있다. 키는 날짜와 시간이다.
    result = df.copy()
    history = df[["dteday", "hr", TARGET]]
    for feature, days in LAG_DAYS.items():
        # 과거 기록의 키를 앞으로 밀면 현재 날짜에서 정확히 days일 전 기록을 조회한다.
        shifted = history.assign(dteday=history["dteday"] + pd.Timedelta(days=days))
        shifted = shifted.rename(columns={TARGET: feature})
        # 과거 기록이 없으면 0대가 아닌 NaN. 중복 키로 행이 늘면 조용히 진행하지 않는다.
        result = result.merge(shifted, on=["dteday", "hr"], how="left", validate="one_to_one")
    return result


def boosting_model():
    """부스팅 설정을 공통으로 사용해 최종 비교에서 과거 이력만 바꾼다."""
    # RF와 달리 앞 나무에서 남은 오차를 다음 나무가 순차적으로 보완한다.
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED)


def predict_bike_rentals(model, request_dto):
    """Spring에서 넘어온 모델 입력 DTO 한 건을 예측한다. HTTP 서버는 띄우지 않는다."""
    # 필드를 학습 순서로 정렬한다. 키 누락을 조용히 통과시키지 않고, 이력 없음은 None으로 전달한다.
    frame = pd.DataFrame([{name: request_dto[name] for name in FINAL_FEATURES}], dtype=float)
    predicted = model.predict(frame)[0]
    return max(0, int(round(predicted)))  # 도메인 규칙: 0 이상 정수


def feature_experiments(df, train, test, benchmark, output):
    """05. 달력 피처 실험 → 부스팅 → 과거 이력 → 중요도·모델 저장·추론."""
    print("\n=== 05. 피처 엔지니어링, 변수 중요도, API 모의 추론 ===")
    # [실행 전 예측] 출퇴근 시간을 따로 표시하면 트리도 반드시 더 잘 맞힐까?
    base_rf_mae = benchmark[-1]["test_mae"]
    calendar_train, calendar_test = split_by_date(add_calendar_features(df), SPLIT_DATE)
    evaluate(random_forest(), calendar_train, calendar_test, BASE_FEATURES + CALENDAR_FEATURES,
             benchmark, "RF calendar", "RandomForest + 달력/주기", SPLIT_DATE)
    print(f"[달력 피처 비교] 기존 RF {base_rf_mae:.2f}대 → 추가 후 {benchmark[-1]['test_mae']:.2f}대. "
          "다음 실험에서는 달력 피처를 빼고 모델만 바꾼다.")
    evaluate(boosting_model(), train, test, BASE_FEATURES, benchmark,
             "Exp 5a", "HistGradientBoosting", SPLIT_DATE)

    # 테스트 기간 안의 과거 실측도 예측 당시 확정됐다면 쓸 수 있다(매 시점에 짧은 미래를 예측).
    # 9월 1일 한 번에 4개월치를 예측하는 상황과 다르다. 24시간 전 값도 예측 시 이미 확정돼야 한다.
    # 노트북처럼 과거 CSV의 관측 날씨로 평가한다. 실제 운영에서는 당시 이용 가능했던 예보로 검증한다.
    with_history = add_lag_features(df)
    lag_train, lag_test = split_by_date(with_history, SPLIT_DATE)
    print("\n[과거 이력 조인 예시]\n",
          lag_test[["dteday", "hr", TARGET, *LAG_DAYS]].head().to_string(index=False))
    final_model = boosting_model()  # HGB는 조인에서 생긴 NaN을 그대로 받을 수 있다.
    evaluate(final_model, lag_train, lag_test, FINAL_FEATURES, benchmark,
             "Exp 5b", "HistGradientBoosting + 과거 이력", SPLIT_DATE)

    # 컬럼 하나를 섞었을 때 MAE가 얼마나 늘어나는지 본다. 의존도를 보여주며 인과관계는 아니다.
    importance = permutation_importance(
        final_model, lag_test[FINAL_FEATURES], lag_test[TARGET],
        scoring="neg_mean_absolute_error", n_repeats=5, random_state=SEED, n_jobs=1,
    )
    ranking = pd.Series(importance.importances_mean, index=FINAL_FEATURES).sort_values()
    ax = ranking.plot.barh(figsize=(9, 5))
    save_plot(ax, output, "06_permutation_importance.png", "Permutation Importance", "MAE Increase (bikes)", "Feature")
    print("\n[순열 중요도 TOP 5]\n", ranking.nlargest(5).round(2).to_string())

    # Spring의 빌드 산출물처럼 모델을 파일에 저장한다. 저장소의 기존 모델은 덮어쓰지 않는다.
    model_path = output / "bike_hgb_model.joblib"
    joblib.dump(final_model, model_path)
    # 실제 서버에서는 기동할 때 한 번 로드한다. joblib에는 신뢰할 수 있는 파일만 사용한다.
    reloaded = joblib.load(model_path)
    # 컬럼명과 값이 함께 보이도록 Spring DTO와 같은 키-값 구조로 작성한다.
    sample = {
        "season": 4, "yr": 1, "mnth": 10, "hr": 8,
        "holiday": 0, "weekday": 2, "workingday": 1,
        "weathersit": 1, "temp": 0.48, "atemp": 0.47, "hum": 0.65, "windspeed": 0.15,
        "cnt_prev_day": 510, "cnt_prev_week": 540, "cnt_prev_2week": 525,
    }
    clear = predict_bike_rentals(reloaded, sample)
    # 자습 미션: 날씨만 바꾸거나 근무일·시간을 바꿔 본다. 실제 주말 요청은 이력 값도 시간에 맞게 조회한다.
    rain = predict_bike_rentals(reloaded, {**sample, "weathersit": 3})
    weekend = predict_bike_rentals(reloaded, {**sample, "workingday": 0, "weekday": 6, "hr": 14})
    print(f"\n[API 모의 추론] 날씨 변경: 평일 08시 맑음 {clear}대 → 비/눈 {rain}대")
    print(f"[API 모의 추론] 주말 오후 2시: {weekend}대 (나머지 입력 고정)")


def main():
    """스크립트의 실행 순서와 최종 결과물을 관리한다."""
    parser = argparse.ArgumentParser(description="자전거 대여량 예측: 회귀 노트북 01~05 통합 실습")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="같은 스키마·기간의 hour.csv 경로")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="그래프·CSV·모델 저장 폴더")
    args = parser.parse_args()

    df = load_data(args.data.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    benchmark = []
    explore_data(df, output)

    # 난수 시드를 고정해서 원본과 같은 80:20 랜덤 홀드아웃을 재현한다. 운영 성능 표가 아니다.
    train_rand, test_rand = train_test_split(df, test_size=0.2, random_state=SEED)
    train, test = split_by_date(df, SPLIT_DATE)
    # OpenMP/BLAS의 과도한 병렬 실행을 막아 일반 PC에서도 실행할 수 있게 한다.
    with threadpool_limits(limits=2):
        baseline_and_leakage(train_rand, test_rand, benchmark)
        validate_timeline(df, train_rand, test_rand, train, test, benchmark)
        compare_models(train, test, benchmark, output)
        feature_experiments(df, train, test, benchmark, output)

    # 하드코딩한 노트북 점수가 아닌 현재 실행의 계산 결과를 남긴다.
    report = pd.DataFrame(benchmark)
    report.to_csv(output / "benchmark.csv", index=False, encoding="utf-8-sig")
    compact = report[["experiment", "model", "split", "train_mae", "test_mae", "test_rmse", "test_r2"]]
    print("\n=== 최종 벤치마크 (랜덤과 시계열 점수는 분리해서 해석) ===")
    print(compact.round(4).to_string(index=False))
    scores = report.set_index("experiment")["test_mae"]
    improvement = (1 - scores["Exp 5b"] / scores["Exp 0"]) * 100
    print(f"\n시계열 기준선 → 최종 모델: {scores['Exp 0']:.2f}대 → {scores['Exp 5b']:.2f}대 (MAE {improvement:.1f}% 감소)")
    print(f"결과 폴더: {output}")


# import할 때는 학습하지 않고, 터미널에서 이 파일을 직접 실행할 때만 수행한다.
if __name__ == "__main__":
    main()
