import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# データ読み込み
df = pd.read_csv("/Users/ishibashikenta/Documents/競馬予想/過去データ/oi_all.csv")

# 数値変換
df["着順"]       = pd.to_numeric(df["着順"], errors="coerce")
df["単勝オッズ"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")
df["人気"]       = pd.to_numeric(df["人気"], errors="coerce")
df["斤量"]       = pd.to_numeric(df["斤量"], errors="coerce")
df["距離"]       = pd.to_numeric(df["距離"], errors="coerce")
df["枠"]         = pd.to_numeric(df["枠"], errors="coerce")
df["日付"]       = pd.to_datetime(df["日付"])
df["馬場状態コード"] = df["馬場状態"].map({"良":0, "稍重":1, "重":2, "不良":3})
df["馬場種類コード"] = df["馬場種類"].map({"芝":0, "ダ":1})

df = df.dropna(subset=["着順"])

# 大穴馬を除外（50倍以上）
df = df[df["単勝オッズ"] <= 50]

# オッズ変換
df["単勝オッズ_log"]     = np.log(df["単勝オッズ"].clip(lower=1))
df["単勝オッズ_081"]     = df["単勝オッズ"] ** 0.81
df["単勝オッズ_081_log"] = np.log(df["単勝オッズ_081"].clip(lower=1))

# 複勝オッズ推定
def estimate_fukusho_odds(tan_odds):
    if tan_odds <= 2.0:    ratio = 0.75
    elif tan_odds <= 4.0:  ratio = 0.45
    elif tan_odds <= 7.0:  ratio = 0.35
    elif tan_odds <= 15.0: ratio = 0.275
    else:                  ratio = 0.225
    return max(tan_odds * ratio, 1.0)

df["複勝オッズ_推定"] = df["単勝オッズ"].apply(estimate_fukusho_odds)

df = df.sort_values("日付").reset_index(drop=True)

df["1着"] = (df["着順"] == 1).astype(int)
df["2着"] = (df["着順"] == 2).astype(int)
df["3着"] = (df["着順"] == 3).astype(int)
df["複勝"] = (df["着順"] <= 3).astype(int)

def estimate_class(odds):
    if odds <= 2.0:    return 0
    elif odds <= 5.0:  return 1
    elif odds <= 10.0: return 2
    elif odds <= 20.0: return 3
    else:              return 4

print("過去成績の特徴量を計算中...")

win_rates            = []
second_rates         = []
third_rates          = []
fukusho_rates        = []
waku_win_rates       = []
jockey_win_rates     = []
jockey_fukusho_rates = []
race_class_list      = []

for idx, row in df.iterrows():
    uma    = row["馬名"]
    jockey = row["騎手"]
    date   = row["日付"]
    track  = row["馬場種類"]
    waku   = row["枠"]
    odds   = row["単勝オッズ"]

    past = df[
        (df["馬名"] == uma) &
        (df["日付"] < date) &
        (df["日付"] >= date - pd.Timedelta(days=365))
    ].tail(5)

    win_rates.append(past["1着"].mean() if len(past) > 0 else None)
    second_rates.append(past["2着"].mean() if len(past) > 0 else None)
    third_rates.append(past["3着"].mean() if len(past) > 0 else None)
    fukusho_rates.append(past["複勝"].mean() if len(past) > 0 else None)

    past_waku = df[
        (df["馬名"] == uma) &
        (df["日付"] < date) &
        (df["馬場種類"] == track) &
        (df["枠"] == waku)
    ]
    waku_win_rates.append(past_waku["1着"].mean() if len(past_waku) > 0 else None)

    past_jockey = df[
        (df["騎手"] == jockey) &
        (df["日付"] < date) &
        (df["日付"] >= date - pd.Timedelta(days=365))
    ]
    jockey_win_rates.append(past_jockey["1着"].mean() if len(past_jockey) > 0 else None)
    jockey_fukusho_rates.append(past_jockey["複勝"].mean() if len(past_jockey) > 0 else None)

    race_class_list.append(estimate_class(odds))

    if idx % 2000 == 0:
        print(f"  {idx}/{len(df)} 処理中...")

df["過去勝率"]     = win_rates
df["過去2着率"]    = second_rates
df["過去3着率"]    = third_rates
df["過去複勝率"]   = fukusho_rates
df["枠番勝率"]     = waku_win_rates
df["騎手勝率"]     = jockey_win_rates
df["騎手複勝率"]   = jockey_fukusho_rates
df["レースクラス"] = race_class_list

print("計算完了！")

# 1着用特徴量（単勝オッズ + log）
features_win = [
    "単勝オッズ", "単勝オッズ_log", "人気", "斤量", "距離",
    "馬場状態コード", "馬場種類コード",
    "枠", "過去勝率", "過去2着率", "過去3着率",
    "過去複勝率", "枠番勝率",
    "騎手勝率", "騎手複勝率", "レースクラス"
]

# 2着・3着用特徴量（0.81乗 + log(0.81乗)）
features_place = [
    "単勝オッズ_081", "単勝オッズ_081_log", "人気", "斤量", "距離",
    "馬場状態コード", "馬場種類コード",
    "枠", "過去勝率", "過去2着率", "過去3着率",
    "過去複勝率", "枠番勝率",
    "騎手勝率", "騎手複勝率", "レースクラス"
]

all_features = list(set(features_win + features_place))
df_model = df.dropna(subset=all_features)
print(f"学習データ: {len(df_model)}行")

split = int(len(df_model) * 0.8)

feature_map = {
    "1着": features_win,
    "2着": features_place,
    "3着": features_place
}

models = {}
for target in ["1着", "2着", "3着"]:
    print(f"\n{target}モデルを学習中...")
    feats = feature_map[target]
    X = df_model[feats]
    y = df_model[target]
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{target}予測精度: {acc:.3f}")
    models[target] = model

# 確率を追加
df_model = df_model.copy()
for target in ["1着", "2着", "3着"]:
    feats = feature_map[target]
    df_model[f"{target}確率"] = models[target].predict_proba(df_model[feats])[:, 1]

# テストデータ
df_test = df_model.iloc[split:].copy()
for target in ["1着", "2着", "3着"]:
    feats = feature_map[target]
    df_test[f"{target}確率"] = models[target].predict_proba(df_test[feats])[:, 1]

df_test["EV_単勝"] = df_test["1着確率"] * df_test["単勝オッズ"]

# 正規化
for col in ["1着確率", "2着確率", "3着確率"]:
    df_test[f"{col}_正規化"] = df_test.groupby("race_id")[col].transform(
        lambda x: x / x.sum()
    )
df_test["EV_正規化"] = df_test["1着確率_正規化"] * df_test["単勝オッズ"]

# 閾値別バックテスト
print("\n=== 閾値別バックテスト（単勝）===")
for threshold in [1.0, 1.5, 2.0]:
    ev_t = df_test[df_test["EV_単勝"] > threshold].copy()
    if len(ev_t) == 0:
        continue
    ev_t["的中"] = (ev_t["着順"] == 1).astype(int)
    ev_t["払戻"] = ev_t["的中"] * ev_t["単勝オッズ"] * 100
    bet = len(ev_t) * 100
    ret = ev_t["払戻"].sum()
    roi = ret / bet * 100
    print(f"EV>{threshold}: {len(ev_t)}頭 回収率{roi:.1f}% 的中{int(ev_t['的中'].sum())}回")

# 正規化後バックテスト
print("\n=== 正規化後バックテスト ===")
for threshold in [1.0, 1.2, 1.5, 2.0]:
    ev_t = df_test[df_test["EV_正規化"] > threshold].copy()
    if len(ev_t) == 0:
        continue
    ev_t["的中"] = (ev_t["着順"] == 1).astype(int)
    ev_t["払戻"] = ev_t["的中"] * ev_t["単勝オッズ"] * 100
    bet = len(ev_t) * 100
    ret = ev_t["払戻"].sum()
    roi = ret / bet * 100
    print(f"EV>{threshold}: {len(ev_t)}頭 回収率{roi:.1f}% 的中{int(ev_t['的中'].sum())}回")

# 各レース最高EV馬だけ買う戦略
print("\n=== 各レース最高EV馬だけ買う戦略 ===")
best_per_race = df_test.loc[df_test.groupby("race_id")["EV_正規化"].idxmax()].copy()
best_per_race["的中"] = (best_per_race["着順"] == 1).astype(int)
best_per_race["払戻"] = best_per_race["的中"] * best_per_race["単勝オッズ"] * 100
bet = len(best_per_race) * 100
ret = best_per_race["払戻"].sum()
roi = ret / bet * 100
print(f"対象レース数: {len(best_per_race)}レース")
print(f"投資額合計:   {bet:,}円")
print(f"払戻合計:     {ret:,}円")
print(f"回収率:       {roi:.1f}%")
print(f"的中数:       {int(best_per_race['的中'].sum())}回")

# 複勝バックテスト
print("\n=== 複勝バックテスト ===")
df_test["複勝確率"] = df_test["1着確率_正規化"] + df_test["2着確率_正規化"] + df_test["3着確率_正規化"]
df_test["EV_複勝"] = df_test["複勝確率"] * df_test["複勝オッズ_推定"]

for threshold in [1.0, 1.2, 1.5]:
    ev_t = df_test[df_test["EV_複勝"] > threshold].copy()
    if len(ev_t) == 0:
        continue
    ev_t["的中"] = (ev_t["着順"] <= 3).astype(int)
    ev_t["払戻"] = ev_t["的中"] * ev_t["複勝オッズ_推定"] * 100
    bet = len(ev_t) * 100
    ret = ev_t["払戻"].sum()
    roi = ret / bet * 100
    print(f"EV>{threshold}: {len(ev_t)}頭 回収率{roi:.1f}% 的中{int(ev_t['的中'].sum())}回")

# 馬連バックテスト
print("\n=== 馬連バックテスト ===")
race_ids = df_test["race_id"].unique()
umaren_results = []

for race_id in race_ids:
    race = df_test[df_test["race_id"] == race_id].copy()
    if len(race) < 2:
        continue

    top2 = race.nlargest(2, "1着確率_正規化")
    uma1 = top2.iloc[0]
    uma2 = top2.iloc[1]

    umaren_odds = uma1["単勝オッズ"] * uma2["単勝オッズ"] * 0.8
    umaren_odds = min(umaren_odds, 200)

    actual_1 = race[race["着順"] == 1]["馬名"].values
    actual_2 = race[race["着順"] == 2]["馬名"].values
    top2_names = set(top2["馬名"].values)

    hit = 0
    if len(actual_1) > 0 and len(actual_2) > 0:
        if actual_1[0] in top2_names and actual_2[0] in top2_names:
            hit = 1

    umaren_results.append({
        "race_id": race_id,
        "的中": hit,
        "払戻": hit * umaren_odds * 100,
        "オッズ推定": umaren_odds
    })

df_umaren = pd.DataFrame(umaren_results)
bet = len(df_umaren) * 100
ret = df_umaren["払戻"].sum()
roi = ret / bet * 100
print(f"対象レース数: {len(df_umaren)}レース")
print(f"投資額合計:   {bet:,}円")
print(f"払戻合計:     {ret:,}円")
print(f"回収率:       {roi:.1f}%")
print(f"的中数:       {int(df_umaren['的中'].sum())}回")

# モデル保存
for target in ["1着", "2着", "3着"]:
    joblib.dump(models[target], f"/Users/ishibashikenta/Documents/keiba_model_{target}.pkl")
print("\nモデルを保存しました！")