import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import joblib
import warnings
import urllib3
warnings.filterwarnings("ignore")
urllib3.disable_warnings()

# モデル読み込み
model_1 = joblib.load("/Users/ishibashikenta/Documents/keiba_model_1着.pkl")
model_2 = joblib.load("/Users/ishibashikenta/Documents/keiba_model_2着.pkl")
model_3 = joblib.load("/Users/ishibashikenta/Documents/keiba_model_3着.pkl")

df_all = pd.read_csv("/Users/ishibashikenta/Documents/競馬予想/過去データ/oi_all.csv")

# 過去データ読み込み（特徴量計算用）
df_all = pd.read_csv("/Users/ishibashikenta/Documents/競馬予想/過去データ/oi_all.csv")
df_all["着順"]       = pd.to_numeric(df_all["着順"], errors="coerce")
df_all["単勝オッズ"] = pd.to_numeric(df_all["単勝オッズ"], errors="coerce")
df_all["日付"]       = pd.to_datetime(df_all["日付"])
df_all["1着"] = (df_all["着順"] == 1).astype(int)
df_all["2着"] = (df_all["着順"] == 2).astype(int)
df_all["3着"] = (df_all["着順"] == 3).astype(int)
df_all["複勝"] = (df_all["着順"] <= 3).astype(int)

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def estimate_class(odds):
    if odds <= 2.0:    return 0
    elif odds <= 5.0:  return 1
    elif odds <= 10.0: return 2
    elif odds <= 20.0: return 3
    else:              return 4

def estimate_fukusho_odds(tan_odds):
    if tan_odds <= 2.0:   ratio = 0.75
    elif tan_odds <= 4.0: ratio = 0.45
    elif tan_odds <= 7.0: ratio = 0.35
    elif tan_odds <= 15.0: ratio = 0.275
    else:                  ratio = 0.225
    return max(tan_odds * ratio, 1.0)

def scrape_shutuba(race_id):
    """出馬表を取得"""
    url = f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_list"
    response = requests.get(url, headers=headers, verify=False)
    response.encoding = "EUC-JP"
    soup = BeautifulSoup(response.text, "html.parser")

    # レース情報
    race_info = soup.find("div", class_="RaceData01")
    if not race_info:
        print(f"レース情報が取得できません: {race_id}")
        return None
    race_info_text = race_info.get_text(strip=True)

    import re
    distance   = re.search(r'(\d+)m', race_info_text)
    distance   = int(distance.group(1)) if distance else None
    track_type = "ダ" if "ダ" in race_info_text else "芝"
    condition  = re.search(r'馬場:(\S+)', race_info_text)
    condition  = condition.group(1) if condition else "良"

    # 出馬表テーブル
    table = soup.find("table", class_="Shutuba_Table")
    if not table:
        print(f"出馬表が取得できません: {race_id}")
        return None

    horses = []
    for tr in table.find_all("tr", class_=re.compile("HorseList")):
        cols = tr.find_all("td")
        if len(cols) < 5:
            continue
        try:
            waku     = cols[0].get_text(strip=True)
            umaban   = cols[1].get_text(strip=True)
            uma_name = cols[3].get_text(strip=True)
            seire    = cols[4].get_text(strip=True)
            kinryo   = cols[5].get_text(strip=True)
            jockey   = cols[6].get_text(strip=True)
            horses.append({
                "枠": waku,
                "馬番": umaban,
                "馬名": uma_name,
                "性齢": seire,
                "斤量": kinryo,
                "騎手": jockey,
                "距離": distance,
                "馬場種類": track_type,
                "馬場状態": condition,
                "race_id": race_id
            })
        except:
            continue

    return pd.DataFrame(horses)

def add_features(df, date):
    """特徴量を追加"""
    date = pd.to_datetime(date)

    df["枠"]      = pd.to_numeric(df["枠"], errors="coerce")
    df["斤量"]    = pd.to_numeric(df["斤量"], errors="coerce")
    df["距離"]    = pd.to_numeric(df["距離"], errors="coerce")
    df["馬場状態コード"] = df["馬場状態"].map({"良":0, "稍重":1, "重":2, "不良":3})
    df["馬場種類コード"] = df["馬場種類"].map({"芝":0, "ダ":1})

    win_rates = []
    second_rates = []
    third_rates = []
    fukusho_rates = []
    waku_win_rates = []
    jockey_win_rates = []
    jockey_fukusho_rates = []

    for _, row in df.iterrows():
        uma    = row["馬名"]
        jockey = row["騎手"]
        track  = row["馬場種類"]
        waku   = row["枠"]

        past = df_all[
            (df_all["馬名"] == uma) &
            (df_all["日付"] < date) &
            (df_all["日付"] >= date - pd.Timedelta(days=365))
        ].tail(5)

        win_rates.append(past["1着"].mean() if len(past) > 0 else 0)
        second_rates.append(past["2着"].mean() if len(past) > 0 else 0)
        third_rates.append(past["3着"].mean() if len(past) > 0 else 0)
        fukusho_rates.append(past["複勝"].mean() if len(past) > 0 else 0)

        past_waku = df_all[
            (df_all["馬名"] == uma) &
            (df_all["日付"] < date) &
            (df_all["馬場種類"] == track) &
            (df_all["枠"] == waku)
        ]
        waku_win_rates.append(past_waku["1着"].mean() if len(past_waku) > 0 else 0)

        past_jockey = df_all[
            (df_all["騎手"] == jockey) &
            (df_all["日付"] < date) &
            (df_all["日付"] >= date - pd.Timedelta(days=365))
        ]
        jockey_win_rates.append(past_jockey["1着"].mean() if len(past_jockey) > 0 else 0)
        jockey_fukusho_rates.append(past_jockey["複勝"].mean() if len(past_jockey) > 0 else 0)

    df["過去勝率"]   = win_rates
    df["過去2着率"]  = second_rates
    df["過去3着率"]  = third_rates
    df["過去複勝率"] = fukusho_rates
    df["枠番勝率"]   = waku_win_rates
    df["騎手勝率"]   = jockey_win_rates
    df["騎手複勝率"] = jockey_fukusho_rates

    return df

def predict_race(race_id, date, tan_odds_dict):
    """
    race_id: レースID
    date: レース日付（例: "2026/04/15"）
    tan_odds_dict: 馬名→単勝オッズの辞書
    """
    df = scrape_shutuba(race_id)
    if df is None or len(df) == 0:
        print("出馬表の取得に失敗しました")
        return

    # オッズを追加
    df["単勝オッズ"] = df["馬名"].map(tan_odds_dict)
    df = df[df["単勝オッズ"] <= 50]  # 大穴除外
    df["単勝オッズ_081"] = df["単勝オッズ"] ** 0.81
    df["複勝オッズ_推定"] = df["単勝オッズ"].apply(estimate_fukusho_odds)
    df["レースクラス"] = df["単勝オッズ"].apply(estimate_class)

    # 特徴量追加
    df = add_features(df, date)

    features_win = [
        "単勝オッズ", "人気", "斤量", "距離",
        "馬場状態コード", "馬場種類コード",
        "枠", "過去勝率", "過去2着率", "過去3着率",
        "過去複勝率", "枠番勝率",
        "騎手勝率", "騎手複勝率", "レースクラス"
    ]
    features_place = [
        "単勝オッズ_081", "人気", "斤量", "距離",
        "馬場状態コード", "馬場種類コード",
        "枠", "過去勝率", "過去2着率", "過去3着率",
        "過去複勝率", "枠番勝率",
        "騎手勝率", "騎手複勝率", "レースクラス"
    ]

    # 人気を単勝オッズ順に設定
    df = df.sort_values("単勝オッズ").reset_index(drop=True)
    df["人気"] = range(1, len(df) + 1)

    df["1着確率"] = model_1.predict_proba(df[features_win])[:, 1]
    df["2着確率"] = model_2.predict_proba(df[features_place])[:, 1]
    df["3着確率"] = model_3.predict_proba(df[features_place])[:, 1]

    # 正規化
    df["1着確率_正規化"] = df["1着確率"] / df["1着確率"].sum()
    df["2着確率_正規化"] = df["2着確率"] / df["2着確率"].sum()
    df["3着確率_正規化"] = df["3着確率"] / df["3着確率"].sum()

    # EV計算
    df["EV_単勝"]  = df["1着確率_正規化"] * df["単勝オッズ"]
    df["複勝確率"] = df["1着確率_正規化"] + df["2着確率_正規化"] + df["3着確率_正規化"]
    df["EV_複勝"]  = df["複勝確率"] * df["複勝オッズ_推定"]

    # 結果表示
    print(f"\n{'='*60}")
    print(f"レースID: {race_id}  日付: {date}")
    print(f"{'='*60}")
    print(df[["馬名","人気","単勝オッズ","1着確率_正規化","2着確率_正規化","3着確率_正規化","EV_単勝","EV_複勝"]]
          .sort_values("EV_単勝", ascending=False)
          .to_string(index=False))

    print("\n--- 単勝推奨馬（EV>1.0）---")
    rec = df[df["EV_単勝"] > 1.0].sort_values("EV_単勝", ascending=False)
    if len(rec) > 0:
        for _, r in rec.iterrows():
            print(f"  {r['馬名']} EV={r['EV_単勝']:.2f} 1着確率={r['1着確率_正規化']:.1%} オッズ={r['単勝オッズ']}")
    else:
        print("  該当なし")

    print("\n--- 複勝推奨馬（EV>1.2）---")
    rec_f = df[df["EV_複勝"] > 1.2].sort_values("EV_複勝", ascending=False)
    if len(rec_f) > 0:
        for _, r in rec_f.iterrows():
            print(f"  {r['馬名']} EV={r['EV_複勝']:.2f} 複勝確率={r['複勝確率']:.1%} 推定複勝オッズ={r['複勝オッズ_推定']:.1f}")
    else:
        print("  該当なし")

    return df


# ===== 使い方 =====
# race_idと日付、オッズを入力して実行

race_id = "202444041501"  # ← ここを明日のレースIDに変更
date    = "2026/04/15"    # ← ここを明日の日付に変更

# 馬名→単勝オッズの辞書（前日オッズを手動入力）
tan_odds = {
    "馬名A": 2.5,
    "馬名B": 4.0,
    "馬名C": 6.0,
    # 出走馬全頭分を入力
}

predict_race(race_id, date, tan_odds)