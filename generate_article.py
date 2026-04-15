import pandas as pd
import numpy as np
import joblib
import itertools
import requests
from bs4 import BeautifulSoup
import re
import urllib3
import warnings
from datetime import datetime
import time
warnings.filterwarnings("ignore")
urllib3.disable_warnings()

# モデル読み込み
model_1 = joblib.load("/Users/ishibashikenta/Documents/keiba_model_1着.pkl")
model_2 = joblib.load("/Users/ishibashikenta/Documents/keiba_model_2着.pkl")
model_3 = joblib.load("/Users/ishibashikenta/Documents/keiba_model_3着.pkl")

# 過去データ読み込み
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
    if tan_odds <= 2.0:    ratio = 0.75
    elif tan_odds <= 4.0:  ratio = 0.45
    elif tan_odds <= 7.0:  ratio = 0.35
    elif tan_odds <= 15.0: ratio = 0.275
    else:                  ratio = 0.225
    return max(tan_odds * ratio, 1.0)

def scrape_shutuba(race_id):
    url = f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_list"
    response = requests.get(url, headers=headers, verify=False)
    response.encoding = "EUC-JP"
    soup = BeautifulSoup(response.text, "html.parser")

    race_info_div = soup.find("div", class_="RaceData01")
    if not race_info_div:
        return None, None

    race_info_text = race_info_div.get_text(strip=True)
    distance   = re.search(r'(\d+)m', race_info_text)
    distance   = int(distance.group(1)) if distance else 1600
    track_type = "ダ" if "ダ" in race_info_text else "芝"
    condition  = re.search(r'馬場:(\S+)', race_info_text)
    condition  = condition.group(1) if condition else "良"

    race_name_div = soup.find("div", class_="RaceName")
    race_name = race_name_div.get_text(strip=True) if race_name_div else f"第{race_id[-2:]}R"

    race_info = {
        "レース名": race_name,
        "距離":     distance,
        "馬場種類": track_type,
        "馬場状態": condition,
    }

    table = soup.find("table", class_="ShutubaTable")
    if not table:
        return None, None

    horses = []
    for tr in table.find_all("tr"):
        cols = tr.find_all("td")
        if len(cols) < 8:
            continue
        try:
            waku     = cols[0].get_text(strip=True)
            uma_name = cols[3].find("a").get_text(strip=True) if cols[3].find("a") else cols[3].get_text(strip=True)
            kinryo   = cols[5].get_text(strip=True)
            jockey   = cols[6].find("a").get_text(strip=True) if cols[6].find("a") else cols[6].get_text(strip=True)
            odds_td  = cols[9] if len(cols) > 9 else None
            odds = None
            if odds_td:
                try:
                    odds = float(odds_td.get_text(strip=True))
                except:
                    odds = None

            if uma_name and waku:
                horses.append({
                    "枠":        int(waku) if waku.isdigit() else 1,
                    "馬名":      uma_name,
                    "斤量":      float(kinryo) if kinryo else 56.0,
                    "騎手":      jockey,
                    "単勝オッズ": odds
                })
        except:
            continue

    return race_info, horses

def add_features(df, date):
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

def analyze_race(race_id, date):
    race_info, horses = scrape_shutuba(race_id)
    if not race_info or not horses:
        return None

    df = pd.DataFrame(horses)
    df["距離"]     = race_info["距離"]
    df["馬場種類"] = race_info["馬場種類"]
    df["馬場状態"] = race_info["馬場状態"]

    df = df[df["単勝オッズ"].notna()]
    df = df[df["単勝オッズ"] <= 50].reset_index(drop=True)
    if len(df) == 0:
        return None

    df["単勝オッズ_log"]     = np.log(df["単勝オッズ"].clip(lower=1))
    df["単勝オッズ_081"]     = df["単勝オッズ"] ** 0.81
    df["単勝オッズ_081_log"] = np.log(df["単勝オッズ_081"].clip(lower=1))
    df["複勝オッズ_推定"]    = df["単勝オッズ"].apply(estimate_fukusho_odds)
    df["レースクラス"]       = df["単勝オッズ"].apply(estimate_class)

    df = df.sort_values("単勝オッズ").reset_index(drop=True)
    df["人気"] = range(1, len(df) + 1)
    df = add_features(df, date)

    features_win = [
        "単勝オッズ", "単勝オッズ_log", "人気", "斤量", "距離",
        "馬場状態コード", "馬場種類コード",
        "枠", "過去勝率", "過去2着率", "過去3着率",
        "過去複勝率", "枠番勝率",
        "騎手勝率", "騎手複勝率", "レースクラス"
    ]
    features_place = [
        "単勝オッズ_081", "単勝オッズ_081_log", "人気", "斤量", "距離",
        "馬場状態コード", "馬場種類コード",
        "枠", "過去勝率", "過去2着率", "過去3着率",
        "過去複勝率", "枠番勝率",
        "騎手勝率", "騎手複勝率", "レースクラス"
    ]

    df["1着確率"] = model_1.predict_proba(df[features_win])[:, 1]
    df["2着確率"] = model_2.predict_proba(df[features_place])[:, 1]
    df["3着確率"] = model_3.predict_proba(df[features_place])[:, 1]

    # 正規化なしEV
    df["EV_単勝"]  = df["1着確率"] * df["単勝オッズ"]
    df["複勝確率"] = df["1着確率"] + df["2着確率"] + df["3着確率"]
    df["EV_複勝"]  = df["複勝確率"] * df["複勝オッズ_推定"]

    # 馬連（正規化なし確率で計算）
    top5 = df.nlargest(5, "1着確率")
    umaren_list = []
    for (i, r1), (j, r2) in itertools.combinations(top5.iterrows(), 2):
        prob = r1["1着確率"] * r2["2着確率"] + r2["1着確率"] * r1["2着確率"]
        odds_est = min(r1["単勝オッズ"] * r2["単勝オッズ"] * 0.8, 200)
        ev = prob * odds_est
        umaren_list.append({
            "馬1": r1["馬名"], "馬2": r2["馬名"],
            "的中確率": prob, "推定オッズ": odds_est, "EV": ev
        })
    df_umaren = pd.DataFrame(umaren_list).sort_values("EV", ascending=False)

    return {
        "race_info": race_info,
        "df": df,
        "df_umaren": df_umaren,
        "race_id": race_id
    }

def generate_article(results, date):
    date_jp = datetime.strptime(date, "%Y/%m/%d").strftime("%Y年%m月%d日")

    article = f"""# 🏇【大井競馬AI予想】{date_jp}

> この記事はAI（機械学習）による大井競馬の予想です。過去3年分のレースデータをもとに、各馬の勝率・複勝率・期待値を算出しています。
>
> ⚠️ **オッズについての注意**：本記事で使用しているオッズは**予想オッズ**です。実際の確定オッズとは異なる場合があります。馬券購入の際は必ず確定オッズをご確認ください。

---

## 📊 AIとは？予想の見方

このAIは以下のデータをもとに予測しています：

- 過去1年以内の成績（勝率・複勝率）
- 騎手の勝率・複勝率
- 距離・馬場・枠番との相性
- 単勝オッズ（市場の評価）

**EV（期待値）** とは、100円賭けたときに平均でいくら返ってくるかを示す指標です。EV>1.0なら理論上プラス収支になります。

---

## 🎯 本日の注目レース・推奨馬

"""

    for result in results:
        if result is None:
            continue
        race_info = result["race_info"]
        df = result["df"]
        race_id = result["race_id"]
        race_num = int(race_id[-2:])

        top3 = df.nlargest(3, "EV_単勝")

        article += f"### 第{race_num}R｜{race_info['レース名']}（{race_info['距離']}m {race_info['馬場種類']} {race_info['馬場状態']}）\n\n"
        article += "**🤖 AI注目馬TOP3**\n\n"

        medals = ["🥇", "🥈", "🥉"]
        for i, (_, row) in enumerate(top3.iterrows()):
            article += f"{medals[i]} **{row['馬名']}**（{int(row['人気'])}番人気 予想オッズ{row['単勝オッズ']}倍）\n"
            article += f"　→ AI1着確率: {row['1着確率']:.1%}　EV: {row['EV_単勝']:.2f}\n\n"

        article += "---\n\n"

    article += """---

## 🔒 有料部分｜全馬の詳細データ・買い目

> ここから先は有料コンテンツです。全馬の1着・2着・3着確率と、単勝・複勝・馬連の具体的な買い目を掲載しています。

"""

    for result in results:
        if result is None:
            continue
        race_info = result["race_info"]
        df = result["df"]
        df_umaren = result["df_umaren"]
        race_id = result["race_id"]
        race_num = int(race_id[-2:])

        article += f"### 第{race_num}R 詳細データ\n\n"
        article += "| 馬名 | 人気 | 予想オッズ | 1着確率 | 2着確率 | 3着確率 | EV単勝 | EV複勝 |\n"
        article += "|---|---|---|---|---|---|---|---|\n"

        for _, row in df.sort_values("人気").iterrows():
            article += f"| {row['馬名']} | {int(row['人気'])} | {row['単勝オッズ']} | {row['1着確率']:.1%} | {row['2着確率']:.1%} | {row['3着確率']:.1%} | {row['EV_単勝']:.2f} | {row['EV_複勝']:.2f} |\n"

        # 単勝推奨
        rec_tan = df[df["EV_単勝"] > 1.0].sort_values("EV_単勝", ascending=False)
        if len(rec_tan) > 0:
            article += "\n**✅ 単勝推奨（EV>1.0）**\n\n"
            for _, r in rec_tan.iterrows():
                article += f"- {r['馬名']}（EV={r['EV_単勝']:.2f} 予想オッズ{r['単勝オッズ']}倍）\n"

        # 複勝推奨
        rec_fuku = df[df["EV_複勝"] > 1.2].sort_values("EV_複勝", ascending=False)
        if len(rec_fuku) > 0:
            article += "\n**✅ 複勝推奨（EV>1.2）**\n\n"
            for _, r in rec_fuku.iterrows():
                article += f"- {r['馬名']}（EV={r['EV_複勝']:.2f} 推定複勝オッズ{r['複勝オッズ_推定']:.1f}倍）\n"

        # 馬連推奨
        article += "\n**✅ 馬連推奨（上位3点）**\n\n"
        for _, r in df_umaren.head(3).iterrows():
            article += f"- {r['馬1']} - {r['馬2']}（EV={r['EV']:.2f} 推定オッズ{r['推定オッズ']:.0f}倍）\n"

        article += "\n---\n\n"

    article += """---

## 📈 回収率の記録

AIの予測精度を透明に公開しています。

| 期間 | 的中率 | 回収率 |
|---|---|---|
| バックテスト（2022〜2024年） | - | 単勝EV>1.5: 141.9% |

> ※実際の運用結果は随時更新します

---

*予想はAIによるものであり、投資を推奨するものではありません。馬券購入は自己責任でお願いします。*
"""

    return article

def generate_x_post(results, date):
    date_jp = datetime.strptime(date, "%Y/%m/%d").strftime("%m月%d日")
    post = f"🏇 {date_jp} 大井競馬AI予想\n\n"

    for result in results[:3]:
        if result is None:
            continue
        race_id = result["race_id"]
        race_num = int(race_id[-2:])
        df = result["df"]

        top1 = df.nlargest(1, "EV_単勝").iloc[0]
        post += f"【第{race_num}R】{top1['馬名']}（予想オッズ{top1['単勝オッズ']}倍）EV={top1['EV_単勝']:.2f}\n"

    post += "\n詳細はnoteで👇\n#大井競馬 #競馬予想 #AI予想"
    return post


# ===== メイン処理 =====

date = datetime.today().strftime("%Y/%m/%d")
date_str = datetime.today().strftime("%Y%m%d")
place = "44"

print(f"=== {date} 大井競馬 記事生成 ===\n")

results = []
for race_num in range(1, 13):
    race_id = f"2026{place}{date_str[4:]}{race_num:02d}"
    print(f"取得中: {race_id}")

    result = analyze_race(race_id, date)
    if result:
        results.append(result)
        print(f"  → 取得成功: {result['race_info']['レース名']}")
    else:
        print(f"  → スキップ")

    time.sleep(2)

if len(results) == 0:
    print("取得できたレースがありません")
else:
    # note記事生成
    article = generate_article(results, date)
    filename = f"note_{date_str}.md"
    with open(f"/Users/ishibashikenta/Documents/競馬予想/{filename}", "w", encoding="utf-8") as f:
        f.write(article)
    print(f"\n✅ note記事を保存しました: {filename}")

    # X投稿文生成
    x_post = generate_x_post(results, date)
    print(f"\n=== X投稿文 ===\n{x_post}")