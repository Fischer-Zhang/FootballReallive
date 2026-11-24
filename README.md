# footballsim — 屬性驅動足球比賽模擬

以 Python 實作的足球比賽模擬引擎：60 項球員屬性透過「位置 → 決策 → 執行 → 狀態」四層機制影響每個動作，並以真實聯賽統計校準。

- 設計方案：[docs/design-plan.md](docs/design-plan.md)
- 校準現況與已知差距：[docs/calibration-status.md](docs/calibration-status.md)

## 安裝與執行

需要 Python ≥ 3.10、numpy、pydantic v2、PyYAML（計算校準目標另需 pandas）。

```bash
# 不安裝，直接執行
export PYTHONPATH=src
python -m footballsim league                      # 生成 20 隊虛構聯賽 → data/league.json
python -m footballsim match --home 0 --away 5     # 模擬一場並輸出中文報告（--events 列出事件）
python -m footballsim calibrate --matches 100     # 批次模擬並與五大聯賽實測統計比對
python -m footballsim sensitivity finishing --groups ST,W,AM --mirror   # 屬性敏感度（加強版 vs 原版）
python -m footballsim targets fetch               # 重新下載 football-data.co.uk 並計算比賽層目標
python -m footballsim statsbomb fetch             # 下載 StatsBomb 開放資料（2015/16 五大聯賽抽 100 場）
python -m footballsim statsbomb fit               # 擬合 xT 網格、xG、事件層目標、射門情境模型
python -m footballsim statsbomb fetch360          # 下載 StatsBomb 360 位置資料（Euro 2024、WC 2022 各 25 場）
python -m footballsim statsbomb fit360            # 擬合禁區跑位、傳球成功率與高度、射門／傳中／解圍／帶球傾向、傳球對象

python -m pytest                                  # 測試（約 27 秒）
```

也可以 `pip install -e .` 後使用 `footballsim` 指令。

**低負荷**：`calibrate` 與 `sensitivity` 預設只用 1 個行程（上限 2），以最低排程優先權執行，每個行程有 3 GB 記憶體上限（`FOOTBALLSIM_MEM_GB`），數值函式庫強制單執行緒，失控的比賽會自動中止。單場約 2 秒，100 場約 3.3 分鐘。批次模擬曾兩度導致主機當機，請不要調高行程數，建議每批 ≤ 150 場。

## 專案結構

```
config/engine_params.yaml     所有引擎係數（無魔術數字，可用 --set 覆寫）
config/xt_grid.yaml           StatsBomb 擬合的 16×12 xT 網格
config/fitted.yaml            StatsBomb 2015/16 擬合的射門情境模型、轉移機率（覆蓋 engine_params 的同名項目）
config/fitted360.yaml         StatsBomb 360 擬合：傳球成功率、傳球高度、射門／傳中／解圍／帶球傾向、傳球對象（再覆蓋一次）
config/box_runs.yaml          360 資料中，球在前場時禁區與前沿的真實攻方站位組合（禁區跑位）
config/targets/               真實校準目標（top5.yaml：football-data；event_level.yaml：StatsBomb）
src/footballsim/
  model/      attributes.py（60 項屬性的唯一定義）、positions.py（位置、評分、陣型模板）
  data/       schema.py（pydantic 資料模型）、generator.py（球員/球隊/聯賽生成）、repository.py
  engine/     match.py（主迴圈）、shape.py（陣型快照、跑位、盯人、壓迫、位置連續性）、
              decision.py（選項與 softmax 決策）、actions.py（動作解析）、restarts.py（定位球）、
              referee.py（犯規與出牌）、fatigue.py、manager.py（換人、比分情境、傷病）、core_math.py
  analytics/  stats.py（技術統計、助攻、評分）、report.py（中文比賽報告）
  calibration/ targets.py、statsbomb.py（事件資料下載抽取）、fit.py（xT、xG、事件層目標擬合）、
              statsbomb360.py（360 位置資料下載抽取）、fit360.py（站位與行為模型擬合）、
              runner.py（批次與比對）、sensitivity.py（屬性敏感度）
tests/        核心公式、資料、引擎不變量、校準工具
```

## 目前狀態

- 已完成：M0、M1 引擎，以及 M4 的資料擬合（xT 網格、射門情境模型、行為模型、各種情況後的轉移機率）。
- **不使用全域偏好補丁**：決策偏好只保留戰術指示與球員特質。持球者的行為（要不要射門、傳中、解圍、帶球，以及傳給誰）由 StatsBomb 360 擬合的行為模型決定；成功與否由位置、物理與屬性決定。
- 第四輪：用 360 位置資料比對引擎在同一時刻的站位，修正帶球封堵、禁區跑位、盯人鬆緊、後衛線深度、跑動路線傳球與傳中機制。
- 第五輪（推進路線）：
  - 傳球成功率與傳球高度改用 360 擬合，取代手設係數。
  - 持球方最前 6 人照 360 的縱向層次排列。
  - 盯人與逼搶以 360 的空檔分布校準。
  - 加入帶球傾向與傳球對象模型。
  - 持球節奏依資料的動作密度校準。
- 第六輪（禁區前的防守）：
  - 防線改為快速後退，並設退守下限（禁區線附近）。
  - 區塊壓縮提早開始。
  - 帶球封堵以 ≥ 3 m 的帶球重新擬合，並加入「貼身者邊退邊守」。
  - 以上全部依 360 校準，比較時引擎套用相同可視範圍。
- 第七輪：
  - 禁區＋前沿的攻方站位完全照真實畫面。
  - 吃過黃牌的球員會收斂（依真實「兩黃變紅」比例校準）。
- 校準（100 場）：42 項中 18 項在範圍內、14 項接近、10 項偏離。進球數、主客勝負、射正與進球率、紅牌都已接近真實。目前最大的問題是很少推進到底線附近，所以傳中、角球、頭槌偏少：禁區內防守分布與真實相反，中場打進兩側走廊的長傳也幾乎不存在。另外本方三區太保守、球門球全部短傳。詳見 [docs/calibration-status.md](docs/calibration-status.md)。
- 尚未完成：上述問題、過人傾向、主場優勢的範圍、VAEP 評分、視覺化、賽季層；屬性敏感度需在新版本重新量測。

資料來源：football-data.co.uk（賽果與技術統計）；StatsBomb Open Data（事件資料，使用須標註來源）。球員與球隊皆為程式生成的虛構資料。
