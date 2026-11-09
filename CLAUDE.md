# BBO 投球分析：開發說明（給之後接手的 AI／開發者）

《全民打棒球2》（Windows 視窗模式）即時讀畫面，逐球記錄對手投球落點、球種、球速、結果，網頁顯示分析與下一球預測。使用者：KDBear 與朋友（常玩 2 對 2）。所有回覆用繁體中文。

## 執行流程
- `main.py`：擷取迴圈（前景用 dxcam/mss、背景用 WGC `capture/wgc.py`），每幀縮放成 **1680 寬**後交給 `state/tracker.py`；同時啟動 FastAPI 網頁（`web/app.py` + `web/static/index.html`）。
- `state/tracker.py` `GameSessionTracker.process_frame()`：核心狀態機。
  - 記分板（`detect/scoreboard.py`）：局數、上下半、B/S/O、哪一行在打擊。
  - 打擊畫面判斷（`detect/cursor.py`，好球帶框線亮度）。
  - 落點標記（`detect/markers.py`：藍 ◎＝好球/界外/擊出、紅 ⊗＝壞球；被標籤蓋住的低球用 `detect/lowball.py` 找 BALL/STRIKE 字樣模板 `detect/templates/`）。
  - 標籤文字（球種、球速）、投手/打者名、玩家 ID：PaddleOCR（`ocr/engine.py`），全部在背景執行緒跑，不可卡住主迴圈。
  - 界外／擊出：先暫記，等記分板球數確認後自動改正（`_hit_check`、`_resolve_hit`）。
  - 重播防護：鏡頭切回打擊畫面時會重顯上一球標記，不能重複記。
  - 2 對 2：每隊兩位玩家，**玩家1 投單數局、玩家2 投雙數局**（`_players_for`）。客隊＝記分板上行（先攻）。
- `store/db.py`：SQLite `data/bbo_pitches.db`。`pitches` 一球一筆；`predictions` 每球的預測與命中；`player_view`、`predict`（只分玩家：全部／球數情境／同球數加權混合＋上一球轉移）、`score_prediction`、`prediction_accuracy`。
- `capture/recorder.py`：飛行記錄器（可疑狀況自動存前後畫面到 `data/feedback/`）、整場錄影（預設關）。

## 座標
全部用 0~1 正規化座標，相對遊戲畫面（client area）；校正值在 `config/calibration.default.json`（使用者自己的在 `config/calibration.json`，不進版控）。好球帶 x 0.4516–0.5381、y 0.6935–0.8426。

## 版本與更新
- 版本號三處要一致：`version.json`、`state/tracker.py` 的 `APP_VERSION`、`web/static/index.html` 的 `VERSION`。
- 使用者端 `tools/updater.py` 從本 repo `main` 分支下載 zip，只覆蓋程式檔（不動 `data/`、`.venv/`、`config/calibration.json`）；requirements 變了會自動 pip install。網頁偵測到新版會顯示「更新並重新啟動」。
- 發新版：改程式 → 三處版本號 +1 → `version.json` 的 notes 寫一句改了什麼 → commit、push 到 main。

## 已知限制／待辦
- 投手名錯字／截斷：`_canon_pitcher` 跟資料庫裡出現過 ≥2 次的名字比對（同背號、前綴、相似度 ≥0.6），以出現較多的為準。
- 比賽環境（`detect/environment.py`，2026-11-10e）：右上角天氣框（晴／雨／陰／雪用顏色判斷；風向箭頭用「朝上箭頭」模板 `templates/wind_arrow_up.png` 旋轉比對；風速數字有變才 OCR）、右下角投手／打者卡片的天氣圓圖示。每球存進 pitches 的 weather、wind_dir、wind_angle、wind_speed、pitcher_attr、batter_attr。畫面從本壘看外野：箭頭往外野＝順風、往本壘＝逆風、左右＝側風，風速 ≥5 的順風＝大順風（網頁判斷）。投手能量條在投手頭上（ENERGY_BAR），只在投球前短暫出現，投出前 10 秒內讀到才記進 energy（0~1）。
- 名字色條（`ocr/engine.py` `_has_name_bar`）：打者、投手色條顏色依卡片不同且左右並排，要左右分開判斷；TAB 大頭像蓋住時不讀。2026-11-10f 前後攻半局常讀不到名字就是這個原因。
- 已做（2026-11-10d）：換場清空 ID、每 15 秒與攻守交換時重讀 ID，任一隊全部換人＝自動開新場；即時數據跟玩家＋總球數；九宮格「帶外比例」（x_norm/y_norm 超出 0~1）＋追打率；記分板逐球審核（好壞球判反自動改、漏記補一筆 zone="0" 不計入九宮格）；低優先權、執行緒上限、非打擊畫面降到 10 fps。
- 待做：輕量化進階（只截需要區域、OCR 標籤改 mobile 模型——改前先用錄影比對錯字率）；軌跡辨識球種／球速（先做可行性測試，用 NVIDIA 60fps 錄影；天氣、屬性相剋、投手能量條會影響軌跡，畫面上讀得到就一起記）；修正之前記錯對手 ID 的場次（要讀使用者電腦的 DB）。
- 測試方式：用 `data/recordings/*.mp4 + .jsonl` 重播，OCR 用假物件取代（`sys.modules['ocr.engine']`）；漏記檢查＝前後球數是否連續。
