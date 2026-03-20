# `course/Teaching-Resource/ai/Neural-Network` 實作說明

這個資料夾是文章〈神經網路是什麼？從學習原理到 NN、CNN、RNN 的入門與實作〉與〈圖像識別是怎麼運作的？從像素、濾波、卷積到 CNN 的入門與兩個實作〉的可執行補充。

我把範例拆成三個腳本：

- `nn_example.py`：`MNIST` 全連接分類
- `cnn_example.py`：`MNIST` 卷積分類
- `rnn_example.py`：字元級文字序列預測
- `image_filter_demo.py`：本地圖片的平均濾波 / 權重平均濾波 / 中值濾波 / `Sobel` 邊緣偵測

## 1. 建立虛擬環境

以下以 Windows PowerShell 為主：

```powershell
cd d:\個人網站\jiangphtw.github.io\course\Teaching-Resource\ai
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. 執行範例

### `NN`：全連接分類

```powershell
python nn_example.py --epochs 1
```

### `CNN`：卷積分類

```powershell
python cnn_example.py --epochs 1
```

### `RNN`：字元級文字序列預測

```powershell
python rnn_example.py --epochs 5
```

### 影像前處理：濾波與邊緣偵測

```powershell
python image_filter_demo.py --input path\to\image.png --output-dir .\outputs
```

## 3. 參數說明

### `NN` / `CNN`

- `--epochs`：訓練輪次
- `--batch-size`：批次大小
- `--device`：運算裝置，預設 `cpu`
- `--data-dir`：資料下載目錄，預設 `./.data`

### `RNN`

- `--epochs`：訓練輪次
- `--device`：運算裝置，預設 `cpu`
- `--corpus`：文字資料路徑，預設 `rnn_corpus.txt`
- `--generate-length`：訓練後要生成多少字元

### `image_filter_demo.py`

- `--input`：輸入圖片路徑
- `--output-dir`：輸出結果資料夾

## 4. 補充說明

- `MNIST` 會自動下載到 `./.data`
- 這些範例都用 CPU 友善的預設值，不要求 GPU
- 目標是教學清楚，不是追求最佳準確率
- `image_filter_demo.py` 會把圖片先轉成灰階，再輸出 `average`、`weighted-average`、`median`、`sobel` 四種結果
- 文章使用的濾波示範圖與輸出結果放在 `demo_assets/image_filters/`
