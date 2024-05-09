# california_housing_regression

Python(この資料は3.9.13で作成)の仮想環境を作って有効化する。
```
python -m venv .venv
.\.venv\Scripts\activate
```
続いて、計算にGPUが使えるよう[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)をダウンロード＋インストールする。
バージョンの選択肢として、今日現在（2024-05-09）、CUDA 11.8とCUDA 12.1があるが、ここではCUDA 11.8の想定。

pytorchのインストールも行っておく。
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

最後に、必要なライブラリのインストールを行う。
```
pip install -r requirements.txt
```

