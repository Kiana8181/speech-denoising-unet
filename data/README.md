# Data

The dataset is not stored in this repository (about 3.7 GB unpacked). Download it with:

```bash
python scripts/download_data.py              # full dataset
python scripts/download_data.py --sample 300  # 300 noisy/clean pairs for a quick smoke test
```

Or download the zip manually from <https://www.kaggle.com/datasets/mahdiahmadi81/noisy-audio-mnist>
and extract it so that this folder contains:

```text
data/noisy-audio-mnist/
├── audioMNIST_meta.txt
├── 01/
│   ├── 0_01_0.wav         # clean recording
│   ├── 0_01_0_noisy.wav   # noisy version
│   └── ...
└── ... (60 speaker folders)
```
