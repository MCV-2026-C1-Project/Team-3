# Week 1 – Image Retrieval with Color Histograms

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTKPdk10p7WcYlu0ypvW_SsvzAM2ndGvm-2cWKLMsp_zQsphC4udC6Qpb0F&s=10)



**Team 3**

- Eduard Barnadas Conangla
- Paula, Gálvez Serrano
- Femiloye Oluseun Oyerinde
- Ares Sellart Sánchez

## How to run

Install the dependencies and run the scripts from the `Team-3/` directory:

```bash
pip install -r requirements.txt
python scripts/evaluate_week1.py
python scripts/generate_qst1.py
```

The datasets must be placed in `data/BBDD`, `data/qsd1_w1` and `data/qst1_w1`

## Repository structure

```
Project/
├── data/
│   ├── BBDD/                    # Database images
│   ├── qsd1_w1/                 # Development queries + gt_corresps.pkl
│   └── qst1_w1/                 # Test queries (QST1)
├── src/
│   ├── color_histogram_retrieval.py   # Histogram descriptor and retrieval class
│   ├── distances.py                   # Distance metrics
│   ├── evaluation.py                  # mAP@K and top-5 hit rate
│   ├── data.py                        # Dataset and ground-truth loading
│   └── methods.py                     # Method configurations (audit and submission)
├── scripts/
│   ├── evaluate_week1.py        # Evaluation on QSD1
│   └── generate_qst1.py         # Creation of the submission files for QST1
├── outputs/
│   └── week1/
│       ├── qsd1_results.csv    # QSD1 evaluation table
│       └── QST1/               # Generated submission files
│           ├── method1/        # result.pkl
│           └── method2/        # result.pkl
├── requirements.txt
└── README.md
```

## Scripts

- `scripts/evaluate_week1.py`: evaluates every method on the QSD1 development set (mAP@1, mAP@5 and top-5 hit rate) and saves the table to `outputs/week1/qsd1_results.csv`.
- `scripts/generate_qst1.py`: computes the 10 best database matches for each QST1 test query with `method1` and `method2`, and saves them to `submissions/Team3/week1/QST1/method{1,2}/result.pkl`. Each file is a list of lists of integer image IDs (e.g. `[[7, 2], [76, 4], ...]`, where 7 corresponds to `00007.jpg`).
