"""Đánh giá truy xuất căn tương đồng (RAG retrieval).
Chạy từ thư mục gốc repo:
    python -m evaluation.rag_retrieval_eval build   # tạo golden set
    python -m evaluation.rag_retrieval_eval eval    # chạy đánh giá
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation.ml_eval import load

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "evaluation" / "golden" / "queries.csv"
OUT = ROOT / "evaluation" / "results" / "rag_retrieval.json"
INPUT_COLS = ["Tỉnh thành", "Quận huyện", "Diện tích", "Mặt tiền", "Đường vào",
              "Số tầng", "Số phòng ngủ", "Số toilet", "Pháp lý", "Nội thất", "Hướng nhà"]
K = 3
N_QUERIES = 100
MIN_RELEVANT = 3
SEED = 42


def get_split():
    # Cùng cách chia với ml_eval: DB = train, truy vấn lấy từ test -> không tự khớp
    return train_test_split(load(), test_size=0.2, random_state=42)


TOLERANCES = [0.05, 0.20]  # chỉ số chính là 5%; 20% chỉ để tham khảo


def relevant_mask(db, q, tol=0.05):
    """Căn tương đồng đúng: cùng tỉnh, cùng quận, diện tích lệch tối đa tol."""
    same = (db["Tỉnh thành"] == q["Tỉnh thành"]) & (db["Quận huyện"] == q["Quận huyện"])
    close = (db["Diện tích"] - q["Diện tích"]).abs() <= tol * q["Diện tích"]
    return same & close


def retrieve_similar(house, db, top_k=K):
    """Đúng logic của app_dss.py: lọc tỉnh+quận, không có thì lọc tỉnh, rồi sắp theo chênh lệch diện tích."""
    matched = db[(db["Tỉnh thành"] == house["Tỉnh thành"]) &
                 (db["Quận huyện"] == house["Quận huyện"])]
    if len(matched) == 0:
        matched = db[db["Tỉnh thành"] == house["Tỉnh thành"]]
    matched = matched.copy()
    matched["diff_area"] = (matched["Diện tích"] - house["Diện tích"]).abs()
    return matched.sort_values("diff_area").head(top_k)


def random_same_district(house, db, top_k=K, rng=None):
    """Đối chứng: lấy ngẫu nhiên top_k căn cùng quận (không sắp theo diện tích)."""
    pool = db[(db["Tỉnh thành"] == house["Tỉnh thành"]) &
              (db["Quận huyện"] == house["Quận huyện"])]
    if len(pool) == 0:
        pool = db[db["Tỉnh thành"] == house["Tỉnh thành"]]
    n = min(top_k, len(pool))
    return pool.sample(n=n, random_state=int(rng.integers(1e9)))


def build_golden():
    db, test = get_split()
    rows = []
    for _, q in test.sample(n=min(len(test), 400), random_state=SEED).iterrows():
        n_rel = int(relevant_mask(db, q).sum())
        if n_rel < MIN_RELEVANT:
            continue
        row = {"qid": f"q{len(rows):03d}", "n_relevant_in_db": n_rel}
        row.update({c: q[c] for c in INPUT_COLS})
        rows.append(row)
        if len(rows) == N_QUERIES:
            break
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(GOLDEN, index=False, encoding="utf-8-sig")
    print(f"Đã lưu {len(rows)} truy vấn vào {GOLDEN}")


def score(picked_idx, rel, k=K):
    flags = [bool(rel.loc[i]) for i in picked_idx]
    return sum(flags) / k, any(flags)


def evaluate_tol(golden, db, tol, rng):
    p_ret, h_ret, p_rnd, h_rnd, gaps_ret, gaps_rnd = [], [], [], [], [], []
    for _, q in golden.iterrows():
        house = {c: q[c] for c in INPUT_COLS}
        rel = relevant_mask(db, q, tol)
        top = retrieve_similar(house, db)
        p, h = score(top.index, rel); p_ret.append(p); h_ret.append(h)
        gaps_ret.append(float((top["Diện tích"] - q["Diện tích"]).abs().median()))
        rnd = random_same_district(house, db, rng=rng)
        p, h = score(rnd.index, rel); p_rnd.append(p); h_rnd.append(h)
        gaps_rnd.append(float((rnd["Diện tích"] - q["Diện tích"]).abs().median()))
    return {
        "retriever_app_logic": {f"precision@{K}": round(float(np.mean(p_ret)), 3),
                                f"hit@{K}": round(float(np.mean(h_ret)), 3),
                                "median_area_gap_m2": round(float(np.median(gaps_ret)), 2)},
        "baseline_random_same_district": {f"precision@{K}": round(float(np.mean(p_rnd)), 3),
                                          f"hit@{K}": round(float(np.mean(h_rnd)), 3),
                                          "median_area_gap_m2": round(float(np.median(gaps_rnd)), 2)},
    }


def evaluate():
    db, _ = get_split()
    golden = pd.read_csv(GOLDEN, encoding="utf-8-sig")
    rng = np.random.default_rng(SEED)
    fallback = 0
    for _, q in golden.iterrows():
        top = retrieve_similar({c: q[c] for c in INPUT_COLS}, db)
        same = ((top["Tỉnh thành"] == q["Tỉnh thành"]) &
                (top["Quận huyện"] == q["Quận huyện"])).all()
        fallback += int(not same)
    results = {
        "n_queries": int(len(golden)),
        "k": K,
        "fallback_rate_no_same_district": round(fallback / len(golden), 3),
        "primary_tolerance_5pct": evaluate_tol(golden, db, 0.05, rng),
        "reference_tolerance_20pct": evaluate_tol(golden, db, 0.20, rng),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    build_golden() if cmd == "build" else evaluate()
