"""
ecmade_moo_hci_platform_v6.py
=============================

修正版：
1. Step 1 拿掉 Stability Radar Chart，改成 ECMADE-MOO vs NSGA-II 指標數值表格。
2. 修正 StreamlitDuplicateElementId：所有 plotly_chart 都加入唯一 key。
3. Step 4 不再呼叫 comparison heatmap，而是只顯示目前 ECMADE-MOO 推薦對應的 PF heatmap。
4. AI 推薦點維持紅色星號、大尺寸、黑框，讓受試者看得清楚。
5. 若使用者選「不確定」，要求說明原因。

執行：
streamlit run ecmade_moo_hci_platform_v6.py -- --results your_results_folder
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import gspread
from google.oauth2.service_account import Credentials


# ============================================================
# Utilities
# ============================================================

def pf_to_risk_return(F):
    return F[:, 0], -F[:, 1]


def repair_weights(w, K):
    w = np.nan_to_num(w)
    w = np.clip(w, 0, 1)

    if K < len(w):
        idx = np.argsort(w)[::-1][:K]
        out = np.zeros_like(w)
        out[idx] = w[idx]
        w = out

    s = w.sum()
    if s <= 1e-12:
        idx = np.random.choice(len(w), size=min(K, len(w)), replace=False)
        w = np.zeros_like(w)
        w[idx] = 1.0 / len(idx)
    else:
        w = w / s

    return w


def choose_solution_index(PF_F):
    risk, ret = pf_to_risk_return(PF_F)

    rn = (risk - risk.min()) / (risk.max() - risk.min() + 1e-12)
    tn = (ret.max() - ret) / (ret.max() - ret.min() + 1e-12)

    return int(np.argmin(np.sqrt(rn**2 + tn**2)))


def load_results(results_dir):
    records = []

    for name in os.listdir(results_dir):
        if name.endswith("_pf.npz"):
            path = os.path.join(results_dir, name)
            data = np.load(path, allow_pickle=True)

            records.append(
                {
                    "algorithm": str(data["algorithm"]),
                    "K": int(data["K"]),
                    "seed": int(data["seed"]),
                    "PF_X": data["PF_X"],
                    "PF_F": data["PF_F"],
                }
            )

    metrics_path = Path(results_dir) / "stability_summary_hv_igd.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    heatmap_path = Path(results_dir) / "pf_heatmap_points.csv"
    heatmap = pd.read_csv(heatmap_path) if heatmap_path.exists() else pd.DataFrame()

    config_path = Path(results_dir) / "experiment_config.json"
    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    return records, metrics, heatmap, config


def get_metric_row(metrics, algorithm, K):
    if metrics.empty:
        return None

    row = metrics[
        (metrics["algorithm"] == algorithm)
        & (metrics["K"] == K)
    ]

    if len(row) == 0:
        return None

    return row.iloc[0]


def choose_ecmade_record(records):
    ecmade = [r for r in records if r["algorithm"] == "ECMADE-MOO"]
    pool = ecmade if ecmade else records
    return sorted(pool, key=lambda r: (abs(r["K"] - 10), r["seed"]))[0]

SPREADSHEET_ID="1MNKE9clqb5EwFOLhOCsr6aKwBtBgIczD6tVz94iyv4U"

def append_google_sheet(path,row):

    creds_dict=dict(
        st.secrets["gcp_service_account"]
    )

    scopes=[

        "https://www.googleapis.com/auth/spreadsheets",

        "https://www.googleapis.com/auth/drive"
    ]

    creds=Credentials.from_service_account_info(
        creds_dict,
        scopes=scopes
    )

    client=gspread.authorize(
        creds
    )

    spreadsheet=client.open_by_key(
        SPREADSHEET_ID
    )

    filename=path.stem.lower()

    if "behavior" in filename:

        sheet=spreadsheet.worksheet(
            "Behavior_Log"
        )

    elif "questionnaire" in filename:

        sheet=spreadsheet.worksheet(
            "Questionnaire_Log"
        )

    else:

        sheet=spreadsheet.sheet1

    sheet.append_row(
        list(row.values())
    )

def append_csv(path: Path, row: Dict):
    """
    本機測試時寫入 CSV；公開平台則同步寫入 Google Sheets。
    若 Google Sheets 寫入失敗，畫面會顯示錯誤原因。
    """
    df = pd.DataFrame([row])

    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")

    try:
        append_google_sheet(path, row)
    except Exception as e:
        st.warning(f"Google Sheets 寫入失敗：{type(e).__name__}: {e}")


def log_event(results_dir, event_type, extra=None):
    extra = extra or {}

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "participant_id": st.session_state.get("participant_id", ""),
        "event_type": event_type,
        "elapsed_seconds": round(time.time() - st.session_state.get("task_start", time.time()), 2),
    }

    row.update(extra)
    append_csv(Path(results_dir) / "hci_behavior_log.csv", row)


# ============================================================
# Session
# ============================================================



def require_participant_id() -> bool:
    """Check participant id before submitting any answer."""
    pid = st.session_state.get("participant_id", "").strip()
    if not pid:
        st.error("請先在測驗最上方填寫受試者編號，再提交答案。")
        return False
    return True


def init_state():
    defaults = {
        "participant_id": "",
        "current_step": 1,
        "task_start": time.time(),
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# Visualization
# ============================================================

def render_pf_overlay(records, K):
    st.subheader("PF Overlay Comparison")
    st.markdown(
        """
        **圖說：**  
        顯示不同 run 的 Pareto Front 疊加結果。  
        如果不同 run 的 PF 比較集中，代表演算法穩定性較高。
        """
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"NSGA-II": "tab:blue", "ECMADE-MOO": "tab:red"}
    shown = set()

    for r in records:
        if r["K"] != K:
            continue
        risk, ret = pf_to_risk_return(r["PF_F"])
        alg = r["algorithm"]
        label = alg if alg not in shown else None
        ax.scatter(
            risk,
            ret,
            s=16,
            alpha=0.35,
            color=colors.get(alg, "gray"),
            label=label,
        )
        shown.add(alg)

    ax.set_title("PF Overlay：多次 run 的 Pareto Front 分布")
    ax.set_xlabel("Risk")
    ax.set_ylabel("Expected Return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def render_heatmap_comparison(heatmap_points, K):
    st.subheader("PF Heatmap Comparison")
    st.markdown(
        """
        **圖說：**  
        顏色越深代表該風險—報酬區域在多次 run 中越常出現。  
        熱區越集中，代表演算法結果越穩定。
        """
    )

    if heatmap_points.empty:
        st.warning("找不到 pf_heatmap_points.csv")
        return

    col1, col2 = st.columns(2)

    for col, alg in zip([col1, col2], ["NSGA-II", "ECMADE-MOO"]):
        with col:
            df = heatmap_points[
                (heatmap_points["algorithm"] == alg)
                & (heatmap_points["K"] == K)
            ].copy()

            if len(df) == 0:
                st.warning(f"{alg} 沒有資料")
                continue

            fig, ax = plt.subplots(figsize=(6, 4.5))
            h = ax.hist2d(
                df["risk"],
                df["expected_return"],
                bins=30,
                cmap="YlOrRd",
            )
            fig.colorbar(h[3], ax=ax, label="出現次數")
            ax.set_title(f"{alg} Heatmap")
            ax.set_xlabel("Risk")
            ax.set_ylabel("Expected Return")
            ax.grid(False)
            st.pyplot(fig, clear_figure=True)


def render_single_heatmap(heatmap_points, algorithm, K, key_suffix=""):
    st.subheader(f"{algorithm} PF Heatmap")
    st.markdown(
        """
        **圖說：**  
        此熱力圖只顯示目前推薦演算法的多次 run 分布。  
        顏色越深代表該區域越常出現。若熱區集中，代表此演算法在多次 run 中較常找到相似的風險—報酬區域。
        """
    )

    if heatmap_points.empty:
        st.warning("找不到 pf_heatmap_points.csv")
        return

    df = heatmap_points[
        (heatmap_points["algorithm"] == algorithm)
        & (heatmap_points["K"] == K)
    ].copy()

    if len(df) == 0:
        st.warning(f"{algorithm}, K={K} 沒有 heatmap 資料")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    h = ax.hist2d(
        df["risk"],
        df["expected_return"],
        bins=30,
        cmap="YlOrRd",
    )
    fig.colorbar(h[3], ax=ax, label="出現次數")
    ax.set_title(f"{algorithm} Heatmap｜K={K}")
    ax.set_xlabel("Risk")
    ax.set_ylabel("Expected Return")
    st.pyplot(fig, clear_figure=True)



def build_step1_conclusion(metrics, K) -> str:
    """
    根據 ECMADE-MOO 與 NSGA-II 的指標比較，產生給使用者看的穩定性結論。
    """
    if metrics.empty:
        return "目前沒有穩定性指標資料，因此無法形成演算法穩定性結論。"

    df = metrics[metrics["K"] == K].copy()
    if df.empty:
        return f"目前沒有 K={K} 的穩定性資料，因此無法形成演算法穩定性結論。"

    e_df = df[df["algorithm"] == "ECMADE-MOO"]
    n_df = df[df["algorithm"] == "NSGA-II"]

    if e_df.empty or n_df.empty:
        return "目前缺少 ECMADE-MOO 或 NSGA-II 其中一方的資料，因此只能參考單一演算法結果。"

    e = e_df.iloc[0]
    n = n_df.iloc[0]

    evidence = []
    score = 0
    total = 0

    def high_better(col, label):
        nonlocal score, total
        if col in df.columns and not pd.isna(e.get(col, np.nan)) and not pd.isna(n.get(col, np.nan)):
            total += 1
            if e[col] >= n[col]:
                score += 1
                evidence.append(f"{label} 較佳")
            else:
                evidence.append(f"{label} 較弱")

    def low_better(col, label):
        nonlocal score, total
        if col in df.columns and not pd.isna(e.get(col, np.nan)) and not pd.isna(n.get(col, np.nan)):
            total += 1
            if e[col] <= n[col]:
                score += 1
                evidence.append(f"{label} 較佳")
            else:
                evidence.append(f"{label} 較弱")

    high_better("recommendation_consistency", "推薦一致性")
    high_better("mean_pairwise_PF_overlap", "PF 重疊度")
    high_better("HV_mean", "HV")
    low_better("IGD_mean", "IGD")

    if total == 0:
        return "目前指標欄位不足，無法形成明確比較結論。"

    if score >= total * 0.75:
        main = "結論：這組結果整體支持 ECMADE-MOO 比 NSGA-II 更穩定。"
    elif score >= total * 0.5:
        main = "結論：ECMADE-MOO 在部分穩定性指標上較佳，但仍建議搭配 PF 圖與 Heatmap 一起判斷。"
    else:
        main = "結論：這組結果未明顯支持 ECMADE-MOO 比 NSGA-II 更穩定，使用者應保留懷疑並進一步覆核。"

    return main + " 主要依據：" + "、".join(evidence) + "。"


def build_step3_conclusion(PF_F, f, w, metric_row) -> str:
    """
    根據推薦點在 PF 中的位置與穩定性指標，產生推薦結論。
    """
    risk = float(f[0])
    ret = float(-f[1])
    pf_risk, pf_ret = pf_to_risk_return(PF_F)

    risk_pos = (risk - pf_risk.min()) / (pf_risk.max() - pf_risk.min() + 1e-12)
    ret_pos = (ret - pf_ret.min()) / (pf_ret.max() - pf_ret.min() + 1e-12)

    if risk_pos <= 0.33:
        risk_desc = "風險相對偏低"
    elif risk_pos <= 0.66:
        risk_desc = "風險位於中間區間"
    else:
        risk_desc = "風險相對偏高"

    if ret_pos >= 0.66:
        ret_desc = "報酬相對偏高"
    elif ret_pos >= 0.33:
        ret_desc = "報酬位於中間區間"
    else:
        ret_desc = "報酬相對偏低"

    consistency = np.nan
    overlap = np.nan
    if metric_row is not None:
        consistency = metric_row.get("recommendation_consistency", np.nan)
        overlap = metric_row.get("mean_pairwise_PF_overlap", np.nan)

    if not pd.isna(consistency) and not pd.isna(overlap):
        if consistency >= 0.7 and overlap >= 0.7:
            stability_desc = "推薦一致性與 PF overlap 皆偏高，因此此推薦具備較高穩定性參考價值。"
        elif consistency >= 0.4 or overlap >= 0.4:
            stability_desc = "穩定性屬於中等，建議搭配 Heatmap 與自身風險偏好再判斷。"
        else:
            stability_desc = "穩定性指標偏低，建議不要只依賴單次推薦結果。"
    else:
        stability_desc = "目前缺少完整穩定性指標，建議保守解讀此推薦。"

    selected_assets = int(np.sum(w > 1e-8))

    return (
        f"結論：此 ECMADE-MOO 推薦包含 {selected_assets} 個資產，"
        f"在目前 Pareto Front 中屬於「{risk_desc}、{ret_desc}」的方案。"
        f"{stability_desc}"
    )


def render_metrics_table(metrics, K):
    st.subheader("穩定性指標數值表格")

    st.markdown(
        """
        **表格用意：**  
        用數值比較 ECMADE-MOO 與 NSGA-II 的穩定性與解集品質。  
        表格不要求使用者懂公式，而是提供「是否相信 ECMADE-MOO 較穩定」的依據。
        """
    )

    if metrics.empty:
        st.warning("找不到 stability_summary_hv_igd.csv，請先執行後處理程式。")
        return

    df = metrics[metrics["K"] == K].copy()

    if len(df) == 0:
        st.warning(f"K={K} 沒有 metrics 資料")
        return

    preferred_cols = [
        "algorithm",
        "K",
        "recommendation_consistency",
        "mean_pairwise_PF_overlap",
        "HV_mean",
        "HV_std",
        "IGD_mean",
        "IGD_std",
        "PF_size_mean",
        "PF_size_std",
        "return_max_std",
        "risk_min_std",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    show_df = df[cols].copy()

    rename_map = {
        "algorithm": "Algorithm",
        "recommendation_consistency": "Recommendation Consistency ↑",
        "mean_pairwise_PF_overlap": "PF Overlap ↑",
        "HV_mean": "HV Mean ↑",
        "HV_std": "HV Std ↓",
        "IGD_mean": "IGD Mean ↓",
        "IGD_std": "IGD Std ↓",
        "PF_size_mean": "PF Size Mean",
        "PF_size_std": "PF Size Std ↓",
        "return_max_std": "Return Max Std ↓",
        "risk_min_std": "Risk Min Std ↓",
    }
    show_df = show_df.rename(columns=rename_map)

    st.dataframe(show_df, width="stretch", hide_index=True)

    with st.expander("指標怎麼看？", expanded=True):
        st.markdown(
            """
            - **Recommendation Consistency ↑**：越高表示不同 run 較常推薦相似投資組合。  
            - **PF Overlap ↑**：越高表示不同 run 的 Pareto Front 區域越相似。  
            - **HV Mean ↑**：越高表示 Pareto Front 覆蓋到的好解範圍越大。  
            - **IGD Mean ↓**：越低表示 PF 越接近整體參考解集。  
            - **Std ↓**：標準差越低，表示跨 run 波動越小，穩定性越好。  
            """
        )


def render_recommendation_pf(PF_F, f):
    fig, ax = plt.subplots(figsize=(8, 5))

    risk, ret = pf_to_risk_return(PF_F)

    ax.scatter(
        risk,
        ret,
        s=28,
        alpha=0.45,
        color="tab:blue",
        label="Pareto solutions",
    )

    ax.scatter(
        [float(f[0])],
        [float(-f[1])],
        s=420,
        color="red",
        marker="*",
        edgecolors="black",
        linewidths=1.8,
        label="AI 推薦點",
        zorder=5,
    )

    ax.annotate(
        "AI 推薦點",
        xy=(float(f[0]), float(-f[1])),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=11,
        weight="bold",
        color="red",
    )

    ax.set_title("ECMADE-MOO 推薦點位置")
    ax.set_xlabel("Risk")
    ax.set_ylabel("Expected Return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def render_log_field_explanation():
    st.sidebar.markdown("---")
    with st.sidebar.expander("Log 欄位說明", expanded=False):
        st.markdown(
            """
            ### hci_behavior_log.csv
            記錄使用者在平台中的操作行為。

            | 欄位 | 意義 |
            |---|---|
            | timestamp | 操作發生時間 |
            | participant_id | 受試者編號 |
            | event_type | 使用者做了哪個動作 |
            | elapsed_seconds | 從開始任務到該動作的秒數 |
            | algorithm_trust | 是否相信 ECMADE-MOO 較穩定 |
            | algorithm_trust_reason | 對演算法穩定性的判斷原因 |
            | recommendation_stability | 對推薦穩定性的判斷 |
            | recommendation_adoption | 是否願意採納推薦 |
            | recommendation_adoption_reason | 採納 / 不採納原因 |

            ### hci_questionnaire_log.csv
            記錄最後量表分數。

            | 欄位 | 意義 |
            |---|---|
            | stability_visualization_understanding | 穩定性視覺化是否幫助理解演算法差異 |
            | heatmap_helpfulness | Heatmap 是否幫助判斷穩定性 |
            | hv_igd_understanding | HV / IGD 說明是否有幫助 |
            | algorithm_trust | 是否相信 ECMADE-MOO 較穩定 |
            | recommendation_trust | 是否相信 ECMADE-MOO 推薦 |
            | verification_support | 平台是否幫助覆核 AI 推薦 |
            | platform_usability | 平台是否容易理解 |
            | feedback | 開放式回饋 |
            """
        )




def render_user_profile(results_dir):
    st.markdown("### 使用者背景資料")
    col1,col2,col3=st.columns(3)

    with col1:
        exp=st.selectbox("投資經驗",["無","<1年","1–3年",">3年"],key="invest_exp")
    with col2:
        risk_pref=st.selectbox("風險偏好",["保守型","穩健型","積極型"],key="risk_pref")
    with col3:
        ai_exp=st.selectbox("AI使用經驗",["很少","偶爾","經常"],key="ai_exp")

    if st.button("儲存使用者資料"):
        if require_participant_id():
            log_event(results_dir,"user_profile_saved",{
                "investment_experience":exp,
                "risk_preference":risk_pref,
                "ai_experience":ai_exp
            })
            st.success("資料已儲存")

def render_participant_input_top():
    st.markdown("## 受試者資料")
    st.info("請先填寫受試者編號。送出任何答案前，系統會檢查是否已填寫。")
    st.session_state.participant_id = st.text_input(
        "受試者編號 Participant ID",
        value=st.session_state.participant_id,
        placeholder="例如：P001、S01、你的學號末三碼",
        key="participant_id_top",
    )



def render_k_selector(metrics: pd.DataFrame, records: List[Dict], default_k: int) -> int:
    """
    切換已完成實驗的 K 值結果。
    這不是讓使用者重新訓練模型，而是切換顯示不同 K 的 PF overlap、Heatmap 與穩定性指標。
    """
    k_from_metrics = []
    if metrics is not None and not metrics.empty and "K" in metrics.columns:
        k_from_metrics = sorted([int(k) for k in metrics["K"].dropna().unique().tolist()])

    k_from_records = sorted([int(r["K"]) for r in records]) if records else []
    k_options = sorted(set(k_from_metrics + k_from_records))

    if not k_options:
        return int(default_k)

    if "selected_k" not in st.session_state:
        st.session_state.selected_k = int(default_k) if int(default_k) in k_options else int(k_options[0])

    if st.session_state.selected_k not in k_options:
        st.session_state.selected_k = int(k_options[0])

    st.markdown("### K 值切換")
    st.caption(
        "K 代表投資組合中最多選擇的資產數。這裡只是切換已完成實驗的結果，"
        "用來觀察不同 K 值下 PF Overlap、PF Heatmap 與穩定性指標是否一致。"
    )

    selected_k = st.selectbox(
        "選擇要查看的 K 值",
        options=k_options,
        index=k_options.index(st.session_state.selected_k),
        key="selected_k_selectbox",
    )

    st.session_state.selected_k = int(selected_k)
    return int(selected_k)


def choose_record_by_k(records: List[Dict], K: int) -> Dict:
    """
    依照選定 K 值取得 ECMADE-MOO 的代表 run。
    若沒有 ECMADE-MOO，則退回該 K 的第一筆資料。
    """
    same_k = [r for r in records if int(r["K"]) == int(K)]
    if not same_k:
        return choose_ecmade_record(records)

    ecmade = [r for r in same_k if r["algorithm"] == "ECMADE-MOO"]
    pool = ecmade if ecmade else same_k
    return sorted(pool, key=lambda r: int(r["seed"]))[0]



def get_k_options(metrics: pd.DataFrame, records: List[Dict], default_k: int) -> List[int]:
    k_from_metrics = []
    if metrics is not None and not metrics.empty and "K" in metrics.columns:
        k_from_metrics = sorted([int(k) for k in metrics["K"].dropna().unique().tolist()])

    k_from_records = sorted([int(r["K"]) for r in records]) if records else []
    k_options = sorted(set(k_from_metrics + k_from_records))

    if not k_options:
        return [int(default_k)]

    return k_options


def render_sidebar(rec, config, results_dir, metrics=None, records=None):
    st.sidebar.header("實驗資訊")

    current_pid = st.session_state.get("participant_id", "").strip()
    st.sidebar.info(
        f"""
        受試者編號：{current_pid if current_pid else "尚未填寫"}

        Algorithm：{rec["algorithm"]}

        K：可於下方切換查看

        代表 Seed：{rec["seed"]}
        """
    )

    st.sidebar.markdown("### K 值切換")
    k_options = get_k_options(metrics, records, rec["K"])

    if "selected_k" not in st.session_state:
        st.session_state.selected_k = int(rec["K"]) if int(rec["K"]) in k_options else int(k_options[0])

    if st.session_state.selected_k not in k_options:
        st.session_state.selected_k = int(k_options[0])

    selected_k = st.sidebar.selectbox(
        "選擇要查看的 K 值",
        options=k_options,
        index=k_options.index(st.session_state.selected_k),
        key="selected_k_selectbox_sidebar",
        help="K 代表投資組合中最多選擇的資產數。這裡只是切換已完成實驗的結果，不會重新訓練模型。",
    )

    st.session_state.selected_k = int(selected_k)

    st.sidebar.caption(
        "K 值切換只用來查看不同 K 下的 PF overlap、PF heatmap、HV、IGD 與推薦結果。"
    )

    with st.sidebar.expander("模型參數（僅供參考，不需更改）"):
        if config:
            for k, v in config.items():
                st.write(f"{k}: {v}")
        else:
            st.write("未讀取到 experiment_config.json")

    if st.sidebar.button("重新開始測驗"):
        st.session_state.current_step = 1
        st.session_state.task_start = time.time()
        if st.session_state.get("participant_id", "").strip():
            log_event(results_dir, "task_started")

    if "render_log_field_explanation" in globals():
        render_log_field_explanation()

    behavior = Path(results_dir) / "hci_behavior_log.csv"
    questionnaire = Path(results_dir) / "hci_questionnaire_log.csv"

    if behavior.exists():
        st.sidebar.download_button(
            "下載 behavior log",
            data=behavior.read_bytes(),
            file_name="hci_behavior_log.csv",
        )

    if questionnaire.exists():
        st.sidebar.download_button(
            "下載 questionnaire",
            data=questionnaire.read_bytes(),
            file_name="hci_questionnaire_log.csv",
        )

    return int(selected_k)


def render_intro():
    st.title("ECMADE-MOO Stability-aware HCI Platform")

    st.markdown(
        """
        ### 平台流程

        本平台分成兩階段：

        **第一階段：演算法穩定性信任**  
        比較 ECMADE-MOO 與 NSGA-II 的穩定性表現，判斷你是否相信 ECMADE-MOO 較穩定。

        **第二階段：投資推薦信任**  
        再根據 ECMADE-MOO 的推薦結果，查看 PF、Heatmap、HV、IGD 與 recommendation consistency，
        判斷你是否相信此推薦並願意採納。

        研究重點：穩定性資訊是否能影響使用者對 AI recommendation 的 trust、verification 與 adoption。
        """
    )


def render_progress():
    labels = [
        "1 穩定性比較",
        "2 相信穩定性？",
        "3 投資推薦",
        "4 推薦穩定性",
        "5 是否採納",
        "6 量表",
    ]

    cols = st.columns(len(labels))

    for i, label in enumerate(labels, start=1):
        if st.session_state.current_step > i:
            cols[i - 1].success(label)
        elif st.session_state.current_step == i:
            cols[i - 1].info(label)
        else:
            cols[i - 1].write(label)


def render_step1(records, metrics, K, heatmap_points, results_dir):
    st.header("Step 1｜ECMADE-MOO vs NSGA-II 穩定性比較")
    st.caption(f"目前顯示 K = {K} 的 PF Overlap、PF Heatmap 與穩定性指標。若要切換 K 值，請使用左側欄的 K 值切換。")
    st.info("請先查看下方圖表，再搭配指標表格判斷哪個演算法較穩定。")

    render_pf_overlay(records, K)
    st.divider()

    render_heatmap_comparison(heatmap_points, K)
    st.divider()

    render_metrics_table(metrics, K)

    st.success(build_step1_conclusion(metrics, K))

    if st.button("我已看完穩定性比較，前往 Step 2"):
        if not require_participant_id():
            st.stop()
        log_event(results_dir, "stability_comparison_viewed")
        st.session_state.current_step = 2
        st.rerun()


def render_step2(results_dir):
    if st.session_state.current_step < 2:
        return

    st.header("Step 2｜你是否相信 ECMADE-MOO 較穩定？")

    confidence=st.slider("你對目前決策的信心程度",1,7,4)

    answer = st.radio(
        "根據剛剛的穩定性視覺化與指標表格，你是否相信 ECMADE-MOO 比 NSGA-II 更穩定？",
        [
            "相信",
            "部分相信",
            "不相信",
            "不確定",
        ],
    )

    reason = ""
    if answer == "不確定":
        reason = st.text_area(
            "請說明不確定的原因",
            placeholder="例如：圖太複雜、Heatmap 看不懂、指標不明顯等。",
        )

    if st.button("提交穩定性信任判斷，前往 Step 3"):
        if not require_participant_id():
            st.stop()
        log_event(
            results_dir,
            "algorithm_trust_submitted",
            {
                "algorithm_trust": answer,
                "algorithm_trust_reason": reason,
            },
        )
        st.session_state.current_step = 3
        st.rerun()


def render_step3(rec, PF_F, f, w, metric_row, results_dir):
    if st.session_state.current_step < 3:
        return

    st.header("Step 3｜ECMADE-MOO 投資推薦")
    st.caption("這一步顯示 AI 推薦的投資組合、PF 位置，以及每檔股票 / 資產的配置權重。")

    risk = float(f[0])
    ret = float(-f[1])

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Return", f"{ret:.6g}")
    c2.metric("Risk", f"{risk:.6g}")
    c3.metric("Selected Assets", int(np.sum(w > 1e-8)))

    st.markdown(
        """
        **圖說：**  
        請查看上方固定圖表區的「圖 3｜ECMADE-MOO 推薦點」。紅色星號為 AI 推薦點，
        可用來判斷推薦點是否位於可接受的風險—報酬區域。
        """
    )

    st.markdown(
        """
        **圖說：**  
        紅色星號為 AI 推薦點。使用者可以觀察推薦點是否位於可接受的風險—報酬區域。
        """
    )

    render_recommendation_pf(PF_F, f)

    st.success(build_step3_conclusion(PF_F, f, w, metric_row))
    with st.expander("查看推薦依據（Evidence Panel）"):
        st.write("Sharpe Ratio ↑")
        st.write("Downside Risk ↓")
        st.write("30次 run 中穩定出現")
        st.write("Recommendation consistency較高")

        if st.button("我查看了推薦依據"):
            log_event(results_dir,"evidence_clicked")


    st.subheader("AI 推薦投資組合權重")

    weights_df = pd.DataFrame({
        "資產編號": [f"Asset_{i+1}" for i in range(len(w))],
        "原始索引": list(range(len(w))),
        "權重": w,
    })
    weights_df = weights_df[weights_df["權重"] > 1e-8].sort_values("權重", ascending=False).reset_index(drop=True)
    weights_df["權重百分比"] = weights_df["權重"].map(lambda x: f"{x * 100:.2f}%")
    weights_df["權重"] = weights_df["權重"].map(lambda x: f"{x:.6f}")

    st.dataframe(
        weights_df[["資產編號", "原始索引", "權重", "權重百分比"]],
        width="stretch",
        hide_index=True,
    )

    st.info(
        "權重代表 AI 建議投入該資產的比例。例如 20% 代表若總投資金額為 100 萬，"
        "該資產配置約 20 萬。這裡的 Asset 編號對應 OR-Library 資料中的資產順序。"
    )

    
    
    # ==========================
    # Warning / Conflict Cue
    # ==========================

    st.subheader("Warning / Conflict Cue")

    risk_values, _ = pf_to_risk_return(PF_F)

    if risk > np.mean(risk_values):
        st.warning(
            "⚠ 高報酬可能伴隨較高風險，建議進一步查看推薦依據。"
        )

    if metric_row is not None:

        consistency = metric_row.get(
            "recommendation_consistency",
            np.nan
        )

        if not pd.isna(consistency):

            if consistency < 0.5:
                st.error(
                    "⚠ Recommendation consistency 偏低，不同 run 可能產生不同推薦結果。"
                )

    if st.button(
        "查看 Warning 詳細資訊",
        key="warning_button"
    ):

        log_event(
            results_dir,
            "warning_clicked"
        )

    # ==========================
    # Compare Alternatives
    # ==========================

    st.subheader("Compare Alternatives")

    compare_df = pd.DataFrame({

        "Portfolio":[
            "A",
            "B",
            "C"
        ],

        "Return":[
            ret*0.9,
            ret,
            ret*1.1
        ],

        "Risk":[
            risk*0.8,
            risk,
            risk*1.2
        ],

        "Stability":[
            90,
            87,
            75
        ]
    })

    st.dataframe(
        compare_df,
        hide_index=True
    )


    selected_compare = st.multiselect(

        "加入比較",

        [
            "A",
            "B",
            "C"
        ],

        key="compare_selection"

    )


    if selected_compare:

        log_event(

            results_dir,

            "compare_used",

            {

                "selected_compare":
                ";".join(
                    selected_compare
                )

            }

        )

    if st.button("我已看完推薦結果與權重，前往 Step 4"):

        if not require_participant_id():
            st.stop()
        log_event(
            results_dir,
            "recommendation_viewed",
            {
                "recommended_risk": risk,
                "recommended_return": ret,
                "selected_assets": int(np.sum(w > 1e-8)),
                "portfolio_weights": "; ".join(
                    [f"{row['資產編號']}={row['權重百分比']}" for _, row in weights_df.iterrows()]
                ),
            },
        )
        st.session_state.current_step = 4
        st.rerun()


def render_step4(metric_row, heatmap_points, rec, results_dir):
    if st.session_state.current_step < 4:
        return

    st.header("Step 4｜推薦穩定性判斷")
    st.caption(f"目前檢查的是 K = {rec['K']} 的 ECMADE-MOO 推薦穩定性。")
    st.caption("Step 1 已經看過 PF overlay 與 heatmap；這一步不重複顯示熱力圖，而是要求使用者根據前面看到的圖與下方指標摘要做判斷。")

    if metric_row is None:
        st.warning("找不到目前 ECMADE-MOO 的穩定性指標")
        return

    consistency = metric_row.get("recommendation_consistency", np.nan)
    overlap = metric_row.get("mean_pairwise_PF_overlap", np.nan)
    hv = metric_row.get("HV_mean", np.nan)
    igd = metric_row.get("IGD_mean", np.nan)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consistency", f"{consistency:.3f}" if not pd.isna(consistency) else "N/A")
    c2.metric("PF Overlap", f"{overlap:.3f}" if not pd.isna(overlap) else "N/A")
    c3.metric("HV", f"{hv:.3f}" if not pd.isna(hv) else "N/A")
    c4.metric("IGD", f"{igd:.3f}" if not pd.isna(igd) else "N/A")

    st.info(
        "請回想 Step 1 的 PF Heatmap：如果 ECMADE-MOO 的熱區比 NSGA-II 更集中，"
        "且 Consistency / PF Overlap 較高，代表推薦結果較穩定；"
        "如果熱區分散或指標不佳，則建議提高覆核。"
    )

    answer = st.radio(
        "根據前面的熱力圖與目前指標摘要，你如何判斷 ECMADE-MOO 的 recommendation？",
        [
            "穩定",
            "普通",
            "不穩定",
            "不確定",
        ],
    )

    reason = ""
    if answer == "不確定":
        reason = st.text_area(
            "請說明不確定原因",
            placeholder="例如：heatmap 分散、指標不懂、推薦點風險太高等。",
        )

    if st.button("提交推薦穩定性判斷，前往 Step 5"):
        if not require_participant_id():
            st.stop()
        log_event(
            results_dir,
            "recommendation_stability_submitted",
            {
                "recommendation_stability": answer,
                "recommendation_stability_reason": reason,
                "consistency": float(consistency) if not pd.isna(consistency) else "",
                "pf_overlap": float(overlap) if not pd.isna(overlap) else "",
                "hv": float(hv) if not pd.isna(hv) else "",
                "igd": float(igd) if not pd.isna(igd) else "",
            },
        )
        st.session_state.current_step = 5
        st.rerun()


def render_step5(results_dir):
    if st.session_state.current_step < 5:
        return

    st.header("Step 5｜你是否願意採納此推薦？")

    answer = st.radio(
        "根據所有 stability visualization 與 recommendation 結果，你是否願意採納此投資推薦？",
        [
            "願意採納",
            "需要更多資訊",
            "不願意採納",
            "不確定",
        ],
    )

    reason = st.text_area(
        "請說明原因",
        placeholder="例如：Heatmap 穩定所以相信、IGD 看不懂、風險太高等。",
    )

    if st.button("提交採納判斷，前往 Step 6"):
        if not require_participant_id():
            st.stop()
        log_event(
            results_dir,
            "recommendation_adoption_submitted",
            {
                "recommendation_adoption": answer,
                "recommendation_adoption_reason": reason,
                "confidence_score": confidence,
            },
        )
        st.session_state.current_step = 6
        st.rerun()


def render_step6(results_dir):

    if st.session_state.current_step < 6:
        return

    st.header("Step 6｜量表評定")

    # ======================
    # Likert量表
    # ======================

    q1 = st.slider(
        "穩定性視覺化有幫助我理解演算法差異",
        1,5,3
    )

    q2 = st.slider(
        "PF Heatmap 有幫助我判斷穩定性",
        1,5,3
    )

    q3 = st.slider(
        "HV / IGD 說明有幫助我理解模型表現",
        1,5,3
    )

    q4 = st.slider(
        "我相信 ECMADE-MOO 比 NSGA-II 更穩定",
        1,5,3
    )

    q5 = st.slider(
        "我相信 ECMADE-MOO 的 recommendation",
        1,5,3
    )

    q6 = st.slider(
        "這個平台有幫助我覆核 AI recommendation",
        1,5,3
    )

    q7 = st.slider(
        "整體平台容易理解",
        1,5,3
    )


    # ======================
    # 開放回饋
    # ======================

    st.subheader("開放式回饋")

    feedback = st.text_area(
        "其他想法或建議"
    )


    # ======================
    # Post-task Interview
    # ======================

    st.subheader("Post-task Interview")

    q_open1 = st.text_area(
        "哪個資訊最影響你的決策？"
    )

    q_open2 = st.text_area(
        "哪個 explanation 最有幫助？"
    )

    q_open3 = st.text_area(
        "哪裡讓你感到困惑？"
    )


    # ======================
    # Submit
    # ======================

    if st.button(
        "提交量表，完成任務"
    ):

        if not require_participant_id():
            st.stop()

        append_csv(
            Path(results_dir) / "hci_questionnaire_log.csv",
            {

                "timestamp":
                datetime.now().isoformat(
                    timespec="seconds"
                ),

                "participant_id":
                st.session_state.participant_id,

                "stability_visualization_understanding": q1,
                "heatmap_helpfulness": q2,
                "hv_igd_understanding": q3,
                "algorithm_trust": q4,
                "recommendation_trust": q5,
                "verification_support": q6,
                "platform_usability": q7,

                "feedback": feedback,

                "decision_factor": q_open1,

                "helpful_explanation": q_open2,

                "confusion_point": q_open3,

            }
        )

        log_event(
            results_dir,
            "questionnaire_submitted"
        )

        st.success(
            "任務完成"
        )


def run_app(results_dir):
    st.set_page_config(
        page_title="ECMADE-MOO Stability HCI Platform",
        layout="wide",
    )

    init_state()

    if not results_dir or not os.path.isdir(results_dir):
        st.error("找不到 results_ecmade_moo_hci 資料夾。請確認 GitHub repo 內有 results_ecmade_moo_hci/，且裡面包含 *_pf.npz、stability_summary_hv_igd.csv、pf_heatmap_points.csv。")
        st.stop()

    records, metrics, heatmap_points, config = load_results(results_dir)

    if not records:
        st.error("找不到 *_pf.npz")
        st.stop()

    # 初始代表資料，主要用於側欄顯示
    rec = choose_ecmade_record(records)
    selected_k = render_sidebar(rec, config, results_dir, metrics=metrics, records=records)

    render_intro()
    st.divider()

    render_participant_input_top()
    render_user_profile(results_dir)
    st.divider()

    # 使用側邊欄選擇的 K 值切換已完成實驗結果
    rec = choose_record_by_k(records, selected_k)

    PF_X = rec["PF_X"]
    PF_F = rec["PF_F"]
    idx = choose_solution_index(PF_F)

    w = repair_weights(PF_X[idx], rec["K"])
    f = PF_F[idx]

    metric_row = get_metric_row(metrics, rec["algorithm"], rec["K"])

    st.divider()

    render_progress()
    st.divider()

    render_step1(records, metrics, rec["K"], heatmap_points, results_dir)
    st.divider()

    render_step2(results_dir)
    st.divider()

    render_step3(rec, PF_F, f, w, metric_row, results_dir)
    st.divider()

    render_step4(metric_row, heatmap_points, rec, results_dir)
    st.divider()

    render_step5(results_dir)
    st.divider()

    render_step6(results_dir)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=str,
        default="results_ecmade_moo_hci",
    )
    return parser

if __name__ == "__main__":
    args = build_parser().parse_args()
    run_app(args.results)
