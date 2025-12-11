import pandas as pd
from pathlib import Path

from src.user_model import MODERATE
from src.hybrid_engine import compute_final_scores, rank_actions
from src.ollama_client import ask_ollama
from src.config import DATA_DIR


def main():
    df_path = Path(DATA_DIR) / "processed" / "market_features.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"market_features.csv not found at {df_path}. "
                                f"Run src.market_data.build_market_feature_table() first.")

    df = pd.read_csv(df_path)
    row = df.iloc[-1].to_dict()  # last row as example

    # placeholder ML & pattern signals (you will replace with real ones later)
    ml_conf = {"Buy": 0.5, "Hold": 0.3, "Sell": 0.2}
    pattern_support = {"Buy": 0.1, "Hold": 0.1, "Sell": 0.1}

    model_name = "mistral"  # or "llama3", "gemma2:9b", "phi3"
    llm_out = ask_ollama(model_name, row)
    llm_scores = llm_out.get("action_scores", {})

    for a in ["Buy", "Hold", "Sell"]:
        llm_scores.setdefault(a, 0.0)

    user = MODERATE
    scores = compute_final_scores(
        ml_conf,
        llm_scores,
        pattern_support,
        alpha=0.4,
        beta=0.4,
        gamma=0.2,
        user=user,
    )
    ranking = rank_actions(scores)

    print("User profile:", user.name)
    print("LLM scores:", llm_scores)
    print("Final scores:", scores)
    print("Ranking:", ranking)


if __name__ == "__main__":
    main()
