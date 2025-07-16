import json
import os
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from tqdm import tqdm



def load_log_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_dataset(log_data):
    data = {
        "id": [],
        "context": [],
        "question": [],
        "answer": [],
        "retrieved_contexts": [],
        "reference": []
    }

    
    items = []
    if isinstance(log_data, list):
        items = log_data
    elif isinstance(log_data, dict):
        items = log_data.get("items", [])

    for item in tqdm(items, desc="Preparing dataset"):
        try:
            item_id = item.get("id", f"item-{len(data['id'])+1:03d}")
            system_prompt = item.get("input", {}).get("system", "")
            user_prompt = item.get("input", {}).get("user", "")
            answer = item.get("expected_output", "")

            if not (user_prompt and answer):
                continue

            data["id"].append(item_id)
            data["context"].append(system_prompt)
            data["question"].append(user_prompt)
            data["answer"].append(answer)
            data["retrieved_contexts"].append([system_prompt])
            data["reference"].append(answer)

        except Exception as e:
            print(f"❌ Error processing item: {e}")

    return Dataset.from_dict(data)

def compute_ragas_scores(dataset):
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )
    return result.to_pandas()

def save_scores_to_json(df, output_path):
    scores = []
    for idx, row in df.iterrows():
        scores.append({
            "id": row["id"],
            "faithfulness": round(float(row.get("faithfulness", 0.0)), 4),
            "answer_relevancy": round(float(row.get("answer_relevancy", 0.0)), 4),
            "context_precision": round(float(row.get("context_precision", 0.0)), 4)
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"✅ RAGAS scores saved to {output_path}")

def main():
    input_path = "test_log.json"
    output_path = "ragas_scores_output.json"

    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    log_data = load_log_file(input_path)
    dataset = prepare_dataset(log_data)
    scores_df = compute_ragas_scores(dataset)
    save_scores_to_json(scores_df, output_path)

if __name__ == "__main__":
    main()
