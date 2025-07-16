# 📊 RAGAs Metric Evaluation for LLM Logs

This project evaluates the quality of LLM responses using **RAGAs metrics**:
- ✅ **Faithfulness**
- 🎯 **Answer Relevancy**
- 🔍 **Context Precision**

Given a log of LLM interactions in JSON format, the script computes quality metrics for each response using the [RAGAs](https://github.com/explodinggradients/ragas) framework.

---

## 📁 Files

```
.
├── ragas_integration.py         # Main script to compute metrics
├── test_log.json                # Input: Log of LLM queries and outputs
├── ragas_scores_output.json     # Output: Scores for each item
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## ✅ Example Input (`test_log.json`)

```json
{
  "items": [
    {
      "id": "item-001",
      "input": {
        "system": "You are an expert in computer science and Python programming.",
        "user": "What is the difference between a list and a tuple in Python?"
      },
      "expected_output": "Lists in Python are mutable, meaning their elements can be changed. Tuples, on the other hand, are immutable. Also, lists use square brackets [], while tuples use parentheses ()."
    },
    {
      "id": "item-002",
      "input": {
        "system": "You are a machine learning assistant. Provide accurate, concise answers.",
        "user": "What is overfitting in machine learning?"
      },
      "expected_output": "Overfitting is when a model learns the training data too well, including noise, and fails to generalize to new data. It performs well on training data but poorly on unseen data."
    }
  ]
}
```

---

## 📤 Example Output (`ragas_scores_output.json`)

```json
[
  {
    "id": "item-001",
    "faithfulness": 0.80,
    "answer_relevancy": 0.88,
    "context_precision": 0.85
  },
  {
    "id": "item-002",
    "faithfulness": 0.79,
    "answer_relevancy": 0.87,
    "context_precision": 0.88
  }
]
```

---

## 🧪 Setup Instructions

Install required packages using:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
ragas==0.2.15
datasets
tqdm
pandas
openai
langchain
langchain-openai
transformers
```

---

## 🧠 How It Works

1. **Read Log File**: Loads LLM logs from `test_log.json`.
2. **Dataset Construction**: Extracts:
   - `system` → Context
   - `user` → Query
   - `expected_output` → Answer + Reference
3. **Metric Evaluation**: Uses RAGAs to compute:
   - `faithfulness`
   - `answer_relevancy`
   - `context_precision`
4. **Result Saving**: Writes results to `ragas_scores_output.json`.

---

## ▶️ Run the Script

```bash
python ragas_integration.py
```

If `test_log.json` is in the same folder, this will generate `ragas_scores_output.json`.

---

## 🙋‍♂️ Author
Vardaan Gulati : 
National Institute of Technology, Delhi

