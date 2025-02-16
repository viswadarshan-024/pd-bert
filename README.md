# Fine-Tuned BERT Model for Paraphrase Detection

### Model Description
This is a fine-tuned version of **BERT-base** for **paraphrase detection**, trained on four benchmark datasets: **MRPC, QQP, PAWS-X, and PIT**. The model is designed for applications such as **duplicate content detection, question answering, and semantic similarity analysis**. It offers strong recall capabilities, making it effective in identifying paraphrases even in complex sentence structures.

- **Developed by:** Viswadarshan R R  
- **Model Type:** Transformer-based Sentence Pair Classifier  
- **Language:** English  
- **Finetuned from:** `bert-base-cased`

### Model Sources

- **Repository:** [Hugging Face Model Hub](https://huggingface.co/viswadarshan06/pd-bert/)  
- **Research Paper:** _Comparative Insights into Modern Architectures for Paraphrase Detection_ (Accepted at ICCIDS 2025)  
- **Demo:** (To be added upon deployment)  

## Uses

### Direct Use
- Identifying **duplicate questions** in customer support and FAQs.  
- Improving **semantic search** in retrieval-based systems.  
- Enhancing **document deduplication** and text similarity applications.  

### Downstream Use
This model can be further fine-tuned on domain-specific paraphrase datasets for industries such as **healthcare, legal, and finance**.

### Out-of-Scope Use
- The model is **monolingual** and trained only on **English datasets**, requiring additional fine-tuning for multilingual tasks.  
- May struggle with **idiomatic expressions** or complex figurative language.  

## Bias, Risks, and Limitations

### Known Limitations
- **Higher recall but lower precision**: The model tends to over-identify paraphrases, leading to increased false positives.  
- **Contextual ambiguity**: May misinterpret sentences that require deep contextual reasoning.  

### Recommendations
Users can mitigate the **false positive rate** by applying post-processing techniques or confidence threshold tuning.  

## How to Get Started with the Model

To use the model, install **transformers** and load the fine-tuned model as follows:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
model_path = "viswadarshan06/pd-bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Encode sentence pairs
inputs = tokenizer("The car is fast.", "The vehicle moves quickly.", return_tensors="pt", padding=True, truncation=True)

# Get predictions
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax().item()
print("Paraphrase" if predicted_class == 1 else "Not a Paraphrase")
```

## Training Details  

This model was trained using a combination of four datasets:

- **MRPC**: News-based paraphrases.  
- **QQP**: Duplicate question detection.  
- **PAWS-X**: Adversarial paraphrases for robustness testing.  
- **PIT**: Short-text paraphrase dataset.  

### Training Procedure

- **Tokenizer**: BERT Tokenizer  
- **Batch Size**: 16  
- **Optimizer**: AdamW  
- **Loss Function**: Cross-entropy  

#### Training Hyperparameters
- **Learning Rate**: 2e-5  
- **Sequence Length**:  
  - MRPC: 256  
  - QQP: 336  
  - PIT: 64  
  - PAWS-X: 256  

#### Speeds, Sizes, Times  

- **GPU Used**: NVIDIA A100  
- **Total Training Time**: ~6 hours  
- **Compute Units Used**: 80  

### Testing Data, Factors & Metrics  

#### Testing Data  
The model was tested on combined test sets and evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Runtime  

### Results  

## **BERT Model Evaluation Metrics**  
| Model   | Dataset     | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Runtime (sec) |  
|---------|------------|-------------|--------------|------------|-------------|---------------|  
| BERT | MRPC Validation | 88.24 | 88.37 | 95.34 | 91.72 | 1.41 |  
| BERT | MRPC Test | 84.87 | 85.84 | 92.50 | 89.04 | 5.77 |  
| BERT | QQP Validation | 87.92 | 81.44 | 86.86 | 84.06 | 43.24 |  
| BERT | QQP Test | 88.14 | 82.49 | 86.56 | 84.47 | 43.51 |  
| BERT | PAWS-X Validation | 91.90 | 87.57 | 94.67 | 90.98 | 6.73 |  
| BERT | PAWS-X Test | 92.60 | 88.69 | 95.92 | 92.16 | 6.82 |  
| BERT | PIT Validation | 77.38 | 72.41 | 58.57 | 64.76 | 4.34 |  
| BERT | PIT Test | 86.16 | 64.11 | 76.57 | 69.79 | 0.98 |  

### Summary  
This **BERT-based Paraphrase Detection Model** demonstrates strong **recall capabilities**, making it highly effective at **identifying paraphrases** across varied linguistic structures. While it tends to overpredict paraphrases, it remains a **strong baseline** for **semantic similarity tasks** and can be fine-tuned further for **domain-specific applications**.  

### **Citation**  

If you use this model, please cite:  

```bibtex
@inproceedings{viswadarshan2025paraphrase,
   title={Comparative Insights into Modern Architectures for Paraphrase Detection},
   author={Viswadarshan R R, Viswaa Selvam S, Felcia Lilian J, Mahalakshmi S},
   booktitle={International Conference on Computational Intelligence, Data Science, and Security (ICCIDS)},
   year={2025},
   publisher={IFIP AICT Series by Springer}
}
```  

## Model Card Contact  

📧 Email: viswadarshanrramiya@gmail.com  

🔗 GitHub: [Viswadarshan R R](https://github.com/viswadarshan-024)  
