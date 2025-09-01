# 🤖 BERT: Pre-training and Fine-tuning from Scratch

This repository presents an end-to-end implementation of **BERT (Bidirectional Encoder Representations from Transformers)**, covering both **pre-training from scratch** and **fine-tuning for sentiment classification**. The project showcases how bidirectionality is achieved in BERT, the datasets used for training, and the results obtained when applying the model to real-world tasks.

---

## 📌 Key Features

✅ **Pre-training BERT from scratch** with the original objectives: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) using BookCorpus 
✅ **Fine-tuning BERT** on the IMDB dataset for movie review sentiment classification  
✅ **Complete workflow in Jupyter Notebook**: from data preparation, model setup, pre-training, to fine-tuning and evaluation  
✅ **Annotated reference paper (BERT.pdf)** included for deeper understanding  

---

## 📂 Repository Structure

```
📁 bert-project/
│
├── 📘 BERT.ipynb               # Jupyter Notebook implementing pre-training and fine-tuning from scratch
├── 📘 BERT.pdf                 # Original BERT paper with my annotations
└── 📄 README.md                
```

---

## 🧪 Notebook Overview: `BERT.ipynb`

This notebook contains the full BERT workflow:

### 🔍 Pre-training Stage

**Objective:** Train a BERT model from scratch to achieve bidirectionality.

- Implements **Masked Language Modeling (MLM)**: randomly masks tokens in a sentence and trains the model to predict them using both left and right context.  
- Implements **Next Sentence Prediction (NSP)**: trains the model to determine whether one sentence follows another, enabling the model to learn inter-sentence relationships.  
- Pre-training performed using the **BookCorpus dataset**, as in the original BERT paper.

---

### 🎯 Fine-tuning Stage

**Objective:** Adapt the pre-trained model for downstream classification.

- Fine-tuned on the **IMDB dataset** (50,000 balanced positive/negative movie reviews).  
- Added a **classification head** on top of the pre-trained BERT encoder.  
- Achieved an **F1-Score of 0.7947** on sentiment classification.  

---

## 📊 Results

- Pre-training successfully replicated the **bidirectional context understanding** that makes BERT powerful.  
- Fine-tuning demonstrated that transfer learning with pre-trained transformers **outperforms traditional ML models** like Naive Bayes and Logistic Regression on text classification tasks.  
- The achieved **F1-Score of 0.7947** validates the effectiveness of the pipeline.

---

## 📈 Conclusion

This project demonstrates how BERT can be built **from the ground up** and adapted to real-world tasks through fine-tuning. By replicating both pre-training objectives (MLM + NSP) and applying them to downstream tasks, the work highlights the strength of **transformer-based models** in natural language processing.

---

