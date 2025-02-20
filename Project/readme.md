# 🔍 BERT for Sentiment Analysis  

## 📌 Project Overview  
This project focuses on developing a **reduced version of BERT** to achieve efficient and effective results in **sentiment classification of movie reviews**. It involves both **pre-training and fine-tuning**, with **classification of IMDB movie reviews** as the fine-tuning task.  

The motivation behind this project is to explore **efficient BERT implementations** that maintain strong accuracy while reducing computational overhead.  

---

## 📖 Background  
**BERT (Bidirectional Encoder Representations from Transformers)** is a deep learning model developed by **Google AI**, designed to understand **contextual meaning in text bidirectionally**. Unlike previous NLP models that processed text in one direction, BERT captures the **full context of words** using a **Transformer-based neural network**.  

The key innovations in BERT include:  
- **Bidirectional text processing** for better contextual understanding.  
- **Masked Language Modeling (MLM)**: Pretraining using randomly masked words to improve predictions.  
- **Next Sentence Prediction (NSP)**: Learning sentence relationships for tasks like question-answering and text completion.  

This project leverages **BERT-tiny**, a lightweight version of BERT, to perform **sentiment analysis** while optimizing for **CPU efficiency**.  

---

## 🏗 Model Architecture  

### **1️⃣ Pre-training Phase**  
The **pre-training phase** uses a **BERT-tiny model** with the following configurations:  
- **Hidden size:** 128  
- **Attention heads per layer:** 2  
- **Transformer layers:** 2  
- **Feedforward dimensions:** 512  

This **smaller architecture** allows for **faster processing and lower memory usage**, making it ideal for **CPU-based environments**.  

### **2️⃣ Input Representation**  
- Uses **WordPiece tokenization** to handle rare words.  
- Incorporates **position embeddings** to track word order in sentences.  
- Processes input **bidirectionally** for full contextual understanding.  

### **3️⃣ Fine-tuning for Sentiment Analysis**  
- Uses the **IMDB movie review dataset** with **positive and negative sentiment labels**.  
- Fine-tunes a **pretrained BERT model** for binary classification.  
- Evaluates performance using **accuracy metrics**.  

---

## 📊 Model Training  
### **Pre-training Tasks:**  
✅ **Masked Language Modeling (MLM):** Predicting missing words in a sentence.  
✅ **Next Sentence Prediction (NSP):** Determining if two sentences are logically connected.  

### **Fine-tuning:**  
✅ **Dataset:** IMDB movie review dataset (**50,000 reviews**)  
✅ **Objective:** Classify movie reviews as **positive or negative**  
✅ **Accuracy Achieved:** **81.91%**  

---


