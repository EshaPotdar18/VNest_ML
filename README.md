# 🤖 ML/NLP Engineer Intern Challenge – Sentiment Classification on Amazon Reviews

## 🎯 Objective  
Build an end-to-end sentiment classification pipeline using both **NLTK** and **Hugging Face Transformers**. Showcase skills in data preprocessing, rule-based vs. transformer-based sentiment scoring, evaluation, and multilingual extension.

## 📋 Task Overview

- Select a labeled dataset (`Amazon Fine Food Reviews`)
- Perform EDA and visualization of class distribution
- Run rule-based sentiment scoring using VADER
- Run transformer-based scoring using RoBERTa
- Compare model predictions and evaluate
- Visualize model agreement and disagreement
- Bonus: Discuss multilingual extension strategy


## 📁 Project Structure

/notebooks/
├── ML_Internship_VNest.ipynb # Complete pipeline: EDA, NLTK, VADER, RoBERTa, comparison, evaluation

/data/
├── Reviews.csv # Subset (500 samples) of Amazon reviews

Root Files
├── README.md # This file

---

## 📦 Dataset

- **Source**: [Amazon Fine Food Reviews (Kaggle)](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- Downloaded via `kagglehub`
- Sample size: 500 reviews

---

## 🔁 Pipeline Overview

### Step 0: Download Dataset from Kaggle

```python
import kagglehub
snap_amazon_fine_food_reviews_path = kagglehub.dataset_download('snap/amazon-fine-food-reviews')
```
### Step 1: Load & Explore Data

- Load reviews from Reviews.csv
- Quick EDA using seaborn & matplotlib
- Visualize distribution of star ratings

### Step 2: NLTK Tokenization & Named Entity Recognition

- Tokenization using TreebankWordTokenizer
- POS tagging
- Named Entity Recognition using maxent_ne_chunker
  
### Step 3: Rule-Based Sentiment (VADER)

- SentimentIntensityAnalyzer from NLTK
- Produces neg, neu, pos, and compound scores
- Visualized sentiment scores across ratings
  
### Step 4: Transformer-Based Sentiment (RoBERTa)

- Model: cardiffnlp/twitter-roberta-base-sentiment
- Tokenized using AutoTokenizer and scored using AutoModelForSequenceClassification
- Applied softmax to raw logits
- Compared RoBERTa scores to VADER

### Step 5: Result Comparison

- Merged outputs from both models
- Compared scores using seaborn pairplot
- Explored misaligned examples: 1-star with positive sentiment, 5-star with negative sentiment

**## Model Evaluation**

Evaluated the RoBERTa sentiment predictions by mapping star ratings to binary sentiment (positive if score > 3):
```python
from sklearn.metrics import classification_report
y_true = results_df['Score'].apply(lambda x: 'positive' if x > 3 else 'negative')
y_pred = results_df['roberta_pos'].apply(lambda x: 'positive' if x > 0.5 else 'negative')
print(classification_report(y_true, y_pred))
```
## Model Insights
- VADER is quick and rule-based, but lacks deep contextual understanding.
- RoBERTa shows much better context-aware performance.
- Edge cases revealed mismatches between star ratings and sentiment.

## Suggested Improvements
- Fine-tune RoBERTa on this dataset for better alignment with Amazon-style reviews.
- Use stratified sampling to balance class distribution.
- Extend the model to support multilingual sentiment analysis (see below).

**🌍 Multilingual Extension**

✅ Use a multilingual model:
```python
MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
```
- Supports over 100 languages.
- Works well on non-English product reviews or multilingual social media posts.

## Additional Steps
- Include multilingual samples via translation (e.g., googletrans) or public datasets like amazon-massive, paws-x.
- Fine-tune the model on mixed-language datasets.
- Use a language detector (e.g., langdetect, langid, or Hugging Face papluca/xlm-roberta-base-language-detection) to switch models dynamically.

**💡 Insight**
- Multilingual transformers like XLM-RoBERTa allow you to maintain one model for many languages, enabling global scalability without sacrificing context.

**📁 Files**
- File	Description
- Reviews.csv	Amazon Fine Food Reviews
- notebook.ipynb	Jupyter Notebook with full pipeline
- README.md	Project documentation

## 📝 Requirements
'''python
pandas
numpy
matplotlib
seaborn
nltk
tqdm
transformers
scipy
kagglehub
'''
## Expected Deliverables
✅ Complete notebook with analysis
✅ Evaluation metrics with comparison plots
✅ Insights report (model_report.md)
✅ Optional multilingual proposal

## Author
- Final-year Computer Science Engineer
- Passionate about NLP, multilingual modeling, and practical AI solutions
- Made using Python, NLTK, and Hugging Face 🤗
