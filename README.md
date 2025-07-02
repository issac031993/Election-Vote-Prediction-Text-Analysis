# Election-Vote-Prediction-Text-Analysis
Built ML models (SVM, KNN, Naive Bayes, LDA) to predict voter preference based on survey responses. Applied text mining techniques to U.S. presidential speeches to uncover linguistic patterns.
# Bank-Segmentation-Insurance-Claim-Prediction
Used clustering and machine learning models (CART, ANN, Random Forest) to segment bank customers and predict insurance claims. Optimized model performance using test/train validation metrics.
# 🗳️ Decoding Voter Preferences & NLP on Presidential Speeches

---

## 🚀 Project Overview
📈 **Repo:** `Machine-Learning-Voter-Prediction-NLP-Speeches`

This project is divided into two main parts:

1. **Voter Prediction:** Using survey data to predict political party preference (Labour vs Conservative) and support strategic election planning.
2. **NLP Analysis:** Extracting linguistic patterns and key themes from historical US presidential inaugural speeches.

---

## 🗳️ Part 1: Predicting Voter Preferences

### 🔍 Business Problem
A leading news channel, CNBE, wants to analyze recent election survey data to predict voting outcomes and guide exit polls.

### 📊 Dataset Highlights
- **Rows:** 1525 voters
- **Features:** 9 (age, economic perceptions, party preferences, EU stance, political knowledge, gender)
- **Target:** `vote` (Labour / Conservative)

### 🚀 Models & Results
| Model                         | Train Accuracy | Test Accuracy |
|--------------------------------|----------------|---------------|
| Logistic Regression (GridCV)   | 84%            | 82%           |
| LDA (GridCV)                   | 84%            | 81%           |
| K-Nearest Neighbors (GridCV)   | 88%            | 78%           |
| Naïve Bayes                    | 85%            | 81%           |
| Random Forest (GridCV)         | 100% (overfit) | 81%           |
| AdaBoost with Random Forest    | 100% (overfit) | 81%           |

✅ **Insights:**  
- Logistic Regression and LDA showed the best generalization balance.
- KNN had highest train but dropped on test set, indicating potential overfitting.
- Random Forest & AdaBoost achieved perfect train accuracy but did not improve test performance.

### 📈 Key Visuals
- Distribution & box plots for univariate analysis.
- !Distribution plots](https://github.com/user-attachments/assets/60d60c6d-3230-4dda-a495-7fe104908da9)
- ![boxplots](https://github.com/user-attachments/assets/0c74f0f5-4699-453d-9f93-fa6836001ad2)
- ROC curves comparing model performance(taken the 2 best performer)
- ![Random forest](https://github.com/user-attachments/assets/baf769c8-b554-424c-bc9d-c149d038f985)

![ROC Bagging](https://github.com/user-attachments/assets/263cfa0e-86f0-4eae-b53c-0803fe550075)

![ROC Comparison](images/roc_curves.png)

### 💡 Recommendations
- Focus future voter outreach on segments identified as strongly predictive by top models (economic perception, political knowledge).
- Use model probability outputs to prioritize swing voters.

---

## 🇺🇸 Part 2: NLP Analysis of Inaugural Speeches

### 🔍 Overview
Analyzed the inaugural addresses of:
- **Roosevelt (1941)**
- **Kennedy (1961)**
- **Nixon (1973)**

Used **NLTK** to compute:
- Character, word, sentence counts.
- Most frequent words after stopword removal.
- Word clouds to visually represent speech themes.

### 📝 Examples of Findings
| President | Words | Most Frequent Words      |
|-----------|-------|--------------------------|
| Roosevelt | 1526 → 808 | "know", "spirit", "us" |
| Kennedy   | 1543 → 862 | "us", "world", "let"   |
| Nixon     | 2006 → 1035| "us", "peace", "new"   |

![Wordcloud](https://github.com/user-attachments/assets/0d890aa2-0843-4d6a-9728-b6ede1c93141)

✅ **Insights:**  
- “Us” featured heavily in Kennedy and Nixon speeches, emphasizing unity.
- Roosevelt focused on “spirit” & “know”, aligning with wartime resolve.

---

## 🛠️ Tools & Skills Covered
- 🐍 Python: pandas, numpy, seaborn, matplotlib, scikit-learn, nltk, wordcloud
- 📊 Machine Learning: Logistic Regression, LDA, KNN, Naive Bayes, Random Forest, AdaBoost
- ⚙️ Model tuning with GridSearchCV & performance evaluation via ROC-AUC
- 📝 NLP: stopword removal, frequency analysis, word cloud generation

---

## ⚙️ How to Run
- Clone the repo, install packages (`pip install -r requirements.txt`).
- Run `ML project_Issac_Abraham_Problem1.ipynb` for voter prediction.
- Run `ML_project_Issac_Abraham_Problem2.ipynb` for speech NLP analysis.

---

## 🤝 About Me
I completed this project independently, demonstrating my ability to handle full data pipelines from preprocessing to business storytelling.

🔗 [LinkedIn](https://linkedin.com/in/yourprofile)

---

✅ **Thanks for exploring my machine learning & NLP projects!**
