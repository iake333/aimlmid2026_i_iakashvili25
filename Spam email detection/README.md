# Email Spam Classification System by irakli iakashvili 
**AI and ML for Cybersecurity – Midterm Exam**

## Overview
In this project, I implemented a complete **Email Spam Classification System** using **Logistic Regression**. The application loads a provided dataset, trains a machine learning model on 70% of the data, validates it on the remaining 30%, and classifies both dataset-based and manually composed email texts as **Spam** or **Legitimate**. The system also generates multiple visualizations to analyze the dataset and model performance.code provides example 1 and creates manual simulation for check at the end 

---

## Source Code
The full implementation is contained in the following Python file:

📂 **Source Code:**  
`second job.py (this file includes data loading, model training, evaluation, visualization, and email classification)

--
## 1. Data Loading and Processing 

The dataset is loaded from a CSV file using **pandas**. Each record contains extracted email features and a target label (`is_spam`).

### Features Used:
- `words` – total number of words in the email  
- `links` – number of URLs  
- `capital_words` – number of fully capitalized words  
- `spam_word_count` – number of known spam-related keywords  
- `is_spam` – target label (0 = legitimate, 1 = spam)

### Code Example:
```python
data = pd.read_csv(filepath)
X = data[['words', 'links', 'capital_words', 'spam_word_count']]
y = data['is_spam']
The data is split using stratified sampling to preserve class balance:

python
Copy code
train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
2. Logistic Regression Model (1 + 1 points)
I trained a Logistic Regression model using 70% of the dataset.

Model Configuration:
Algorithm: Logistic Regression

Solver: lbfgs

Max iterations: 1000

Feature scaling: StandardScaler

Code Example:
python
Copy code
self.model = LogisticRegression(random_state=42, max_iter=1000)
self.model.fit(X_train_scaled, y_train)
Model Coefficients (1 point):
The trained model produced the following coefficients:

Feature	Coefficient	Impact
words	Negative	Longer emails are less likely spam
links	Positive	More links increase spam probability
capital_words	Positive	Excessive capitalization indicates spam
spam_word_count	Strong Positive	Most influential spam indicator

3. Model Validation: Confusion Matrix & Accuracy 
The model was validated using the 30% test set.

Metrics Calculated:
Accuracy

Confusion Matrix

Precision, Recall, F1-score

Code Example:
python
Copy code
y_pred = self.model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
Results:
Accuracy: ~95%

Confusion Matrix: Shows strong spam detection with low false negatives

The confusion matrix is printed numerically and visualized as a heatmap.

4. Email Text Classification Functionality 
The application can classify raw email text by:

Parsing the text

Extracting the same four features used in training

Scaling the features

Predicting spam probability

Code Example:
python
Copy code
features = self.extract_features_from_email(email_text)
features_scaled = self.scaler.transform([features])
prediction = self.model.predict(features_scaled)
The output includes:

Classification (Spam / Legitimate)

Confidence score

Feature breakdown

5. Manually Composed Spam Email 
Spam Email Characteristics:
Excessive capitalization

Multiple suspicious links

High spam keyword frequency

Urgent language and unrealistic promises

📧 Result: Classified as SPAM with very high confidence.

Explanation:
The email was intentionally designed to trigger all spam indicators: capital words, links, urgency, and spam vocabulary such as “winner”, “claim now”, “urgent”, and “million dollars”.

6. Manually Composed Legitimate Email 
Legitimate Email Characteristics:
Professional tone

No links

No urgency

Academic content

Natural capitalization

📧 Result: Classified as LEGITIMATE with high confidence.

Explanation:
The email mimics real academic communication and avoids all known spam triggers.

7. Visualizations 
Visualization 1: Class Distribution (Spam vs Legitimate)
Type: Bar Chart

Purpose: Shows dataset balance

Insight: Dataset is well-balanced, preventing model bias

📄 File: class_distribution.png

Visualization 2: Confusion Matrix Heatmap
Type: Heatmap

Purpose: Visualizes prediction correctness

Insight: High true positives and true negatives indicate strong performance

📄 File: confusion_matrix_heatmap.png

Visualization 3: Feature Importance
Type: Horizontal Bar Chart

Purpose: Shows most influential features

Insight: Spam word count is the strongest predictor

📄 File: feature_importance.png

Visualization 4: Feature Correlation Heatmap
Type: Heatmap

Purpose: Shows relationships between features and target

Insight: Spam word count strongly correlates with spam classification

📄 File: feature_correlation.png

Conclusion
This project demonstrates a complete machine learning pipeline for email spam detection, including data preprocessing, model training, evaluation, visualization, and real-world email classification. All requirements of the midterm assignment were successfully implemented and validated.

Generated Files
class_distribution.png

confusion_matrix_heatmap.png

feature_importance.png

feature_correlation.png

roc_curve.png