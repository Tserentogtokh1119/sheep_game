
# logic regression bolon decision tree haritsuulalt

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)


# HTTPS presence: Checks if URL starts with "https"
# Dot count: Number of dots (subdomains often indicate phishing)
# Special characters: Presence of @, #, $, %, & (commonly used in phishing)
# Digit count: Number of digits in URL
# URL length: Total length (longer URLs often suspicious)
# Dash count: Number of hyphens
# Query parameters: Presence of ? (used for passing data)
# Subdomain depth: How many subdomains exist
# Domain age: How old the domain is (simulated in this code)

# Өгөгдлийг унших

df = pd.read_csv("D:\semester5\kaggle_project1\phishingLink\data\phishing_simple.csv")

# Label - ийг тоон утга болгон хувиргах. Хэрэв phishing байвал 1, benign байвал 0
df["label"] = df["label"].map({"benign": 0, "phishing": 1})

# Шинж чанарууд (feature) үүсгэх 
def extract_features(df):
    df["https"] = df["URL"].apply(lambda x: 1 if str(x).startswith("https") else 0) # https байвал 1, үгүй бол 0
    df["dot_count"] = df["URL"].apply(lambda x: str(x).count('.')) # цэгийг тоолох
    df["special_char"] = df["URL"].apply(lambda x: 1 if any(c in str(x) for c in ['@', '#', '$', '%', '&']) else 0)
    df["digit_count"] = df["URL"].apply(lambda x: sum(c.isdigit() for c in str(x)))
    df["url_length"] = df["URL"].apply(lambda x: len(str(x)))
    df["dash_count"] = df["URL"].apply(lambda x: str(x).count('-'))
    df["query_params"] = df["URL"].apply(lambda x: 1 if '?' in str(x) else 0)
    df["subdomain_depth"] = df["URL"].apply(lambda x: str(x).count('.'))
    # Домайн нас (жишээ - бодит өгөгдөл өөр өөр аргаар тооцно.)
    np.random.seed(42)
    df["domain_age_days"] = np.random.randint(1, 365*5, len(df))
    return df

df = extract_features(df)

# Онцлог шинж чанарууд
features = ["length", "https", "dot_count", "special_char", "digit_count", "url_length", "dash_count", 
            "query_params", "subdomain_depth", "domain_age_days"]
X = df[features]
y = df["label"]

# Өгөгдлийг стандартчилах 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# Өгөгдлийг хуваах
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("="*60)
print("PHISHING ЛИНК ИЛРҮҮЛЭХ ХОЁР АЛГОРИТМЫН ХАРЬЦУУЛАЛТ")
print("="*60)
print(f"Өгөгдлийн хэмжээ: {X.shape}")
print(f"Phishing эзлэх хувь: {y.mean()*100:.2f}%")
print(f"Сургалтын өгөгдөл: {X_train.shape}")
print(f"Тестийн өгөгдөл: {X_test.shape}")
print("="*60)

# Ашиглах алгоритмууд
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Зөвхөн 2 алгоритмыг тодорхойлох
algorithms = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Үр дүнг хадгалах
results = []

# Алгоритм бүрийг сургах ба үнэлэх
for name, model in algorithms.items():
    print(f"\n{name} алгоритмыг сургаж байна...")
    
    # Зангаар сургах 
    model.fit(X_train, y_train)
    
    # Таамаглах
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Үнэлгээний үзүүлэлтүүд
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Үр дүнг таамаглах 
    results.append({
        'Algorithm': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc if roc_auc is not None else 'N/A',
        'CV F1 Mean': cv_mean,
        'CV F1 Std': cv_std,
        'Model': model
    })
    
    # Дэлгэрэнгүй мэдээлэл хэвлэх
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  CV F1-Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")

# Үр дүнг dataframe-д хадгалах
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("АЛГОРИТМЫН ХАРЬЦУУЛАЛТ")
print("="*60)
print(results_df[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))

# ВИЗУАЛЧЛАЛ 1: Үнэлгээний үзүүлэлтүүдийн харьцуулалт 
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Хоёр Алгоритмын Үнэлгээний Үзүүлэлтүүдийн Харьцуулалт', fontsize=16)

# Accuracy
bars1 = axes[0, 0].bar(results_df['Algorithm'], results_df['Accuracy'], color=['blue', 'green'])
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Accuracy')
axes[0, 0].axhline(y=results_df['Accuracy'].mean(), color='r', linestyle='--', alpha=0.7)
for bar in bars1:
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')

# Precision
bars2 = axes[0, 1].bar(results_df['Algorithm'], results_df['Precision'], color=['blue', 'green'])
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision')
axes[0, 1].axhline(y=results_df['Precision'].mean(), color='r', linestyle='--', alpha=0.7)
for bar in bars2:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')

# Recall
bars3 = axes[0, 2].bar(results_df['Algorithm'], results_df['Recall'], color=['blue', 'green'])
axes[0, 2].set_ylabel('Recall')
axes[0, 2].set_title('Recall')
axes[0, 2].axhline(y=results_df['Recall'].mean(), color='r', linestyle='--', alpha=0.7)
for bar in bars3:
    height = bar.get_height()
    axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')

# F1-Score
bars4 = axes[1, 0].bar(results_df['Algorithm'], results_df['F1-Score'], color=['blue', 'green'])
axes[1, 0].set_ylabel('F1-Score')
axes[1, 0].set_title('F1-Score')
axes[1, 0].axhline(y=results_df['F1-Score'].mean(), color='r', linestyle='--', alpha=0.7)
for bar in bars4:
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')

# ROC-AUC
roc_auc_values = results_df[results_df['ROC-AUC'] != 'N/A']['ROC-AUC']
roc_auc_algorithms = results_df[results_df['ROC-AUC'] != 'N/A']['Algorithm']
bars5 = axes[1, 1].bar(roc_auc_algorithms, roc_auc_values, color=['blue', 'green'])
axes[1, 1].set_ylabel('ROC-AUC')
axes[1, 1].set_title('ROC-AUC')
if len(roc_auc_values) > 0:
    axes[1, 1].axhline(y=roc_auc_values.mean(), color='r', linestyle='--', alpha=0.7)
for bar in bars5:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')

# CV F1-Score
bars6 = axes[1, 2].bar(results_df['Algorithm'], results_df['CV F1 Mean'], color=['blue', 'green'])
axes[1, 2].set_ylabel('CV F1 Mean')
axes[1, 2].set_title('Cross-Validation F1 Score')
axes[1, 2].axhline(y=results_df['CV F1 Mean'].mean(), color='r', linestyle='--', alpha=0.7)
for bar in bars6:
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ВИЗУАЛЧЛАЛ 2: ROC муруй
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['blue', 'green']
for i, (idx, row) in enumerate(results_df.iterrows()):
    if row['ROC-AUC'] != 'N/A':
        model = row['Model']
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ax.plot(fpr, tpr, label=f"{row['Algorithm']} (AUC = {row['ROC-AUC']:.3f})", 
                color=colors[i], linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Хоёр Алгоритмын Харьцуулалт')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.show()

# ВИЗУАЛЧЛАЛ 3: Confusion матрицууд
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Хоёр Алгоритмын Confusion Matrix', fontsize=16)

for idx, (ax, row) in enumerate(zip(axes, results_df.iterrows())):
    name, data = row
    model = data['Model']
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Phishing'],
                yticklabels=['Benign', 'Phishing'])
    ax.set_title(f"{data['Algorithm']}\nF1-Score: {data['F1-Score']:.3f}")
    ax.set_xlabel('Таамаглагдсан')
    ax.set_ylabel('Бодит')

plt.tight_layout()
plt.show()

# ВИЗУАЛЧЛАЛ 4: Алгоритмын гүйцэтгэлийн дулааны зураг
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV F1 Mean']
heatmap_data = results_df[['Algorithm'] + metrics].set_index('Algorithm')

plt.figure(figsize=(10, 4))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', center=0.8)
plt.title('Хоёр Алгоритмын Гүйцэтгэлийн Харьцуулалт')
plt.tight_layout()
plt.show()

# ВИЗУАЛЧЛАЛ 5: Шинж чанаруудын ач холбогдол (зөвхөн Decision Tree-д)
print("\n" + "="*60)
print("ШИНЖ ЧАНАРУУДЫН АЧ ХОЛБОГДОЛ (DECISION TREE)")
print("="*60)

# Decision Tree-ийн шинж чанаруудын ач холбогдол
dt_model = results_df[results_df['Algorithm'] == 'Decision Tree']['Model'].values[0]
dt_importances = dt_model.feature_importances_

# Logistic Regression-ийн коэффициентууд (үнэмлэхүй утгаар)
lr_model = results_df[results_df['Algorithm'] == 'Logistic Regression']['Model'].values[0]
lr_coefs = np.abs(lr_model.coef_[0])

# Харьцуулалт
importance_df = pd.DataFrame({
    'Feature': features,
    'Decision Tree Importance': dt_importances,
    'Logistic Regression Coef (abs)': lr_coefs
})

importance_df = importance_df.sort_values('Decision Tree Importance', ascending=False)

print("\nШинж чанаруудын ач холбогдол:")
print(importance_df.to_string(index=False))

# Шинж чанаруудын ач холбогдлыг визуалчлах
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Decision Tree
axes[0].barh(importance_df['Feature'], importance_df['Decision Tree Importance'], color='green')
axes[0].set_xlabel('Ач холбогдол')
axes[0].set_title('Decision Tree - Шинж чанаруудын ач холбогдол')

# Logistic Regression
axes[1].barh(importance_df['Feature'], importance_df['Logistic Regression Coef (abs)'], color='blue')
axes[1].set_xlabel('Коэффициентийн үнэмлэхүй утга')
axes[1].set_title('Logistic Regression - Шинж чанаруудын нөлөө')

plt.tight_layout()
plt.show()

# Алгоритмын онцлог шинжүүдийн харьцуулалт 
print("\n" + "="*60)
print("АЛГОРИТМЫН ОНЦЛОГ ШИНЖҮҮДИЙН ХАРЬЦУУЛАЛТ")
print("="*60)

algorithm_properties = {
    'Algorithm': ['Logistic Regression', 'Decision Tree'],
    'Төрөл': ['Шугаман', 'Модонд суурилсан'],
    'Тайлбарлах чадвар': ['Өндөр', 'Өндөр'],
    'Сургалтын хурд': ['Хурдан', 'Хурдан'],
    'Overfitting эрсдэл': ['Бага', 'Өндөр'],
    'Давуу тал': ['Шугаман хэв маягийг сайн таньдаг', 
                  'Шугаман бус хамаарлыг сайн таньдаг'],
    'Сул тал': ['Шугаман бус хамаарлыг таньж чаддаггүй',
                'Overfitting болох эрсдэл өндөр'],
    'Phishing илрүүлэхэд тохиромжтой': ['Энгийн, тодорхой шинж чанарууд',
                                         'Нарийн төвөгтэй харилцан үйлчлэлтэй шинж чанарууд']
}

properties_df = pd.DataFrame(algorithm_properties)
print(properties_df.to_string(index=False))

# Шилдэг алгоритмыг сонгох
best_f1 = results_df.loc[results_df['F1-Score'].idxmax()]

print("\n" + "="*60)
print("ДҮГНЭЛТ")
print("="*60)
print(f"Хамгийн өндөр F1-Score: {best_f1['Algorithm']} (F1 = {best_f1['F1-Score']:.4f})")

# Нарийвчилсан дүн шинжилгээ
print("\n" + "="*60)
print("НАРИЙВЧИЛСАН ШИНЖИЛГЭЭ")
print("="*60)

print("""
Logistic Regression:
- Давуу тал: Шугаман хамаарлыг сайн таньдаг, интерпретаци хийхэд хялбар
- Сул тал: Нарийн төвөгтэй, шугаман бус харилцан үйлчлэлийг таньж чаддаггүй

Decision Tree:
- Давуу тал: Шугаман бус хамаарлыг таньж чадна, интерпретаци хийхэд хялбар
- Сул тал: Overfitting болох эрсдэл өндөр, жижиг өөрчлөлтөд мэдрэмтгий

Phishing илрүүлэхэд хамгийн чухал шинж чанарууд:
1. URL урт (урт URL ихэвчлэн phishing)
2. Дэд домайны тоо
3. Тусгай тэмдэгтүүд (@, #, $ гэх мэт)
4. HTTPS байх эсэх

ЗӨВЛӨМЖ:
1. Хэрэв танд алгоритмын шийдвэрийг тайлбарлах (interpretability) чухал бол: Decision Tree
2. Хэрэв танд илүү тогтвортой алгоритм хэрэгтэй бол: Logistic Regression
3. Хэрэв өгөгдлийнхээ шугаман бус хамаарлыг ашиглахыг хүсвэл: Decision Tree
4. Хэрэв өгөгдөл шугаман хамааралтай бол: Logistic Regression
""")

# Нэмэлт: GridSearch ашиглан гиперпараметрийг оновчтой болгох
print("\n" + "="*60)
print("ГИПЕРПАРАМЕТРИЙГ ОНОВЧТОЙ БОЛГОХ (GRIDSEARCH)")
print("="*60)

# Logistic Regression параметрүүд
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Decision Tree параметрүүд
dt_param_grid = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# GridSearch хийх
print("\nLogistic Regression гиперпараметрийг оновчтой болгох...")
lr_grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), 
                       lr_param_grid, cv=5, scoring='f1', n_jobs=-1)
lr_grid.fit(X_train, y_train)
print(f"Шилдэг параметрүүд: {lr_grid.best_params_}")
print(f"Шилдэг F1-Score: {lr_grid.best_score_:.4f}")

print("\nDecision Tree гиперпараметрийг оновчтой болгох...")
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                       dt_param_grid, cv=5, scoring='f1', n_jobs=-1)
dt_grid.fit(X_train, y_train)
print(f"Шилдэг параметрүүд: {dt_grid.best_params_}")
print(f"Шилдэг F1-Score: {dt_grid.best_score_:.4f}")

# Оновчтой болгосон загваруудыг харьцуулах
print("\n" + "="*60)
print("ОНОВЧТОЙ БОЛГОСОН ЗАГВАРУУДЫН ХАРЬЦУУЛАЛТ")
print("="*60)

optimized_lr = lr_grid.best_estimator_
optimized_dt = dt_grid.best_estimator_

optimized_lr.fit(X_train, y_train)
optimized_dt.fit(X_train, y_train)

y_pred_lr = optimized_lr.predict(X_test)
y_pred_dt = optimized_dt.predict(X_test)

optimized_results = pd.DataFrame({
    'Algorithm': ['Logistic Regression (Optimized)', 'Decision Tree (Optimized)'],
    'Accuracy': [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_dt)],
    'Precision': [precision_score(y_test, y_pred_lr), precision_score(y_test, y_pred_dt)],
    'Recall': [recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_dt)],
    'F1-Score': [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_dt)]
})

print("\nОновчтой болгосон загваруудын үр дүн:")
print(optimized_results.to_string(index=False))

# Анхны болон оновчтой загваруудын харьцуулалт
comparison_df = pd.concat([
    results_df[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']],
    optimized_results
], ignore_index=True)

print("\n" + "="*60)
print("АНХНЫ БА ОНОВЧТОЙ ЗАГВАРУУДЫН ХАРЬЦУУЛАЛТ")
print("="*60)
print(comparison_df.to_string(index=False))