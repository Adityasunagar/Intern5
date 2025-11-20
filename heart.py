import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import graphviz

# 1. Load data
df = pd.read_csv('heart.csv')  # adjust path
X = df.drop('target', axis=1)
y = df['target']

# 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42)

# 3. Baseline Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_train = dt.predict(X_train)
y_pred_test  = dt.predict(X_test)
print("Decision Tree (no depth limit) – train acc:", accuracy_score(y_train, y_pred_train))
print("Decision Tree – test acc:",  accuracy_score(y_test,  y_pred_test))

# 4. Control tree depth
for depth in [2,3,4,5,6,7,8]:
    dt2 = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt2.fit(X_train, y_train)
    acc = accuracy_score(y_test, dt2.predict(X_test))
    print("Depth:", depth, " Test accuracy:", acc)

# pick best depth e.g. depth=4
dt_best = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_best.fit(X_train, y_train)

# visualize
dot_data = export_graphviz(dt_best, out_file=None,
                           feature_names=X.columns,
                           class_names=['NoDisease','Disease'],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("heart_tree_best")

# 5. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest test acc:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# 6. Feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind='bar')

# 7. Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print("Random Forest CV accuracy: %.3f ± %.3f" % (cv_scores.mean(), cv_scores.std()))

# (Optional) Hyperparameter tuning
param_grid = {
    'n_estimators': [50,100,200],
    'max_depth': [None,4,6,8],
    'max_features': ['sqrt','log2', None]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
best_rf = grid.best_estimator_
print("Best RF test acc:", accuracy_score(y_test, best_rf.predict(X_test)))
