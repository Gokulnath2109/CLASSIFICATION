from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

flower_data = load_iris()

features = flower_data.data
labels = flower_data.target

train_x, test_x, train_y, test_y = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(train_x, train_y)

pred_y = model.predict(test_x)

print("Accuracy:", accuracy_score(test_y, pred_y))

print("\nConfusion Matrix:")
print(confusion_matrix(test_y, pred_y))

sample = [[5.1, 3.5, 1.4, 0.2]]
result = model.predict(sample)

print("\nPredicted class:", result[0])
print("Flower type:", flower_data.target_names[result[0]])