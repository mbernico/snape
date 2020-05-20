
from snape.make_dataset import make_dataset
from snape.score_dataset import score_dataset

# configuration json examples can be found in doc
conf = {
    "type": "classification",
    "n_classes": 2,
    "n_samples": 1000,
    "n_features": 10,
    "out_path": "./",
    "output": "my_dataset",
    "n_informative": 3,
    "n_duplicate": 0,
    "n_redundant": 0,
    "n_clusters": 2,
    "weights": [0.8, 0.2],
    "pct_missing": 0.00,
    "insert_dollar": "Yes",
    "insert_percent": "Yes",
    "star_schema": "No",
    "label_list": [],
    "random_seed": 42
}

# make_dataset creates an artificial dataset using the passed dictionary
make_dataset(config=conf)

# a dataset's testkey can be compared to a prediction file using score_dataset()
results = score_dataset(y_file="student_testkey.csv", y_hat_file="student_predictions.csv")
# results is a tuple of (a_primary_metric, classification_report)
print("AUC = " + str(results[0]))
print(results[1])
