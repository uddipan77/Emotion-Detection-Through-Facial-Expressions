import pandas as pd
import numpy as np
from collections import Counter

# Load the dataset
df = pd.read_csv("data/fer2013.csv")

# Count samples per class
class_counts = Counter(df['emotion'])
print("Class Distribution:", class_counts)

total_samples = sum(class_counts.values())
#class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

class_weights = {cls: np.log(total_samples / count) for cls, count in class_counts.items()}

print("Class Weights:", class_weights)
