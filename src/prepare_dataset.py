import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def main():
    # Load IRIS dataset
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Add species column (target)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Add random location attribute (0 or 1)
    np.random.seed(42)  # For reproducibility
    df['location'] = np.random.choice([0, 1], size=len(df))

    # Save to CSV in data directory
    df.to_csv('data/iris_with_location.csv', index=False)
    print("Saved modified IRIS dataset to data/iris_with_location.csv")

if __name__ == "__main__":
    main()
