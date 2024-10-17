import numpy as np
from utils import SMOTE, balance_classes


def test_smote():
    """
    Test the SMOTE function by checking if the generated samples are of the correct shape.
    """
    X_minority = np.array([[1, 2], [3, 4], [5, 6]])
    num_samples = 3

    synthetic_samples = SMOTE(X_minority, num_samples)

    assert synthetic_samples.shape[
               0] == num_samples, f"Expected {num_samples} synthetic samples, got {synthetic_samples.shape[0]}"

    print(f"Test SMOTE: Passed - {num_samples} synthetic samples generated.")
    print(f"Synthetic Samples:\n{synthetic_samples}")


def test_balance_classes():
    """
    Test the balance_classes function by checking if the dataset is balanced after applying SMOTE.
    """

    X_train = np.array([
        [1, 2], [2, 3], [3, 1], [1, 4], [2, 2], [3, 3],
        [4, 2], [5, 3],
        [6, 1],
    ])

    Y_train = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2])

    X_balanced, y_balanced = balance_classes(X_train, Y_train)

    unique, counts = np.unique(y_balanced, return_counts=True)
    max_count = np.max(counts)

    assert np.all(
        counts == max_count), f"Expected all classes to have {max_count} instances, but got {dict(zip(unique, counts))}."

    print(f"Test balance_classes: Passed - Dataset balanced with {max_count} instances per class.")
    print(f"Balanced Features:\n{X_balanced}")
    print(f"Balanced Labels:\n{y_balanced}")


def main():
    """
    Main function to run the tests for SMOTE and balance_classes.
    """
    print("Running SMOTE Test...")
    test_smote()

    print("\nRunning balance_classes Test...")
    test_balance_classes()


if __name__ == "__main__":
    main()
