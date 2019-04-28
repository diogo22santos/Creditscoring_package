import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import pipeline

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

TESTING_DATA_FILE = DATASET_DIR / 'test.csv'
TRAINING_DATA_FILE = DATASET_DIR / 'loans_data.csv'
TARGET = 'Defaulted'

FEATURES = ['Amount','City','Country','DateOfBirth','DebtToIncome','Education',
'EmploymentPosition','ExistingLiabilities','Gender','HomeOwnershipType',
'IncomeFromPrincipalEmployer','Interest rate (APR)','LoanDuration','NewCreditCustomer',
'NoOfPreviousLoansBeforeLoan','VerificationType','PreviousScore']


def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline."""

    save_file_name = 'random_forest_model.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)

    print('saved pipeline')


def run_training() -> None:
    """Train the model."""

    # read training data
    data = pd.read_csv(TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=0.2,
        random_state=0)  # we are setting the seed here

    # applying the pipeline
    pipeline.default_pipe.fit(X_train[FEATURES],
                            y_train)

    save_pipeline(pipeline_to_persist=pipeline.default_pipe)


if __name__ == '__main__':
    run_training()
