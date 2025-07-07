import pandas as pd
import numpy as np
from utils.Logger import CustomLogger
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif

class DataBase():
    def __init__(self, logger: CustomLogger=None):
        self.train_df = None
        self.test_df = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.le = None
        self.feature_names = None
        self.sorted_features = None ## Feature ranking based on Mutual Information
        self.logging = logger

    def load_data(self):
        # Data loading
        self.train_df = pd.read_csv('/app/data/hibrid_dataset_GOOSE_train.csv', sep=',')
        self.test_df = pd.read_csv('/app/data/hibrid_dataset_GOOSE_test.csv', sep=',')
        self.logging.info(f"Original dataset (Train): \n{self.train_df.head().to_string()}")
        self.logging.info(f"Original dataset (Test): \n{self.test_df.head().to_string()}")
        self.logging.info(f"Unique classes in the test dataset: {self.test_df['class'].unique()}")
        self.logging.info(f"Unique classes in the training dataset: {self.train_df['class'].unique()}")

        # Remove specific attacks from the training set
        # self.train_df = self.train_df[self.train_df['class'] != 'random_replay']
        # self.train_df = self.train_df[self.train_df['class'] != 'inverse_replay']
        # self.train_df = self.train_df[self.train_df['class'] != 'masquerade_fake_fault']
        # self.train_df = self.train_df[self.train_df['class'] != 'masquerade_fake_normal']
        # self.train_df = self.train_df[self.train_df['class'] != 'injection']
        # self.train_df = self.train_df[self.train_df['class'] != 'high_StNum']
        # self.train_df = self.train_df[self.train_df['class'] != 'poisoned_high_rate']
        # self.logging.info(f"Remaining unique classes in the training dataset: {self.train_df['class'].unique()}")
        # self.logging.info(f"Size of the training dataset after filtering: {len(self.train_df)}")

        # Remove specific attacks from the test set
        # self.test_df = self.test_df[self.test_df['class'] != 'random_replay']
        # self.test_df = self.test_df[self.test_df['class'] != 'inverse_replay']
        # self.test_df = self.test_df[self.test_df['class'] != 'masquerade_fake_fault']
        # self.test_df = self.test_df[self.test_df['class'] != 'masquerade_fake_normal']
        # self.test_df = self.test_df[self.test_df['class'] != 'injection']
        # self.test_df = self.test_df[self.test_df['class'] != 'high_StNum']
        # self.test_df = self.test_df[self.test_df['class'] != 'poisoned_high_rate']
        # self.logging.info(f"Remaining unique classes in the test dataset: {self.test_df['class'].unique()}")
        # self.logging.info(f"Size of the test dataset after filtering: {len(self.test_df)}")

        self.train_df = self.train_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)

        # Enriched columns from the ERENO dataset to be removed, if necessary
        columns_to_remove = [
            "isbARmsValue", "isbBRmsValue", "iisbCRmsValue", "ismARmsValue", "ismBRmsValue", "ismCRmsValue",
            "vsbARmsue", "vsbBRmsValue", "vsbCRmsValue", "vsmARmsValue", "vsmBRmsValue", "vsmCRmsValue",
            "isbATrapAreaSum", "isbBTrapAreaSum", "isbCTrapAreaSum", "ismATrapAreaSuValm", "ismBTrapAreaSum",
            "ismCTrapAreaSum", "vsbATrapAreaSum", "vsbBTrapAreaSum", "vsbCTrapAreaSum", "vsmATrapAreaSum",
            "vsmBTrapAreaSum", "vsmCTrapAreaSum", "stDiff", "sqDiff", "gooseLengthDiff", "cbStatusDiff",
            "apduSizeDiff", "frameLengthDiff", "timestampDiff", "tDiff", "timeFromLastChange", "delay"
        ]

        initial_features_train = self.train_df.shape[1] - 1  # Subtracting 1 to exclude the 'class' column
        initial_features_test = self.test_df.shape[1] - 1  # Subtracting 1 to exclude the 'class' column

        self.logging.info(f"Initial number of features in the training dataset: {initial_features_train}")
        self.logging.info(f"Initial number of features in the test dataset: {initial_features_test}")
        # Removing enriched and NaN columns (Uncomment the next 6 lines)
        # self.train_df = self.train_df.drop(columns=columns_to_remove, errors='ignore')
        # self.test_df = self.test_df.drop(columns=columns_to_remove, errors='ignore')
        #
        # remove_features_train = self.train_df.shape[1] - 1  # Subtracting 1 to exclude the 'class' column
        # remove_features_test = self.test_df.shape[1] - 1  # Subtracting 1 to exclude the 'class' column
        # self.logging.info(f"Number of features in the training dataset after removing enriched ones: {remove_features_train}")
        # self.logging.info(f"Number of features in the test dataset after removing enriched ones:: {remove_features_test}")

        # Splitting features and labels
        self.x_train = self.train_df.drop(columns=['class'])
        self.y_train = self.train_df['class']
        self.x_test = self.test_df.drop(columns=['class'])
        self.y_test = self.test_df['class']

    def preprocess_data(self):
        # Identify numerical columns
        num_cols = self.x_train.select_dtypes(include=[np.number]).columns
        cat_cols = self.x_train.select_dtypes(include=['object']).columns

        # Use StandardScaler to normalize numerical data
        scaler = StandardScaler()
        self.x_train[num_cols] = scaler.fit_transform(self.x_train[num_cols])
        self.x_test[num_cols] = scaler.transform(self.x_test[num_cols])

        # Use OneHotEncoder for categorical columns
        if len(cat_cols) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            cat_encoded = encoder.fit_transform(self.x_train[cat_cols])
            cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))

            # Drop original categorical columns and concat encoded columns
            self.x_train = pd.concat([self.x_train[num_cols], cat_encoded_df], axis=1)
            cat_encoded_test = encoder.transform(self.x_test[cat_cols])
            cat_encoded_test_df = pd.DataFrame(cat_encoded_test, columns=encoder.get_feature_names_out(cat_cols))
            self.x_test = pd.concat([self.x_test[num_cols], cat_encoded_test_df], axis=1)

        # Initialize LabelEncoder for labels/classes
        self.le = LabelEncoder()

        self.le.fit(self.y_train)

        self.y_train = self.le.transform(self.y_train)
        self.y_test = self.le.transform(self.y_test) # Direct transformation of test labels
        self.feature_names = self.x_train.columns.tolist()

    def get_data(self):
        if self.x_train is None or self.y_train is None or self.x_test is None or self.y_test is None:
            raise ValueError("Data not loaded. Please call load_data() before get_data().")
        return self.x_train, self.y_train, self.x_test, self.y_test, self.le

    def load_and_preprocess(self):
        self.load_data()
        self.preprocess_data()
        self.logging.info("Preprocessing completed successfully.")

    def rank_features(self):
        self.logging.info("Ranking Features using Mutual Information for composing RCL.")
        # Mutual Information (MI) measures the mutual dependence between two random variables.
        # In the context of feature selection, it evaluates how much information about the label
        # is provided by a particular feature.
        ig_scores = mutual_info_classif(self.x_train, self.y_train, random_state=42)
        self.logging.info("Feature ranking completed.")
        self.sorted_features = sorted(zip(self.feature_names, ig_scores), key=lambda x: x[1], reverse=True)

    def print_feature_scores(self):
        self.logging.info("\nMutua Information for Features:")
        if self.sorted_features is None:
            self.logging.info("No feature scores to print. Run rank_features() first.")
            return
        for feature, score in self.sorted_features:
            self.logging.info(f"Feature {feature}: MI = {score:.4f}")