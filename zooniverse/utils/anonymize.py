import math

import pandas as pd
from hashlib import blake2b

class UserAnonymizer:
    """A class to load, anonymize, and save data from a CSV file."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.load_data()

    def load_data(self):
        """Load the data from a CSV file into a pandas DataFrame."""
        try:
            self.df = pd.read_csv(self.filepath)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Failed to load data: {e}")

    def anonymize_data(self):
        """Anonymize the 'user_id' and 'user_name' columns in the DataFrame."""
        if self.df is not None:
            # Anonymize 'user_id' by hashing
            self.df['user_id'] = self.df['user_id'].apply(lambda x: blake2b(str(x).encode(), digest_size=16).hexdigest() if not pd.isnull(x) else x)

            # Anonymize 'user_name' by hashing
            self.df['user_name'] = self.df['user_name'].apply(lambda x: blake2b(x.encode(), digest_size=16).hexdigest() if isinstance(x, str) else x)
            print("Anonymization completed.")
        else:
            print("Data not loaded. Cannot anonymize.")

    def save_anonymized_data(self, output_filepath):
        """Save the anonymized DataFrame to a new CSV file."""
        if self.df is not None:
            self.df.to_csv(output_filepath, index=False)
            print(f"Anonymized data saved to {output_filepath}")
        else:
            print("Data not loaded. Cannot save.")
