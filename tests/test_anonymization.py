import copy

import pandas as pd
import pytest
from zooniverse.utils.anonymize import UserAnonymizer

@pytest.fixture(scope="module")
def sample_data():
    """Fixture to provide sample data for testing."""
    data = {
        "flight_site_code": [None, None, "BahiaNegra"],
        "image_name": [None, None, "MBN05-2_221.jpg"],
        "subject_id": [58, 78926950, 78964570],
        "x": [937.0723266601562, 557.1560668945312, 251.48855590820312],
        "y": [58.004669189453125, 920.5582885742188, 82.25997924804688],
        "tool_label": ["Adult Male not in a lek", "Adult Male not in a lek", "Adult Male with a lek"],
        "phase_tag": ["Iguanas 3rd launch", "Iguanas 3rd launch", "Iguanas 3rd launch"],
        "user_id": [1983945, 1983945, None],
        "user_name": ["ANDREAVARELA89", "ANDREAVARELA89", "not-logged-in-74c5dfdab0b0efe3c382"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def anonymizer(sample_data):
    """Fixture to create a DataAnonymizer instance with the sample data loaded."""
    anonymizer = UserAnonymizer(None)  # Path is None since we're directly assigning DataFrame
    anonymizer.df = copy.deepcopy(sample_data)
    return anonymizer

def test_user_id_anonymization(anonymizer, sample_data):
    """Test that user_id is anonymized and consistent across the DataFrame."""
    anonymizer.anonymize_data()
    anonymized_ids = anonymizer.df['user_id'].unique()
    assert len(anonymized_ids) == len(anonymizer.df['user_id'].unique()), "Anonymized user IDs should have the same unique count as original."
    for user_id in anonymized_ids:
        assert user_id not in sample_data['user_id'].values, "Anonymized user ID should not be present in the original user IDs."

def test_user_name_anonymization(anonymizer, sample_data):
    """Test that user_name is anonymized and consistent across the DataFrame."""
    anonymizer.anonymize_data()
    anonymized_names = anonymizer.df['user_name'].unique()
    assert len(sample_data.user_name.unique()) == len(anonymized_names), "Anonymized user names should have the same unique count as original."
    for user_name in anonymized_names:
        assert user_name not in sample_data['user_name'].values, "Anonymized user name should not be present in the original user names."

def test_data_integrity_post_anonymization(anonymizer, sample_data):
    """Test that other data columns remain unchanged after anonymization."""
    original_df_without_users = anonymizer.df.drop(columns=['user_id', 'user_name'])
    anonymizer.anonymize_data()
    anonymized_df_without_users = anonymizer.df.drop(columns=['user_id', 'user_name'])
    pd.testing.assert_frame_equal(original_df_without_users, anonymized_df_without_users, check_like=True)
