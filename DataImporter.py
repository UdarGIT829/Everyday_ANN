import pandas as pd

from FeatureEncoder import FeatureEncoder

class DataImporter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.data.dropna(inplace=True)  # Drop rows with NaN values
        self.feature_encoders = {}
        self.input_features = None
        self.output_feature = None
        self.prepare_data()
    
    def prepare_data(self):
        columns = self.data.columns
        input_columns = columns[:-1]
        output_column = columns[-1]
        
        # Encode input features
        encoded_inputs = []
        for column in input_columns:
            encoder = FeatureEncoder(column, self.data[column])
            self.feature_encoders[column] = encoder
            encoded_inputs.append(encoder.encoded_data)
        
        # Combine encoded input features into a DataFrame
        self.input_features = pd.concat(encoded_inputs, axis=1)
        
        # Encode output feature
        output_encoder = FeatureEncoder(output_column, self.data[output_column])
        self.feature_encoders[output_column] = output_encoder
        self.output_feature = output_encoder.encoded_data
    
    def get_encoded_data(self):
        return self.input_features, self.output_feature