import pandas as pd

class FeatureEncoder:
    def __init__(self, column_name, column_data):
        self.column_name = column_name
        self.column_data = column_data
        self.encoded_data = None
        self.dtype = None
        self.mapping = None
        
        self.detect_dtype_and_encode()
    
    def detect_dtype_and_encode(self):
        try:
            self.encoded_data = self.column_data.astype(int)
            self.dtype = 'int'
        except ValueError:
            try:
                self.encoded_data = self.column_data.astype(float)
                self.dtype = 'float'
            except ValueError:
                try:
                    self.encode_strings()
                    self.dtype = 'string'
                except Exception as e:
                    print(f"Failed to encode column '{self.column_name}': {e}")
                    self.encoded_data = self.column_data
                    self.dtype = 'unknown'
    
    def encode_strings(self):
        try:
            unique_values = self.column_data.unique()
            self.mapping = {value: idx for idx, value in enumerate(unique_values)}
            self.encoded_data = self.column_data.map(self.mapping)
        except Exception as e:
            print(f"Error encoding strings in column '{self.column_name}': {e}")
            raise

    def decode(self, encoded_value):
        try:
            if self.dtype == 'string':
                reverse_mapping = {v: k for k, v in self.mapping.items()}
                return reverse_mapping.get(encoded_value, None)
            return encoded_value
        except Exception as e:
            print(f"Error decoding value '{encoded_value}' in column '{self.column_name}': {e}")
            return None
