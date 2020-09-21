from featurestore import feature_handling

FILENAME = 'test-data.csv'
FILEPATH = 'tests/data'

df = feature_handling.load_csv(FILEPATH, FILENAME)
print(df.head())

