import pandas as pd

csv_file = ["Testing/01/robot_log.csv", "Testing/08/robot_log.csv", "Testing/14/robot_log.csv", "Testing/21/robot_log.csv", "Testing/26/robot_log.csv", "Testing/43/robot_log.csv"]

for file in csv_file:
    df = pd.read_csv(file, sep=';')
    def convert_path(path):
        path = path.split('\\')[-2:]
        path = file.split('/')[:2] + path
        return '\\'.join(path)
    df['Path'] = df['Path'].apply(convert_path)
    print(df['Path'][0])
    df = df[pd.to_numeric(df['SteerAngle'], errors='coerce').notna()]
    df['SteerAngle'] = df['SteerAngle'].astype(float)

    df.to_csv(file.replace('robot_log.csv', 'processed_robot_log.csv'), index=False)

df = pd.read_csv("Training/robot_log.csv", sep=';')
def convert_path(path):
    path = path.split('\\')[-2:]
    path = "Training/" + '\\'.join(path)
    return path
df['Path'] = df['Path'].apply(convert_path)
print(df['Path'][0])
df.to_csv("Training/processed_robot_log.csv", index=False)