import pandas as pd
import random

def processing(car_data, save_path, save_data=False):
    """
    Process the car data by filling missing values in the 'type' and 'seats' features
    with the most common values for each brand and model combination.

    Parameters:
    - car_data (pd.DataFrame): Input car data.
    - save_path (str): File path to save the processed data.
    - save_data (bool): Flag to indicate whether to save the processed data to a file.

    Returns:
    None
    """

    #Process to fill type, seats features
    type_of_car = car_data[['brand', 'model', 'type', 'seats']]
    most_common_values = type_of_car.groupby(['brand', 'model'])[['type', 'seats']].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.Series({'type': None, 'seats': None}))
    most_common_values = most_common_values.reset_index()

    merged_df = pd.merge(car_data, most_common_values[['brand', 'model', 'type', 'seats']], on=['brand', 'model'], how='left')
    merged_df = merged_df.drop(columns=['type_x', 'seats_x'])
    merged_df.rename(columns={'type_y':'type', 'seats_y':'seats'}, inplace=True)

    new_car_data = merged_df.drop(columns=['id', 'list_id', 'list_time', 'origin', 'color'])
    print(new_car_data.isna().sum())

    #Some basic process
    new_car_data.dropna(inplace=True)
    new_car_data['price'] = new_car_data['price'] / 1e6
    new_car_data['seats'] = new_car_data['seats'].astype(int)

    print(new_car_data.head())

    if save_data:
        new_car_data.to_csv(save_path)

def processing_2(car_data, save_path, save_data=False):
    """
    Process the car data by filling missing values in the 'type' and 'seats' features
    with the most common values for each brand and model combination. Randomly fill
    missing values in the 'color' feature if needed.

    Parameters:
    - car_data (pd.DataFrame): Input car data.
    - save_path (str): File path to save the processed data.
    - save_data (bool): Flag to indicate whether to save the processed data to a file.

    Returns:
    None
    """

    #Process to fill type, seats features
    type_of_car = car_data[['brand', 'model', 'type', 'seats']]
    most_common_values = type_of_car.groupby(['brand', 'model'])[['type', 'seats']].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.Series({'type': None, 'seats': None}))
    most_common_values = most_common_values.reset_index()

    merged_df = pd.merge(car_data, most_common_values[['brand', 'model', 'type', 'seats']], on=['brand', 'model'], how='left')
    merged_df = merged_df.drop(columns=['type_x', 'seats_x'])
    merged_df.rename(columns={'type_y':'type', 'seats_y':'seats'}, inplace=True)

    new_car_data = car_data.drop(columns=['id', 'list_id', 'list_time', 'origin'])
    print(new_car_data.isna().sum())

    new_car_data['color'].fillna(random.choice(new_car_data['color'].unique()), inplace=True)

    #Some basic process
    new_car_data.dropna(inplace=True)
    new_car_data['price'] = new_car_data['price'] / 1e6
    new_car_data['seats'] = new_car_data['seats'].astype(int)

    print(new_car_data.head())

    if save_data:
        new_car_data.to_csv(save_path)

def processing_3(car_data, save_path, save_data=False):
    """
    Process the car data by dropping unnecessary columns and handling missing values.
    Randomly fill missing values in the 'color' feature if needed.

    Parameters:
    - car_data (pd.DataFrame): Input car data.
    - save_path (str): File path to save the processed data.
    - save_data (bool): Flag to indicate whether to save the processed data to a file.

    Returns:
    None
    """
    
    new_car_data = car_data.drop(columns=['id', 'list_id', 'list_time'])
    print(new_car_data.isna().sum())

    new_car_data['color'].fillna(random.choice(new_car_data['color'].unique()), inplace=True)

    #Some basic process
    new_car_data.dropna(inplace=True)
    new_car_data['price'] = new_car_data['price'] / 1e6
    new_car_data['seats'] = new_car_data['seats'].astype(int)

    print(new_car_data.head())

    if save_data:
        new_car_data.to_csv(save_path)

if __name__ == '__main__':
    car_data = pd.read_csv('car.csv')
    processing(car_data, 'new_car.csv', True)
    processing_2(car_data, 'new_car_2.csv', True)
    processing_3(car_data, 'new_car_3.csv', True)
