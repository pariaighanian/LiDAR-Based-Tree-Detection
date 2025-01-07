import numpy as np
from laspy import read
from pandas import DataFrame

def read_lidar_to_df(filepath):
    """
    Reads a LAS file and converts it to a pandas DataFrame with essential fields.

    Parameters:
        filepath (str): Path to the LAS file.
    
    Returns:
        pd.DataFrame: DataFrame containing LiDAR data with selected features.
    """
    las = read(filepath)
    
    lidar_points = np.array((
        las.x,                       
        las.y,                       
        las.z,                       
        las.intensity,              
        las.classification,          
        las.return_number,           
        las.number_of_returns        
    )).transpose()

    columns = ['x', 'y', 'z', 'intensity', 'classification', 'return_number', 'number_of_returns']
    
    lidar_df = DataFrame(lidar_points, columns=columns)
    
    # Filtering for points classified as High Vegetation (classification == 5)
    lidar_df = lidar_df[lidar_df['classification'] == 5].reset_index(drop=True)
    
    return lidar_df

