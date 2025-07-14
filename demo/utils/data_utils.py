import pandas as pd

def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                  'Patient_Age', 'Patient_Gender', 'View_Position',
                  'Original_Image_Width', 'Original_Image_Height',
                  'Original_Image_Pixel_Spacing_X',
                  'Original_Image_Pixel_Spacing_Y', 'dfd']
    df['Finding_Labels'] = df['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])
    label_map = dict(zip(df['Image_Index'], df['Finding_Labels']))
    return label_map

def get_ground_truth(image_name, label_map):
    return label_map.get(image_name, []) 