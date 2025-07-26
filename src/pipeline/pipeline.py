import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Receives raw dataframe of lines extracted from PDF,
    preprocesses it to create the features needed for prediction.
    """

    df = df.copy()

    # Convert 'is_italic' to bool -> int
    def to_bool(x):
        x_str = str(x).strip().lower()
        if x_str in ['true', '1', 'yes']:
            return True
        elif x_str in ['false', '0', 'no']:
            return False
        return False

    if 'is_italic' in df.columns:
        df['is_italic'] = df['is_italic'].apply(to_bool).astype(int)

    # Convert 'is_bold' to int
    if 'is_bold' in df.columns:
        df['is_bold'] = df['is_bold'].astype(int)

    # Label encode 'font'
    if 'font' in df.columns:
        le_font = LabelEncoder()
        df['font_encoded'] = le_font.fit_transform(df['font'])
        df.drop(columns=['font'], inplace=True)

    # Label encode 'color'
    if 'color' in df.columns:
        le_color = LabelEncoder()
        df['color_encoded'] = le_color.fit_transform(df['color'])
        df.drop(columns=['color'], inplace=True)

    # Drop other non-feature text columns if they exist
    for col in ['text', 'block_number', 'line_number', 'span_number', 'page_number']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    print("df PIPELINE:")
    print(df.head())
    return df
