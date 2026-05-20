import pandas as pd
import numpy as np

def build_mortality_matrix(filepath: str, year_range: tuple = (1960, 2020), max_age: int = 89) -> pd.DataFrame:
    """
    Loads pre-averaged annual period life tables for Germany, filters by 
    the specified historical timeline and age brackets, and constructs 
    the 2D mortality matrix M.
    """
    # Load the historical life table data from the 'DEU' Excel sheet
    df = pd.read_excel(filepath, sheet_name='DEU')
    
    # Filter the dataset to include only your target calendar years and age cohorts
    # Limiting age to 89 reduces extreme old-age volatility caused by small sample sizes
    mask = (
        (df['Year1'] >= year_range[0]) & 
        (df['Year1'] <= year_range[1]) & 
        (df['Age'] <= max_age)
    )
    df_filtered = df[mask]
    
    # Pivot the clean data directly into a structured, two-dimensional matrix M
    # Rows (index) correspond to Age (0 to max_age); Columns correspond to Year1 (1960 to 2022)
    M = df_filtered.pivot(index='Age', columns='Year1', values='mx')
    
    # Apply simple linear interpolation horizontally across columns (axis=1, the time dimension)
    # 'limit_direction="both"' ensures that missing values at the boundaries (start/end years) 
    # are filled using the closest available year's trend (backward/forward fill fallback)
    M = M.interpolate(method='linear', axis=1, limit_direction='both')
    
    return M

# --- Local Execution Block ---

# Name of your Excel workbook matching your directory configuration
chemin_excel = "Germany_LifeTables.xlsx"

try:
    # Build the core historical matrix M for your Lee-Carter model estimation
    matrice_M = build_mortality_matrix(filepath=chemin_excel, year_range=(1960, 2022), max_age=89)
    
    # Log successful execution parameters and output structural dimensions
    print("The mortality matrix was successfully created !")
    print(f"Matrice dimensions (Âges x Années) : {matrice_M.shape}")
    print("\nOverview of the first rows and columns :")
    print(matrice_M.head())
    
except Exception as e:
    # Fallback to display file management or structural database errors
    print(f"There was an error during the execution : {e}")