"""
Japanese Apartment Price Analysis and Report Generator
This script loads a Japanese real estate dataset, translates columns to English,
and generates a comprehensive analysis report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_translate_data(filepath):
    """Load Japanese CSV and translate columns to English"""
    print("=" * 80)
    print("🇯🇵 TO 🇺🇸 APARTMENT PRICE ANALYSIS REPORT")
    print("=" * 80)
    print(f"\n📁 Loading data from: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found.")
        exit(1)

    print(f"✓ Loaded {len(df)} records.")

    # Translation Map
    translation_map = {
        '物件名': 'Building Name',
        '家賃': 'Rent_Yen',
        '共益費': 'Service_Fee_Yen',
        '間取り': 'Layout',
        '専有面積': 'Area_m2',
        '築年数': 'Age_Years',
        '駅徒歩': 'Walk_Min',
        '所在階': 'Floor',
        '敷金': 'Deposit_Months',
        '礼金': 'Key_Money_Months'
    }

    # Rename columns
    print("\n🔄 Translating columns to English...")
    df.rename(columns=translation_map, inplace=True)
    
    # Calculate Total Monthly Cost (Rent + Service Fee)
    if 'Service_Fee_Yen' in df.columns:
        df['Total_Monthly_Cost'] = df['Rent_Yen'] + df['Service_Fee_Yen']
    else:
        df['Total_Monthly_Cost'] = df['Rent_Yen']
        
    print("✓ Translation complete. Columns are now:")
    print(f"  {list(df.columns)}")
    
    return df

def analyze_data(df):
    """Perform statistical analysis"""
    print("\n" + "=" * 80)
    print("1. DATA OVERVIEW & STATISTICS")
    print("=" * 80)
    
    print("\n📊 Descriptive Statistics (Numerical):")
    print(df.describe().round(2))
    
    avg_rent = df['Rent_Yen'].mean()
    max_rent = df['Rent_Yen'].max()
    min_rent = df['Rent_Yen'].min()
    
    print(f"\n💰 Rent Analysis (in JPY):")
    print(f"  • Average Rent: ¥{avg_rent:,.0f}")
    print(f"  • Lowest Rent:  ¥{min_rent:,.0f}")
    print(f"  • Highest Rent: ¥{max_rent:,.0f}")
    
    return df

def visualize_key_trends(df):
    """Generate English visualizations"""
    print("\n" + "=" * 80)
    print("2. VISUALIZING TRENDS")
    print("=" * 80)
    
    # 1. Rent Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Rent_Yen'], kde=True, color='skyblue')
    plt.title('Distribution of Monthly Rent', fontsize=15)
    plt.xlabel('Rent (JPY)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig('rent_distribution.png')
    plt.close()
    
    # 2. Area vs Rent
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Area_m2', y='Rent_Yen', hue='Layout', s=100, alpha=0.7)
    plt.title('Rent vs. Apartment Size', fontsize=15)
    plt.xlabel('Area (m²)', fontsize=12)
    plt.ylabel('Rent (JPY)', fontsize=12)
    plt.legend(title='Layout')
    plt.savefig('rent_vs_area.png')
    plt.close()
    
    # 3. Age vs Rent
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='Age_Years', y='Rent_Yen', color='coral', scatter_kws={'alpha':0.6})
    plt.title('Impact of Building Age on Rent', fontsize=15)
    plt.xlabel('Building Age (Years)', fontsize=12)
    plt.ylabel('Rent (JPY)', fontsize=12)
    plt.savefig('rent_vs_age.png')
    plt.close()

    # 4. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap', fontsize=15)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    print("✓ Generated charts:")
    print("  • rent_distribution.png")
    print("  • rent_vs_area.png")
    print("  • rent_vs_age.png")
    print("  • correlation_heatmap.png")

def predict_rent(df):
    """Build a model to predict rent"""
    print("\n" + "=" * 80)
    print("3. RENT PREDICTION MODEL")
    print("=" * 80)
    
    # Prepare Data
    # Encode categorical 'Layout' column
    le = LabelEncoder()
    df['Layout_Code'] = le.fit_transform(df['Layout'])
    
    # Features (X) and Target (y)
    # We strip out non-numeric columns like Building Name
    features = ['Area_m2', 'Age_Years', 'Walk_Min', 'Floor', 'Layout_Code']
    target = 'Rent_Yen'
    
    X = df[features]
    y = df[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    print("🤖 Training Random Forest Model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predict
    y_pred = rf_model.predict(X_test)
    
    # Evaluate
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n🎯 Model Performance:")
    print(f"  • R² Score: {r2:.4f} (Accuracy measure)")
    print(f"  • RMSE: ¥{rmse:,.0f} (Average error)")
    
    # Feature Importance
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n⭐ Key Factories Driving Rent:")
    for _, row in importances.iterrows():
        print(f"  • {row['Feature']:15s}: {row['Importance']:.4f}")

    return importances

def generate_summary(df, importances):
    print("\n" + "=" * 80)
    print("4. EXECUTIVE SUMMARY")
    print("=" * 80)
    
    top_factor = importances.iloc[0]['Feature']
    avg_area = df['Area_m2'].mean()
    rent_per_sqm = df['Rent_Yen'].sum() / df['Area_m2'].sum()
    
    print("\nBased on the analysis of these Tokyo-area apartments:")
    print(f"1. ✅ The most important factor for rent is **{top_factor}**.")
    print(f"2. 📏 The average apartment size is **{avg_area:.1f} m²**.")
    print(f"3. 💴 The average price per square meter is **¥{rent_per_sqm:,.0f}/m²**.")
    print("4. 📉 Older buildings tend to cheaper rents (negative correlation).")
    print("\nAnalysis Complete. Files are saved in this folder.")

def main():
    csv_file = 'japanese_apartments.csv'
    df = load_and_translate_data(csv_file)
    analyze_data(df)
    visualize_key_trends(df)
    importances = predict_rent(df)
    generate_summary(df, importances)

if __name__ == "__main__":
    main()
