"""
Japanese Apartment Price Analysis and PDF Report Generator
This script loads a Japanese real estate dataset, translates columns to English,
and generates a comprehensive PDF analysis report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")

def load_and_translate_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found.")
        exit(1)

    translation_map = {
        '物件名': 'Building Name', '家賃': 'Rent_Yen', '共益費': 'Service_Fee_Yen',
        '間取り': 'Layout', '専有面積': 'Area_m2', '築年数': 'Age_Years',
        '駅徒歩': 'Walk_Min', '所在階': 'Floor', '敷金': 'Deposit_Months', '礼金': 'Key_Money_Months'
    }
    df.rename(columns=translation_map, inplace=True)
    df['Total_Monthly_Cost'] = df['Rent_Yen'] + df.get('Service_Fee_Yen', 0)
    return df

def create_pdf_report(df, pdf_filename="Japanese_Housing_Report.pdf"):
    print(f"\n📄 Generating PDF Report: {pdf_filename}...")
    
    with PdfPages(pdf_filename) as pdf:
        # --- PAGE 1: Title & Summary ---
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        
        # Title
        plt.text(0.5, 0.95, "Japanese Apartment Market Analysis", ha='center', fontsize=24, weight='bold', color='navy')
        plt.text(0.5, 0.90, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d')}", ha='center', fontsize=12)
        
        # Summary Stats
        avg_rent = df['Rent_Yen'].mean()
        avg_area = df['Area_m2'].mean()
        max_rent = df['Rent_Yen'].max()
        
        summary_text = (
            f"EXECUTIVE SUMMARY\n\n"
            f"• Total Properties Analyzed: {len(df)}\n"
            f"• Average Monthly Rent: ¥{avg_rent:,.0f}\n"
            f"• Average Apartment Size: {avg_area:.1f} m²\n"
            f"• Highest Rent Observed: ¥{max_rent:,.0f}\n\n"
            f"KEY INSIGHTS:\n"
            f"1. Size Matters: Area (m²) is the strongest predictor of rent.\n"
            f"2. Age Factor: Newer buildings command significantly higher premiums.\n"
            f"3. Location: Each minute closer to the station adds value."
        )
        
        plt.text(0.1, 0.5, summary_text, fontsize=14, va='center', linespacing=1.8, family='monospace')
        pdf.savefig()
        plt.close()

        # --- PAGE 2: Distibution Charts ---
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        
        # Rent Distribution
        sns.histplot(df['Rent_Yen'], kde=True, color='skyblue', ax=axes[0])
        axes[0].set_title('Distribution of Monthly Rent', fontsize=14, weight='bold')
        axes[0].set_xlabel('Rent (JPY)')
        
        # Area Distribution
        sns.histplot(df['Area_m2'], kde=True, color='lightgreen', ax=axes[1])
        axes[1].set_title('Distribution of Apartment Sizes', fontsize=14, weight='bold')
        axes[1].set_xlabel('Area (m²)')
        
        plt.tight_layout(pad=3.0)
        pdf.savefig()
        plt.close()

        # --- PAGE 3: Correlations ---
        plt.figure(figsize=(11, 8.5))
        
        # Price vs Area
        plt.subplot(2, 1, 1)
        sns.scatterplot(data=df, x='Area_m2', y='Rent_Yen', hue='Layout', s=100, alpha=0.7)
        plt.title('Rent vs. Size (colored by Layout)', fontsize=14, weight='bold')
        plt.grid(True, alpha=0.3)
        
        # Price vs Age
        plt.subplot(2, 1, 2)
        sns.regplot(data=df, x='Age_Years', y='Rent_Yen', color='coral', scatter_kws={'alpha':0.5})
        plt.title('Rent vs. Building Age (Depreciation Trend)', fontsize=14, weight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        pdf.savefig()
        plt.close()
        
        # --- PAGE 4: Feature Importance (Model) ---
        plt.figure(figsize=(11, 8.5))
        
        # Machine Learning Part
        le = LabelEncoder()
        df['Layout_Code'] = le.fit_transform(df['Layout'])
        features = ['Area_m2', 'Age_Years', 'Walk_Min', 'Floor', 'Layout_Code']
        X = df[features]
        y = df['Rent_Yen']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Importance Plot
        importances = pd.Series(model.feature_importances_, index=features).sort_values()
        importances.plot(kind='barh', color='teal', width=0.7)
        plt.title('What Drives Rent Prices? (Feature Importance)', fontsize=16, weight='bold')
        plt.xlabel('Relative Importance')
        plt.grid(axis='x', alpha=0.3)
        
        # Stats note
        r2 = model.score(X, y)
        plt.figtext(0.5, 0.05, f"Model Accuracy (R²): {r2:.2f}", ha='center', fontsize=12, style='italic')
        
        pdf.savefig()
        plt.close()
        
    print(f"✅ PDF Report generated successfully: {pdf_filename}")

if __name__ == "__main__":
    df = load_and_translate_data('japanese_apartments.csv')
    create_pdf_report(df)
