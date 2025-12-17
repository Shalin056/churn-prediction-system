"""
generate_data.py

Generates synthetic customer churn dataset with STRONG predictive signals.
This creates realistic data that ML models can actually learn from.

Author: [Shalin Bhavsar]
Date: 2025
"""

import numpy as np
import pandas as pd
from faker import Faker
import random
from pathlib import Path

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)


def generate_churn_dataset(n_records: int = 200000, output_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic churn dataset with strong predictive signals.
    
    Parameters
    ----------
    n_records : int
        Number of customer records to generate
    output_path : str, optional
        Path to save the CSV file
    
    Returns
    -------
    pd.DataFrame
        Generated dataset with churn labels
    """
    
    print(f"Generating {n_records:,} customer records...")
    data = []
    
    for i in range(n_records):
        customer_id = f"CUST_{i+1:06d}"
        
        # Demographics
        gender = random.choice(["Male", "Female"])
        age = np.random.randint(18, 80)
        
        # Contract details (MAJOR churn predictor)
        contract = random.choices(
            ["Month-to-month", "One year", "Two year"],
            weights=[0.55, 0.30, 0.15],
        )[0]
        
        # Tenure (correlated with contract type)
        if contract == "Month-to-month":
            tenure = np.random.randint(1, 36)
        elif contract == "One year":
            tenure = np.random.randint(6, 48)
        else:  # Two year
            tenure = np.random.randint(12, 72)
        
        # Payment method
        payment_method = random.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
        )
        
        # Monthly charges (higher = more churn risk)
        monthly_charges = round(np.random.normal(70, 25), 2)
        monthly_charges = max(20, min(monthly_charges, 150))
        total_charges = round(monthly_charges * tenure, 2)
        
        # Support tickets (more tickets = more problems = higher churn)
        support_tickets = np.random.poisson(2)
        
        # Usage score (low engagement = high churn risk)
        base_usage = 50 + (tenure / 72) * 20
        usage_score = round(np.random.normal(base_usage, 15), 2)
        usage_score = max(0, min(usage_score, 100))
        
        # =====================================================================
        # CHURN PROBABILITY CALCULATION (STRONG SIGNALS)
        # =====================================================================
        
        churn_prob = 0.15  # Base rate
        
        # CONTRACT TYPE (Most important - 35% impact)
        if contract == "Month-to-month":
            churn_prob += 0.35
        elif contract == "One year":
            churn_prob += 0.10
        else:  # Two year
            churn_prob -= 0.20
        
        # TENURE (Second most important - 25% impact)
        if tenure < 6:
            churn_prob += 0.25
        elif tenure < 12:
            churn_prob += 0.15
        elif tenure < 24:
            churn_prob += 0.05
        else:
            churn_prob -= 0.15
        
        # MONTHLY CHARGES (20% impact)
        if monthly_charges > 100:
            churn_prob += 0.20
        elif monthly_charges > 80:
            churn_prob += 0.10
        elif monthly_charges < 40:
            churn_prob -= 0.05
        
        # SUPPORT TICKETS (20% impact)
        if support_tickets >= 5:
            churn_prob += 0.20
        elif support_tickets >= 3:
            churn_prob += 0.10
        elif support_tickets == 0:
            churn_prob -= 0.05
        
        # USAGE SCORE (20% impact)
        if usage_score < 30:
            churn_prob += 0.20
        elif usage_score < 50:
            churn_prob += 0.10
        elif usage_score > 70:
            churn_prob -= 0.10
        
        # PAYMENT METHOD (8% impact)
        if payment_method == "Electronic check":
            churn_prob += 0.08
        elif payment_method == "Bank transfer":
            churn_prob -= 0.05
        
        # AGE (8% impact)
        if age < 25:
            churn_prob += 0.08
        elif age > 60:
            churn_prob -= 0.05
        
        # INTERACTION EFFECT: New expensive customer
        if total_charges < 500 and monthly_charges > 80:
            churn_prob += 0.15
        
        # Clip to valid range
        churn_prob = min(max(churn_prob, 0.0), 0.95)
        
        # Generate churn outcome
        churn = np.random.choice([0, 1], p=[1 - churn_prob, churn_prob])
        
        data.append([
            customer_id,
            gender,
            age,
            tenure,
            contract,
            payment_method,
            monthly_charges,
            total_charges,
            support_tickets,
            usage_score,
            churn,
        ])
    
    # Create DataFrame
    columns = [
        "CustomerID",
        "Gender",
        "Age",
        "Tenure",
        "Contract",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "SupportTickets",
        "UsageScore",
        "Churn",
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Dataset saved to: {output_path}")
    
    return df


def print_dataset_summary(df: pd.DataFrame):
    """Print summary statistics of the generated dataset."""
    
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    print(f"\nTotal records: {len(df):,}")
    
    print("\n📊 Churn Distribution:")
    print(df['Churn'].value_counts())
    churn_rate = df['Churn'].mean()
    print(f"Churn Rate: {churn_rate:.1%}")
    
    print("\n📋 Contract Distribution:")
    print(df['Contract'].value_counts())
    
    print("\n⚠️ Churn Rate by Contract:")
    contract_churn = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
    for contract, rate in contract_churn.items():
        print(f"  {contract:20s}: {rate:.1%}")
    
    print("\n📅 Churn Rate by Tenure Groups:")
    df['TenureGroup'] = pd.cut(
        df['Tenure'], 
        bins=[0, 6, 12, 24, 72], 
        labels=['0-6 months', '6-12 months', '12-24 months', '24+ months']
    )
    tenure_churn = df.groupby('TenureGroup')['Churn'].mean()
    for group, rate in tenure_churn.items():
        print(f"  {group:20s}: {rate:.1%}")
    
    print("\n💰 Churn Rate by Monthly Charges:")
    df['ChargeGroup'] = pd.cut(
        df['MonthlyCharges'], 
        bins=[0, 50, 80, 100, 150], 
        labels=['Low ($0-50)', 'Medium ($50-80)', 'High ($80-100)', 'Very High ($100+)']
    )
    charge_churn = df.groupby('ChargeGroup')['Churn'].mean()
    for group, rate in charge_churn.items():
        print(f"  {group:20s}: {rate:.1%}")
    
    print("\n🎫 Churn Rate by Support Tickets:")
    ticket_churn = df.groupby('SupportTickets')['Churn'].mean().head(8)
    for tickets, rate in ticket_churn.items():
        print(f"  {tickets} tickets: {rate:.1%}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Configuration
    N_RECORDS = 200000
    OUTPUT_FILE = Path(__file__).parent / "raw" / "synthetic_churn_data.csv"
    
    # Generate dataset
    df = generate_churn_dataset(n_records=N_RECORDS, output_path=OUTPUT_FILE)
    
    # Print summary
    print_dataset_summary(df)
    
    print(f"\n✅ Data generation complete!")
    print(f"📁 File location: {OUTPUT_FILE}")
    print(f"📊 Shape: {df.shape}")
    print(f"\n💡 Next step: Run the EDA notebook to explore the data")