import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta

def get_user_inputs():
    print("="*60)
    print("DCA INDEX STRATEGY SIMULATOR")
    print("="*60)
    
    print("\nSTEP 1: Select Index / ETF Symbol")
    print("Common Indices (use '^' prefix for major indices):")
    print("- ^SPG100 : S&P Global 100 (Best 'All World' proxy)")
    print("- ^GSPC   : S&P 500")
    print("- ^NYA    : NYSE Composite")
    print("- ^IXIC   : Nasdaq Composite")
    print("- ^FTW5000: Wilshire 5000")
    print("- VT       : Vanguard Total World Stock ETF (recommended)")
    
    ticker = input("\nEnter Index / ETF Symbol [Default 'VT']: ").strip().upper()
    if not ticker:
        ticker = "VT"

    print("\nSTEP 2: Select Market Episode")
    print("1. The 2008 Great Financial Crisis (Oct 2007 – Dec 2017)")
    print("2. The 2022 Tech/Inflation Crash (Jan 2022 – Today)")
    print("3. The 2000 Dot-Com Bust (Jan 2000 – Dec 2007)")
    print("4. The 2020 COVID Crash (Jan 2020 – Dec 2022)")
    print("5. Custom Start / End Dates")
    
    while True:
        choice = input("\nEnter choice (1–5): ").strip()
        if choice == "1":
            return ticker, "2008_GFC", date(2007, 10, 1), date(2017, 12, 31), False
        elif choice == "2":
            return ticker, "2022_Crash", date(2022, 1, 1), date.today(), False
        elif choice == "3":
            return ticker, "2000_DotCom", date(2000, 1, 1), date(2007, 12, 31), False
        elif choice == "4":
            return ticker, "2020_COVID", date(2020, 1, 1), date(2022, 12, 31), False
        elif choice == "5":
            print("\nCustom dates (YYYY-MM-DD format)")
            start_str = input("Enter start date [e.g. 2000-01-01]: ").strip()
            end_str = input("Enter end date [e.g. 2010-12-31]: ").strip()
            try:
                sim_start = date.fromisoformat(start_str)
                sim_end = date.fromisoformat(end_str)
                return ticker, "CUSTOM", sim_start, sim_end, False
            except ValueError:
                print("Invalid date format. Please try again.")
        else:
            print("Invalid input. Please enter 1–5.")

def find_first_thursday(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != 3:
        d += timedelta(days=1)
    return d

def months_between(start_date: date, end_date: date) -> int:
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

def run_dca_simulation():
    symbol, scenario_name, sim_start, sim_end, _ = get_user_inputs()
    monthly_investment = 10000.0
    
    print(f"\nFetching {symbol} data from {sim_start.strftime('%Y-%m-%d')} to {sim_end.strftime('%Y-%m-%d')}...")
    data = yf.download(symbol, start=sim_start, end=sim_end + timedelta(days=15), 
                      progress=False, auto_adjust=False)
    
    if data.empty:
        print(f"\nERROR: No data retrieved for {symbol}. Please verify the symbol.")
        return

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    
    total_invested = 0.0
    total_shares = 0.0
    history = []
    
    current_month = sim_start.replace(day=1)
    while current_month <= sim_end:
        dca_target = find_first_thursday(current_month.year, current_month.month)
        ts_target = pd.Timestamp(dca_target)
        
        available_dates = data.index[data.index >= ts_target]
        if not available_dates.empty:
            actual_date = available_dates[0]
            price = float(data.loc[actual_date, price_column])
            
            units_bought = monthly_investment / price
            total_shares += units_bought
            total_invested += monthly_investment
            portfolio_value = total_shares * price
            avg_cost = total_invested / total_shares if total_shares > 0 else 0.0
            
            history.append({
                'Date': actual_date,
                'Price': price,
                'Total_Invested': total_invested,
                'Portfolio_Value': portfolio_value,
                'Drawdown_Pct': (portfolio_value - total_invested) / total_invested,
                'Total_Shares': total_shares,
                'Avg_Cost': avg_cost
            })
        
        # Advance to next month
        if current_month.month == 12:
            current_month = date(current_month.year + 1, 1, 1)
        else:
            current_month = date(current_month.year, current_month.month + 1, 1)

    if not history:
        print("No investment records generated.")
        return
    
    df = pd.DataFrame(history)
    
    # Calculate months from start
    start_date = df.iloc[0]['Date'].date()
    df['Months_From_Start'] = df['Date'].apply(lambda d: months_between(start_date, d.date()))
    
    # Key Episodes
    peak_record = df.iloc[0]
    peak_price = peak_record['Price']
    
    bottom_idx = df['Drawdown_Pct'].idxmin()
    bottom_record = df.loc[bottom_idx]
    max_drawdown_pct = bottom_record['Drawdown_Pct'] * 100
    
    # Break-even
    post_bottom_df = df.iloc[bottom_idx:].reset_index(drop=True)
    be_mask = post_bottom_df['Portfolio_Value'] >= post_bottom_df['Total_Invested']
    be_record = post_bottom_df[be_mask].iloc[0] if be_mask.any() else None
    
    # Recovery to initial peak price
    recovery_mask = post_bottom_df['Price'] >= peak_price
    recovery_record = post_bottom_df[recovery_mask].iloc[0] if recovery_mask.any() else None
    
    # Final record
    final_record = df.iloc[-1]
    
    # Lump-sum comparison
    lump_sum_shares = final_record['Total_Invested'] / peak_record['Price']
    lump_sum_value = lump_sum_shares * final_record['Price']
    months_total = final_record['Months_From_Start']
    years = months_total / 12.0 if months_total > 0 else 0
    
    dca_cagr = ((final_record['Portfolio_Value'] / final_record['Total_Invested']) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    lump_cagr = ((lump_sum_value / final_record['Total_Invested']) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    # --- TEXT OUTPUT ---
    print("\n" + "="*95)
    print(f"RESULTS: {symbol} | {scenario_name.replace('_', ' ')}")
    print("="*95)
    
    print(f"\nEPISODE 1: THE PEAK (First Investment)")
    print(f"Date              : {peak_record['Date'].strftime('%Y-%m-%d')}")
    print(f"Price             : ${peak_record['Price']:,.2f}")
    
    print(f"\nEPISODE 2: THE BOTTOM (Maximum Portfolio Drawdown)")
    print(f"Date              : {bottom_record['Date'].strftime('%Y-%m-%d')}")
    print(f"Portfolio Drawdown: {max_drawdown_pct:.1f}%")
    print(f"Total Invested    : ${bottom_record['Total_Invested']:,.0f}")
    print(f"Portfolio Value   : ${bottom_record['Portfolio_Value']:,.0f}")
    print(f"Months from Start : {int(bottom_record['Months_From_Start'])}")
    
    if be_record is not None:
        be_months_from_bottom = int(be_record['Months_From_Start'] - bottom_record['Months_From_Start'])
        print(f"\nEPISODE 3: DCA PORTFOLIO BREAK-EVEN")
        print(f"Date                  : {be_record['Date'].strftime('%Y-%m-%d')}")
        print(f"Total Invested        : ${be_record['Total_Invested']:,.0f}")
        print(f"Portfolio Value       : ${be_record['Portfolio_Value']:,.0f}")
        print(f"Index Level           : ${be_record['Price']:,.2f} (still below peak)")
        print(f"Months from Start     : {int(be_record['Months_From_Start'])}")
        print(f"Months from Bottom    : {be_months_from_bottom}")
        print(f"Average Cost Basis    : ${be_record['Avg_Cost']:,.2f}")
    
    if recovery_record is not None:
        profit = recovery_record['Portfolio_Value'] - recovery_record['Total_Invested']
        months_rec = int(recovery_record['Months_From_Start'])
        years_rec = months_rec / 12.0 if months_rec > 0 else 0
        rec_cagr = ((recovery_record['Portfolio_Value'] / recovery_record['Total_Invested']) ** (1 / years_rec) - 1) * 100 if years_rec > 0 else 0.0
        
        print(f"\nEPISODE 4: INDEX PRICE RECOVERY (Index returns to initial peak level)")
        print(f"Date                  : {recovery_record['Date'].strftime('%Y-%m-%d')}")
        print(f"Index Level           : ${recovery_record['Price']:,.2f}")
        print(f"Total Invested        : ${recovery_record['Total_Invested']:,.0f}   ← 60 months of DCA")
        print(f"Portfolio Value       : ${recovery_record['Portfolio_Value']:,.0f}")
        print(f"Months from Start     : {months_rec}")
        print(f"DCA Surplus Profit    : ${profit:,.0f}")
        print(f"Average Annual Return : {rec_cagr:.2f}%")
        print(f"Average Cost Basis    : ${recovery_record['Avg_Cost']:,.2f} "
              f"({(peak_price - recovery_record['Avg_Cost'])/peak_price*100:.1f}% below peak price)")
        print("   → At this point, a lump-sum investor would be roughly break-even on the index,")
        print("     while the DCA investor has already generated meaningful extra profit.")
    
    # FINAL SUMMARY
    print("\n" + "="*95)
    print("FINAL SUMMARY — Full Simulation Period")
    print("="*95)
    print(f"End Date                   : {sim_end.strftime('%Y-%m-%d')}")
    print(f"Total Months               : {int(final_record['Months_From_Start'])}")
    print(f"Total Capital Invested     : ${final_record['Total_Invested']:,.0f}")
    print(f"Maximum Drawdown           : {max_drawdown_pct:.1f}%")
    print(f"DCA Portfolio Value        : ${final_record['Portfolio_Value']:,.0f}")
    print(f"DCA Compound Annual Return : {dca_cagr:.2f}%")
    print(f"Total Shares Accumulated   : {final_record['Total_Shares']:,.2f}")
    print(f"Average Cost Basis         : ${final_record['Avg_Cost']:,.2f}")
    
    print(f"\nLump-Sum Comparison")
    print(f"(If the entire ${final_record['Total_Invested']:,.0f} had been invested on the first day)")
    print(f"Lump-Sum Portfolio Value   : ${lump_sum_value:,.0f}")
    print(f"Lump-Sum CAGR              : {lump_cagr:.2f}%")
    
    if recovery_record is not None:
        print(f"\nNote: DCA showed a surplus profit of ${profit:,.0f} at the time of index price recovery (Episode 4).")
    
    print("\n" + "="*95)

    # === VISUAL OUTPUT (Matplotlib / Seaborn) ===
    print("\nGenerating visual charts... (saved as PNG files in current directory)")
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Chart 1: Portfolio Value vs Total Invested (with shaded drawdown)
    axes[0].plot(df['Date'], df['Portfolio_Value'], label='Portfolio Value', color='blue', linewidth=2)
    axes[0].plot(df['Date'], df['Total_Invested'], label='Total Invested', color='black', linestyle='--')
    axes[0].fill_between(df['Date'], df['Portfolio_Value'], df['Total_Invested'],
                         where=(df['Portfolio_Value'] < df['Total_Invested']),
                         color='red', alpha=0.3, label='Drawdown Period')
    axes[0].set_title("Portfolio Value vs Total Capital Invested")
    axes[0].set_ylabel("USD")
    axes[0].legend()
    
    # Chart 2: Cumulative Shares Accumulated
    axes[1].plot(df['Date'], df['Total_Shares'], color='green', linewidth=2)
    axes[1].set_title("Cumulative Shares Accumulated")
    axes[1].set_ylabel("Number of Shares")
    
    # Chart 3: Drawdown Percentage Curve
    axes[2].plot(df['Date'], df['Drawdown_Pct'] * 100, color='red', linewidth=2)
    axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[2].set_title("Portfolio Drawdown Percentage (%)")
    axes[2].set_ylabel("Drawdown %")
    axes[2].set_xlabel("Date")
    
    plt.tight_layout()
    plt.savefig("DCA_Simulation_Charts.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Charts saved as 'DCA_Simulation_Charts.png' and displayed.")

if __name__ == "__main__":
    run_dca_simulation()