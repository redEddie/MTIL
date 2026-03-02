import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Visualize metrics from CSV logs")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the metrics.csv file")
    parser.add_argument("--x_col", type=str, default="step", help="Column name for the X-axis")
    parser.add_argument("--y_col", type=str, required=True, help="Column name for the Y-axis (e.g., train_loss_step)")
    parser.add_argument("--save_path", type=str, default="plot_result.png", help="Path to save the resulting plot image")
    
    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.csv_path):
        print(f"Error: File not found at {args.csv_path}")
        return

    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Error reading {args.csv_path}: {e}")
        return

    # Check columns
    if args.x_col not in df.columns:
        print(f"Error: X column '{args.x_col}' not found in CSV. Available columns: {list(df.columns)}")
        return
    if args.y_col not in df.columns:
        print(f"Error: Y column '{args.y_col}' not found in CSV. Available columns: {list(df.columns)}")
        return

    # Drop NaNs (empty values) for the relevant columns so plot doesn't break
    plot_df = df[[args.x_col, args.y_col]].dropna()

    if plot_df.empty:
        print(f"Error: No valid data found after dropping NaNs for columns '{args.x_col}' and '{args.y_col}'.")
        return

    # Sort to ensure line plots chronologically
    plot_df = plot_df.sort_values(by=args.x_col)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(plot_df[args.x_col], plot_df[args.y_col], marker='o', markersize=2, linestyle='-', alpha=0.7)
    
    plt.title(f"{args.y_col} over {args.x_col}")
    plt.xlabel(args.x_col)
    plt.ylabel(args.y_col)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300)
    print(f"Plot saved successfully to {args.save_path}")

if __name__ == "__main__":
    main()
