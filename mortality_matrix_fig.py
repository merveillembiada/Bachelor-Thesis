import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_mortality_heatmap(M: pd.DataFrame):
    """
    Generates a heatmap of the log-mortality matrix ln(m_{x,t}) using Matplotlib,
    optimized to render cleanly inside Spyder's inline Plots pane.
    """
    print("-> Rendering log-mortality heatmap...")
    
    # Transform the raw central death rates into log-mortality values
    log_M = np.log(M)
    
    # Configure the dimensions of the figure to match your thesis formatting
    fig, ax = plt.subplots(figsize=(13, 6))
    
    # Create the heatmap grid using imshow
    # 'origin=lower' ensures Age 0 starts at the bottom, matching demographic standards
    cax = ax.imshow(log_M, cmap='plasma', origin='lower', aspect='auto',
                    extent=[log_M.columns.min(), log_M.columns.max(), log_M.index.min(), log_M.index.max()])
    
    # Add the colorbar scale on the right and label it
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('$\ln(m_{x,t})$', fontsize=10)
    
    # Add the structural policy benchmark lines specified in Section 3.2.2 [cite: 56, 278]
    ax.axhline(y=65, color='white', linestyle='--', linewidth=1.5, label='Age 65 (retirement)')
    ax.axhline(y=67, color='yellow', linestyle=':', linewidth=1.5, label='Age 67 (post-2012 reform)')
    
    # Set the precise title and subtitles matching Figure 1 [cite: 54, 276]
    plt.title(
        "Figure 1: Mortality Matrix — Heatmap of $\ln(m_{x,t})$, Germany 1960-2022\n"
        "(Lighter colours = lower mortality; secular improvement visible left$\rightarrow$right at all ages)",
        fontsize=11, 
        fontweight='bold', 
        pad=15
    )
    
    # Label the mathematical axes
    plt.xlabel("Calendar Year $t$", fontsize=10)
    plt.ylabel("Age $x$", fontsize=10)
    
    # Configure explicit tick markers on the axes every 10 units for clean scaling
    ax.set_xticks(np.arange(log_M.columns.min(), log_M.columns.max() + 1, 10))
    ax.set_yticks(np.arange(log_M.index.min(), log_M.index.max() + 1, 10))
    
    # Place the legend in the top-left background area
    plt.legend(loc='upper left', framealpha=0.6)
    
    # Clean layout boundaries
    plt.tight_layout()
    
    # Command for Spyder to display the figure directly inside the Plots tab
    plt.show()