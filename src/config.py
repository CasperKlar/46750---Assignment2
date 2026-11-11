import matplotlib.pyplot as plt


END_OF_DAY_CHARGE = 2 #0.5 * d.battery_capacity_kWh  # Example: 50% reserve

#plot formatting
def plot_fonts():
    "plot parameters"
    plt.rcParams.update({
        'font.size': 14,          # General text
        'axes.titlesize': 20,     # Plot title
        'axes.labelsize': 16,     # Axis labels
        'xtick.labelsize': 16,    # X-axis tick labels
        'ytick.labelsize': 16,    # Y-axis tick labels
        'legend.fontsize': 13     # Legend
        })

MODEL_R_COLOR = "blue"
MODEL_I_COLOR = "orange"
MODEL_C_COLOR = "green"

#Q1 sensititvity specifics
SENSITIVITY_Q1 = False
SENSITIVITY_Q1_BETA = "Q1_S_beta_0.75" #this is only for naming results, manually changed in JSON file 
SENSITIVITY_Q1_BETA_PAIR = ("Q1_S_beta_0.75", "Q1_S_beta_1.0")   # (a, b) -> plot shows b - a
