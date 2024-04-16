import numpy as np, matplotlib.pyplot as plt

from scipy.interpolate import griddata


def plot_test(x, y, test, variable_name):
    test = np.asarray(test).flatten()

    plt.figure(figsize=(18, 6))
    plt.scatter(x, y, c=test, cmap='viridis')
    plt.colorbar()
    plt.title(f'{variable_name} Test Data')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.savefig(f"/Users/leondeligny/Desktop/PDM/Plots/{variable_name}.png") 
    plt.show()


# Function to plot predictions, test data, and differences
def plot_predictions_vs_test(x, y, pred, test, variable_name, layers):
    pred = np.asarray(pred).flatten()
    test = np.asarray(test).flatten()

    diff = pred - test

    plt.figure(figsize=(18, 6))
    
    # Plot predictions
    plt.subplot(1, 3, 1)
    plt.scatter(x, y, c=pred, cmap='viridis')
    plt.colorbar()
    plt.title(f'{variable_name} Predictions')
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot test data
    plt.subplot(1, 3, 2)
    plt.scatter(x, y, c=test, cmap='viridis')
    plt.colorbar()
    plt.title(f'{variable_name} Test Data')
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot differences
    plt.subplot(1, 3, 3)
    
    plt.scatter(x, y, c=diff, cmap='viridis')
    plt.colorbar()
    plt.title(f'{variable_name} Prediction - Test')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.savefig(f"/Users/leondeligny/Desktop/PDM/Plots/{variable_name}_diff_{layers}.png") 
    plt.show()

def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size
def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax
def savefig(filename, crop = True):
    if crop == True:
        plt.savefig('{}.pgf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('{}.pgf'.format(filename))
        plt.savefig('{}.pdf'.format(filename))
        plt.savefig('{}.eps'.format(filename))
def plot_solution(X_star, u_star, index):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200

    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)

    X, Y = np.meshgrid(x,y)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')

    plt.figure(index)
    plt.pcolor(X, Y, U_star, cmap = 'jet')
    plt.colorbar()
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4

    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.0),     # default fig size of 0.9 textwidth
    "pgf.preamble": "\n".join([
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
        ]),
    }
