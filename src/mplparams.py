import matplotlib as mpl
import matplotlib.pyplot as plt
def set_matplotlib_params():
    font = {'family' : 'serif',
            'size'   : 10,
            'serif':  'cmr10'
            }
    mpl.rc('font', **font)
    plt.rcParams["figure.figsize"] = (5.78851, 5.78851/16*9)
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["mathtext.fontset"] = 'cm'
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["axes.titleweight"] = 'bold'
    plt.rcParams["date.autoformatter.year"] = "%Y"
    plt.rcParams["date.autoformatter.month"] = "%Y-%m"
    plt.rcParams["date.autoformatter.day"] = "%Y-%m-%d"
    plt.rcParams["date.autoformatter.hour"] = "%Y-%m-%d %H:%M"
    plt.rcParams["date.autoformatter.minute"] = "%Y-%m-%d %H:%M"
    plt.rcParams["date.autoformatter.second"] = "%Y-%m-%d %H:%M:%S"

    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.8

    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,booktabs,amsmath,amssymb,amsfonts,multirow,type1cm}'
    plt.rcParams['text.latex.preamble'] = r'\usepackage[utf8]{inputenc}'
    plt.rcParams['text.latex.preamble'] = r'\usepackage[exponent-product=\cdot]'
    plt.rcParams['text.latex.preamble'] = r'\usepackage[strings]{underscore}'