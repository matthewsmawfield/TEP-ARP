
import matplotlib.pyplot as plt
import matplotlib as mpl

def apply_tep_style():
    """
    Applies the TEP Manuscript visual style to matplotlib plots.
    Matches the CSS palette defined in manuscript-common.css.
    """
    
    # Color Palette - Violet-navy & gold, sampled from the site header image
    tep_red = '#c1913f'      # Gold (accent / secondary data)
    tep_blue = '#46557c'     # Slate Blue (primary data)
    tep_light_blue = '#798691' # Slate Grey-Blue (tertiary data)
    tep_green = '#241a33'    # Deep Violet (strongest emphasis)

    # Structural colors sampled from the header image
    primary_dark = '#1a1526'    # Near-black violet (headers/text)
    primary_purple = '#4a4250'  # Muted violet-grey (borders/ticks)
    background_light = '#f7f6f9'

    # Map semantic names to the new palette
    accent_magenta = tep_red
    info_blue = tep_blue
    success_green = tep_green # Mapping "green" request to this variable
    
    # High-Res Defaults
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['figure.figsize'] = (14, 9)
    
    # Fonts
    # Using serif to match the academic manuscript (Times New Roman default)
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
    mpl.rcParams['mathtext.fontset'] = 'dejavuserif' # Ensure math matches text
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 11
    mpl.rcParams['ytick.labelsize'] = 11
    mpl.rcParams['legend.fontsize'] = 11
    
    # Colors
    mpl.rcParams['text.color'] = primary_dark
    mpl.rcParams['axes.labelcolor'] = primary_dark
    mpl.rcParams['xtick.color'] = primary_purple
    mpl.rcParams['ytick.color'] = primary_purple
    mpl.rcParams['axes.edgecolor'] = primary_purple
    mpl.rcParams['axes.facecolor'] = 'white' # Keep plot area white for contrast
    mpl.rcParams['figure.facecolor'] = 'white' # Can be set to background_light if desired
    
    # Lines & Markers
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 5
    # Cycle: Gold, Slate Blue, Slate Grey-Blue, Deep Violet, Violet-Grey
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
        tep_red, tep_blue, tep_light_blue, tep_green, primary_purple
    ])
    
    # Grid
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.2
    mpl.rcParams['grid.color'] = primary_purple
    mpl.rcParams['grid.linestyle'] = '-'
    
    # Legend
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 0.95
    mpl.rcParams['legend.edgecolor'] = primary_purple
    
    # Layout
    mpl.rcParams['figure.autolayout'] = True
    
    return {
        'dark': primary_dark,
        'purple': primary_purple,
        'accent': tep_red,
        'red': tep_red,
        'blue': tep_blue,
        'light_blue': tep_light_blue,
        'green': tep_green,
        'bg': background_light
    }
