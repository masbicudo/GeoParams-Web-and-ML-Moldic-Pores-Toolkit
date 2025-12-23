import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from libs.plot_spread import plot_spread
import libs.data as data
import localizable_resources as lr

def plot_best_color_params(
    ax, df, use_color_bar=False, show_means=False, show_dispersion=True,
    show_aggregation=True,
    show_exp_levels=[1, 2, 3, 4, 5]):
    """
    Plot the best color parameters based on user clicks and experience levels.
    This function filters the data based on user input and plots the clicked points.
    """
    
    # Filter data based on user input
    filtered_df = df.sort_values(by=["experience"])
    filtered_df = filtered_df[filtered_df['experience'].isin(show_exp_levels)]

    cmap = plt.get_cmap('rainbow', 5)  # Use a colormap with 5 discrete colors
    exp_levels = [
            f'1 - {lr.str.experience_levels[0]}',
            f'2 - {lr.str.experience_levels[1]}',
            f'3 - {lr.str.experience_levels[2]}',
            f'4 - {lr.str.experience_levels[3]}',
            f'5 - {lr.str.experience_levels[4]}'
        ]

    if use_color_bar:
        # Create a scatter plot with color bar
        sc = ax.scatter(filtered_df['clicked_x'],
                filtered_df['clicked_y'],
                c=filtered_df["experience"],
                cmap=cmap)
        sc.set_clim(0.5, 5.5)
        color_bar = plt.colorbar(sc, ax=ax, label=lr.str.experience)
        color_bar.locator = MaxNLocator(integer=True)
        color_bar.update_ticks()
        color_bar.set_ticks([1, 2, 3, 4, 5])  # Set ticks to match experience values
        color_bar.set_ticklabels(exp_levels)

    if show_aggregation and not filtered_df.empty:
        general_color = "black"
        if show_means:
            ax.scatter(filtered_df['clicked_x'].mean(), filtered_df['clicked_y'].mean(), color=general_color, marker='x', s=200)
        if show_dispersion:
            plot_spread(ax, filtered_df['clicked_x'], filtered_df['clicked_y'], color=general_color, draw_axes=True)

    for it in show_exp_levels:
        df_exp = filtered_df[filtered_df['experience'] == it]
        if df_exp.empty:
            continue
        color = cmap((it - 1)/5)
        if not use_color_bar:
            sc = ax.scatter(df_exp['clicked_x'],
                    df_exp['clicked_y'],
                    color=color,
                    label=exp_levels[it-1])
        if show_means:
            ax.scatter(df_exp['clicked_x'].mean(), df_exp['clicked_y'].mean(), color=color, marker='x', s=200)
        if show_dispersion:
            plot_spread(ax, df_exp['clicked_x'], df_exp['clicked_y'], color=color, draw_axes=True)

    # Set axis limits for x and y from 0 to 255 (range of the coordinates)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)

    # Invert the y-axis (top-to-bottom)
    ax.invert_yaxis()

    ax.set_aspect('equal')

    ax.set_xlabel(lr.str.k_channel_max)
    ax.set_ylabel(lr.str.c_channel_min)
    ax.set_title(lr.str.best_color_params_plot_title)

    if not use_color_bar:
        ax.legend(title=lr.str.clicked_points)
