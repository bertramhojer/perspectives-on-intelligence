import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from demo_mappings import (criteria_mapping, label_mapping, occupation_mapping,
                           region_centers)
from matplotlib import pyplot as plt
from matplotlib.cm import RdYlBu_r
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from scipy import stats

data = pd.read_csv("data/processed/data_encoded.csv", sep=",", encoding="ISO-8859-1")

# Set style parameters with large default font sizes
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24  # Increased base font size
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = '#f5f5f5'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.labelsize'] = 26  # Increased axis label size
plt.rcParams['axes.titlesize'] = 32  # Increased title size
plt.rcParams['xtick.labelsize'] = 24  # Increased tick label size
plt.rcParams['ytick.labelsize'] = 24  # Increased tick label size
plt.rcParams['legend.fontsize'] = 24  # Increased legend font size


def save_figure(save_path, format='pdf'):
    # Ensure format is valid
    if format.lower() not in ['pdf', 'png']:
        raise ValueError("Format must be either 'pdf' or 'png'")
    
    # Save the figure
    plt.savefig(f"figures/{save_path}.{format}", format=format.lower(), bbox_inches='tight', dpi=300)
    plt.close()


## Demographic plots
def plot_area_by_career_stage(data, save_path=None, format='pdf'):
    """
    Create a stacked bar plot showing distribution of research areas by career stage.
    """
    # Create cross-tabulation of research areas and career stages
    stacked_data = pd.crosstab(data['primary_area'], data['career_stage'])

    # Sort the data by total counts
    stacked_data = stacked_data.loc[data['primary_area'].value_counts().index]

    # Create shortened labels
    shortened_labels = [label_mapping.get(area, area) for area in stacked_data.index]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))  # Increased figure size

    # Create stacked bars
    bottom = np.zeros(len(stacked_data))
    bars = []
    colors = plt.cm.GnBu(np.linspace(.2, .9, len(stacked_data.columns)))

    for i, col in enumerate(stacked_data.columns):
        bars.append(ax.bar(range(len(stacked_data)), 
                        stacked_data[col],
                        bottom=bottom,
                        width=0.6,
                        alpha=0.8,
                        label=col,
                        color=colors[i]))
        bottom += stacked_data[col]

    # Customize the plot with larger font sizes
    ax.set_xticks(range(len(shortened_labels)))
    ax.set_xticklabels(shortened_labels, ha='right', rotation=45, fontsize=24)
    ax.set_xlabel('Research Area', fontsize=28)
    ax.set_ylabel('Respondents', fontsize=28)
    ax.set_title('Distribution of Research Areas by Career Stage', fontsize=32, pad=10)

    # Add legend with larger font size
    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), 
             title='Career Stage', fontsize=20, title_fontsize=22)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    if save_path:
        save_figure(save_path, format)
    else:
        plt.show()


def plot_world_map(data, region_type="origin_region", save_path=None, format='pdf'):
    """ 
    Create a world map showing distribution of research areas by career stage.
    """
    # Get value counts of regions from origin_region
    region_counts = data[region_type].value_counts()

    # Create figure and axes with projection
    plt.figure(figsize=(18, 12))  # Increased figure size
    ax = plt.axes(projection=ccrs.Robinson())

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='white', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')

    # Set map extent to show the whole world
    ax.set_global()

    # Add circles for each region
    max_count = region_counts.max()
    for region, center in region_centers.items():
        if region in region_counts.index:
            count = region_counts[region]
            # Scale circle size based on count
            size = 7000 * np.sqrt(count / max_count)  # Increased base circle size
            
            # Create color based on count
            color = plt.cm.Blues(0.3 + 0.7 * (count / max_count))
            
            ax.scatter(center[1], center[0], 
                    s=size,
                    color=color,
                    alpha=0.6,
                    transform=ccrs.PlateCarree(),
                    label=f'{region}')  # Added count to label

    # # Add legend with larger font size
    # ax.legend(loc='center left', 
    #         bbox_to_anchor=(1.05, 0.5),
    #         frameon=True,
    #         facecolor='white',
    #         edgecolor='none',
    #         fontsize=30)  # Increased legend font size

    # Add title with larger font size
    plt.title('Geographic Distribution of Respondents', 
            pad=5,  # Increased padding
            fontsize=38)  # Increased title font size

    # Adjust layout
    plt.tight_layout()

    if save_path:
        save_figure(save_path, format)
    else:
        plt.show()
    

## Analysis plots
def plot_criteria_by_occupation(data, save_path=None, format='pdf'):
    """
    Create a stacked bar plot showing distribution of research areas by career stage.
    """
    # Get occupation columns (excluding Undisclosed)
    occupation_columns = [col for col in data.columns 
                        if col.startswith('occ_') and 'Undisclosed' not in col]

    # Calculate percentages for each criterion and occupation
    percentages_data = []
    occupation_totals = {occ: data[occ].sum() for occ in occupation_columns}

    for criterion in criteria_mapping.keys():
        criterion_data = {'Criterion': criteria_mapping[criterion]}
        for occ in occupation_columns:
            total_in_occ = occupation_totals[occ]
            selected_both = data[(data[occ] == 1) & (data[criterion] == 1)].shape[0]
            percentage = (selected_both / total_in_occ * 100) if total_in_occ > 0 else 0
            criterion_data[occ] = percentage
        percentages_data.append(criterion_data)

    # Create DataFrame
    df_percentages = pd.DataFrame(percentages_data)

    # Create the plot with increased figure size
    fig, ax = plt.subplots(figsize=(10, 14))  # Increased height for better readability

    # Set bar positions
    bar_height = 0.25
    y = np.arange(len(criteria_mapping))
    colors = plt.cm.GnBu(np.linspace(0.4, 0.8, len(occupation_columns)))

    # Create bars for each occupation
    for i, occ in enumerate(occupation_columns):
        occ_label = f"{occ.replace('occ_', '').replace('_', '/')} (n={occupation_totals[occ]})"
        position = y + i * bar_height - bar_height
        ax.barh(position,
                df_percentages[occ],
                height=bar_height,
                label=occ_label,
                color=colors[i],
                alpha=0.8)

    # Customize the plot with larger font sizes
    ax.set_yticks(y)
    ax.set_yticklabels(df_percentages['Criterion'],
                    fontsize=26)  # Increased y-axis label size
    ax.set_xlabel('Percentage of Respondents in Group', fontsize=28)  # Increased x-axis label size
    ax.set_title('Selection of Intelligence Criteria by Occupation', 
                fontsize=32,  # Increased title size
                pad=10)  # Added more padding to title

    # Add legend with larger font size
    ax.legend(bbox_to_anchor=(.42, 1),
            loc='upper left',
            fontsize=18)  # Increased legend font size

    # Set x-axis to percentages and add grid
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)

    # Increase tick label sizes
    ax.tick_params(axis='both', labelsize=22)  # Increased tick label size

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    if save_path:
        save_figure(save_path, format)
    else:
        plt.show()
    


def plot_criteria_by_career_stage(data, save_path=None, format='pdf'):
    """
    Create a stacked bar plot showing distribution of research areas by career stage.
    """
    # Define the career stages we want to include
    selected_stages = ['Research student (PhD, MPhil)', 'Postdoc',
                    'Senior (faculty or industry)', 'Junior (faculty or industry)']

    # Get career stage counts for selected stages
    career_stage_totals = data['career_stage'].value_counts()[selected_stages]

    # Calculate percentages for each criterion and career stage
    percentages_data = []
    for criterion in criteria_mapping.keys():
        criterion_data = {'Criterion': criteria_mapping[criterion]}
        for stage in selected_stages:
            total_in_stage = career_stage_totals[stage]
            selected_both = data[(data['career_stage'] == stage) & (data[criterion] == 1)].shape[0]
            percentage = (selected_both / total_in_stage * 100) if total_in_stage > 0 else 0
            criterion_data[stage] = percentage
        percentages_data.append(criterion_data)

    # Create DataFrame
    df_percentages = pd.DataFrame(percentages_data)

    # Create the plot with increased figure size
    fig, ax = plt.subplots(figsize=(10, 14))  # Increased height for better readability

    # Set bar positions
    bar_height = 0.2
    y = np.arange(len(criteria_mapping))
    colors = plt.cm.GnBu(np.linspace(0.4, 0.8, len(selected_stages)))

    # Create bars for each career stage
    for i, stage in enumerate(selected_stages):
        stage_label = f"{stage.replace(' (faculty or industry)', '')} (n={career_stage_totals[stage]})"
        position = y + i * bar_height - (bar_height * len(selected_stages)/2) + bar_height/2
        ax.barh(position,
            df_percentages[stage],
            height=bar_height,
            label=stage_label,
            color=colors[i],
            alpha=0.8)

    # Customize the plot with larger font sizes
    ax.set_yticks(y)
    ax.set_yticklabels(df_percentages['Criterion'],
                    fontsize=24,  # Increased y-axis label size
                    ha='right')
    ax.set_xlabel('Percentage of Respondents in Group', fontsize=28)  # Increased x-axis label size
    ax.set_title('Selection of Intelligence Criteria by Career Stage', 
                fontsize=32,  # Increased title size
                pad=10)  # Added more padding to title

    # Add legend with larger font size
    ax.legend(bbox_to_anchor=(0.29, 1),
            loc='upper left',
            fontsize=18)  # Increased legend font size

    # Set x-axis to percentages and add grid
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)

    # Increase tick label sizes
    ax.tick_params(axis='both', labelsize=24)  # Increased tick label size

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    if save_path:
        save_figure(save_path, format)
    else:
        plt.show()

    

def plot_sankey_diagram(data, save_path=None, format='pdf'):
    """
    Create a Sankey diagram showing the flow of intelligence criteria by career stage and occupation.
    """
    # Create flow data
    flow_df = data.groupby(['llm_intelligence', 'llm_intelligence_future']).size().reset_index(name='value')

    # Define node order (top to bottom)
    current_order = ['Strongly agree', 'Agree', 'Disagree', 'Strongly disagree']
    future_order = ['Yes, I strongly agree', 'Yes, I agree', 'No, I disagree', 'No, I strongly disagree']

    # Calculate node positions
    def calculate_node_positions(values, order, x_pos):
        total = sum(values)
        positions = {}
        current_pos = 0
        for node in order:
            value = values.get(node, 0)
            height = value / total
            positions[node] = (x_pos, current_pos + height/2, height)
            current_pos += height
        return positions

    # Get node values
    current_values = data['llm_intelligence'].value_counts()
    future_values = data['llm_intelligence_future'].value_counts()

    # Calculate positions
    left_nodes = calculate_node_positions(current_values, current_order, 0)
    right_nodes = calculate_node_positions(future_values, future_order, 1)

    # Create the plot with increased figure size
    fig, ax = plt.subplots(figsize=(14, 10))  # Increased figure size

    # Use GnBu colormap
    colors = plt.cm.GnBu(np.linspace(0.3, 0.9, 4))

    # Draw the nodes (boxes) with increased font size
    def draw_node(pos, height, label, color, width=0.2):
        rect = patches.Rectangle((pos[0], pos[1]-height/2), width, height, 
                            facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        ax.text(pos[0]+width/2, pos[1], 
                fr'$n={int(height*sum(current_values))}$', 
                va='center', ha='center', 
                fontsize=32)  # Increased font size for node labels

    def draw_flow(start, end, value, total_flow, color):
        # Calculate proportional heights on both sides
        start_height = (value/total_flow) * 0.2
        end_height = (value/total_flow) * 1
        
        # Calculate y-coordinates for flow
        start_y = start[1]
        end_y = end[1]
        
        # Calculate top and bottom points for start and end
        start_top = start_y + start_height/2
        start_bottom = start_y - start_height/2
        end_top = end_y + end_height/2
        end_bottom = end_y - end_height/2
        
        # Create the curved path
        verts = [
            (start[0]+0.2, start_top),
            (start[0]+0.4, start_top),
            (end[0]-0.4, end_top),
            (end[0], end_top),
            (end[0], end_bottom),
            (end[0]-0.4, end_bottom),
            (start[0]+0.4, start_bottom),
            (start[0]+0.2, start_bottom),
            (start[0]+0.2, start_top),
        ]
        
        codes = [
            mpath.Path.MOVETO,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.LINETO,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CLOSEPOLY,
        ]
        
        path = mpath.Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color, alpha=0.3, edgecolor='none')
        ax.add_patch(patch)

    # Draw nodes and create legend elements with increased font sizes
    legend_elements = []
    for i, (node, (x, y, h)) in enumerate(left_nodes.items()):
        draw_node((x, y), h, node, colors[i])
        legend_elements.append(patches.Patch(facecolor=colors[i], alpha=0.7,
                                        label=node))
        
    for i, (node, (x, y, h)) in enumerate(right_nodes.items()):
        draw_node((x, y), h, node, colors[i])

    # Draw flows
    total_flow = flow_df['value'].sum()
    for _, row in flow_df.iterrows():
        start = left_nodes[row['llm_intelligence']]
        end = right_nodes[row['llm_intelligence_future']]
        color = colors[current_order.index(row['llm_intelligence'])]
        draw_flow(start, end, row['value'], total_flow, color)

    # Add horizontal legend at the bottom with increased font size
    ax.legend(handles=legend_elements, 
            loc='center', 
            bbox_to_anchor=(0.53, 0.02),
            ncol=4,  # Changed to 2 columns for better readability
            frameon=True,
            columnspacing=1.0,
            fontsize=26)  # Increased legend font size

    # Customize plot with larger font sizes
    ax.set_xlim(-0.2, 1.3)
    ax.set_ylim(-0.1, 1.1)
    plt.axis('off')
    plt.title('Belief Transition:\nCurrent intelligence to future intelligence', 
             fontsize=32,  # Increased title font size
             pad=10)  # Added more padding

    plt.tight_layout()

    if save_path:
        save_figure(save_path, format)
    else:
        plt.show()



def plot_correlation_graph(data, save_path=None, format='pdf'):
    """
    Create a correlation graph showing the correlation between intelligence criteria and career stage.
    """
    # Get criteria columns
    criteria_cols = [col for col in data.columns if col.startswith('crit_')]
    criteria_data = data[criteria_cols]
    # Calculate correlation matrix
    corr_matrix = criteria_data.corr()

    # Calculate p-values using scipy's pearsonr
    p_values = np.zeros_like(corr_matrix.values)
    for i in range(len(criteria_cols)):
        for j in range(len(criteria_cols)):
            if i != j:
                _, p_value = stats.pearsonr(criteria_data[criteria_cols[i]], 
                                        criteria_data[criteria_cols[j]])
                p_values[i,j] = p_value

    # Create network graph
    G = nx.Graph()

    # Add nodes with their total responses
    total_responses = criteria_data.sum()
    for col in criteria_cols:
        node_name = col.replace('crit_', '')
        responses = total_responses[col]
        G.add_node(node_name, responses=responses)

    # Add edges (only for correlations above a threshold)
    threshold = 0.1
    for i in range(len(criteria_cols)):
        for j in range(i+1, len(criteria_cols)):
            correlation = corr_matrix.iloc[i, j]
            if abs(correlation) > threshold:
                G.add_edge(criteria_cols[i].replace('crit_', ''),
                        criteria_cols[j].replace('crit_', ''),
                        weight=correlation,
                        significant=(p_values[i,j] < 0.05))

    # Set up the plot with increased figure size
    plt.figure(figsize=(16, 14))  # Increased figure size
    main_ax = plt.gca()

    # Create circular layout
    pos = nx.circular_layout(G)

    # Calculate node sizes based on total responses
    node_responses = [G.nodes[node]['responses'] for node in G.nodes()]
    min_size = 5000  # Increased minimum node size
    max_size = 10000  # Increased maximum node size
    node_sizes = [min_size + (max_size - min_size) * (responses / max(node_responses))
                for responses in node_responses]

    # Create normalized response values for color mapping
    min_responses = min(node_responses)
    max_responses = max(node_responses)
    norm = Normalize(vmin=min_responses, vmax=max_responses)

    # Create blue color gradient and map node colors
    blues_cmap = LinearSegmentedColormap.from_list('light_blues', plt.cm.Blues(np.linspace(0.3, 0.7, 256)))
    node_colors = blues_cmap(norm(node_responses))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                        node_color=node_colors,
                        node_size=node_sizes,
                        alpha=0.7)

    # Draw edges with width and color based on correlation strength
    edges = list(G.edges(data=True))
    weights = [edge[2]['weight'] for edge in edges]

    # Create color mapping based on actual correlation values
    min_corr = min(weights)
    max_corr = max(weights)
    normalized_weights = [(w - min_corr) / (max_corr - min_corr) for w in weights]
    edge_colors = [plt.cm.RdYlBu_r(0.2 + 0.6 * nw) for nw in normalized_weights]

    # Draw edges with varying colors and styles based on significance
    for (u, v, data), width, color in zip(edges, weights, edge_colors):
        style = 'solid' if data['significant'] else 'dashed'
        nx.draw_networkx_edges(G, pos,
                            edgelist=[(u, v)],
                            width=width*14,  # Increased edge width
                            edge_color=[color],
                            alpha=0.6,
                            style=style)

    # Add labels with total responses with increased font size
    labels = {node: f"{node}\n({int(G.nodes[node]['responses'])})"
            for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, 
                          font_size=28,  # Increased label font size
                          font_weight='bold')

    # Create legend with increased font sizes
    min_r = round(min_corr, 2)
    med_r = round((min_corr + max_corr) / 2, 2)
    max_r = round(max_corr, 2)

    legend_elements = [
        Line2D([0], [0], color=plt.cm.RdYlBu_r(0.2), linewidth=4, label=fr'$\phi_c$ ≈ {min_r}'),
        Line2D([0], [0], color=plt.cm.RdYlBu_r(0.5), linewidth=4, label=fr'$\phi_c$ ≈ {med_r}'),
        Line2D([0], [0], color=plt.cm.RdYlBu_r(0.8), linewidth=4, label=fr'$\phi_c$ ≈ {max_r}')
    ]

    plt.legend(handles=legend_elements,
            title=r'Correlation ($\phi$)',
            loc='upper right',
            bbox_to_anchor=(1, 1),
            fontsize=30,  # Increased legend font size
            title_fontsize=32)  # Increased legend title font size

    plt.title('Intelligence Criteria Network',
            pad=10,  # Added more padding
            fontsize=40,  # Kept large title font size
            fontweight='bold')
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        save_figure(save_path, format)
    else:
        plt.show()


if __name__ == "__main__":

    # Demographic plots
    plot_area_by_career_stage(data, save_path="area_by_career_stage", format="pdf")
    plot_world_map(data, region_type="origin_region", save_path="world_map_origin", format="pdf")
    plot_world_map(data, region_type="work_region", save_path="world_map_work", format="pdf")

    # Analysis plots
    plot_criteria_by_occupation(data, save_path="criteria_by_occupation", format="pdf")
    plot_criteria_by_career_stage(data, save_path="criteria_by_career_stage", format="pdf")
    plot_sankey_diagram(data, save_path="sankey_diagram", format="pdf")
    plot_correlation_graph(data, save_path="correlation_graph", format="pdf")


