import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def vis(csv_in):
    df = pd.read_csv(csv_in)

    # grouping
    groups = {
        'Death-related': ['deaths', 'death', 'dead'],
        'Health-related': ['health', 'avg_health','shield'],
        'Round-related': ['round_info', 'spike_planted', 'map','movement'],
        'Ability-related': ['crowd_control', 'total_ability','information','healing','damage_for_self','damage_for_team'],
        'Weapon-related': ['gun', 'melee', 'ammo', 'inv_state','loadout'],
        'Elimination-related': ['elims', 'damage_done', 'headshots'],
        'Economy-related': ['credits'],
    }
    df['group'] = df['feature'].apply(lambda x: next((group for group, keywords in groups.items() if any(keyword in x for keyword in keywords)), 'Other'))

    specific_colors = {
        'Death-related': '#66c2a5',
        'Economy-related': '#e78ac3',
        'Health-related': '#fc8d62',
        'Round-related': '#8da0cb',
        'Ability-related': '#a6d854',
        'Weapon-related': '#ffd92f',
        'Elimination-related': '#e5c494',
        'Other': '#b3b3b3',
    }

    # top 20 features
    df_sorted = df.sort_values(by='importance', ascending=False)

    # style
    sns.set(style="whitegrid")

    # plotting
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x='importance', y='feature', data=df_sorted, hue='group', dodge=False, palette=specific_colors)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Top 20 Feature Importance with Feature-type Grouping')
    plt.tight_layout()

    # legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:len(groups)], labels=labels[:len(groups)])
    plt.savefig('win_loss/dtree_csv_feature_results/feat_vis_duelists_4', dpi=600, bbox_inches='tight')
    plt.show()

vis('roles/dtree_csv_feature_results_class/df_role_import_dtree_no_map_13.csv')