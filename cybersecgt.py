import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment, linprog
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset of attack patterns
def generate_attack_dataset(n_samples=100):
    """Generate synthetic dataset of attack patterns with various attributes"""
    data = {
        'attack_id': range(1, n_samples + 1),
        'attack_type': np.random.choice(['DDoS', 'Malware', 'Phishing', 'SQLi', 'XSS'], n_samples),
        'severity': np.random.randint(1, 11, n_samples),  # Scale of 1-10
        'frequency': np.random.randint(1, 101, n_samples),  # How often this attack occurs (per day)
        'success_rate': np.random.uniform(0.05, 0.95, n_samples),  # How often the attack succeeds
        'target_value': np.random.randint(1000, 10001, n_samples),  # Value of targeted assets
        'detection_difficulty': np.random.uniform(0.1, 1.0, n_samples),  # How hard to detect
        'mitigation_cost': np.random.randint(100, 2001, n_samples)  # Cost to mitigate
    }

    # Calculate expected damage (severity * success_rate * target_value)
    data['expected_damage'] = data['severity'] * data['success_rate'] * data['target_value']

    return pd.DataFrame(data)

# Generate defense strategies
def generate_defense_strategies(n_strategies=10):
    """Generate synthetic defense strategies with effectiveness against different attack types"""
    strategies = {
        'strategy_id': range(1, n_strategies + 1),
        'strategy_name': [f"Defense_{i}" for i in range(1, n_strategies + 1)],
        'implementation_cost': np.random.randint(500, 5001, n_strategies),
        'maintenance_cost': np.random.randint(100, 1001, n_strategies),
        'resource_usage': np.random.uniform(0.1, 1.0, n_strategies)  # CPU/Memory footprint
    }

    df = pd.DataFrame(strategies)

    # Generate effectiveness against each attack type
    attack_types = ['DDoS', 'Malware', 'Phishing', 'SQLi', 'XSS']
    for attack_type in attack_types:
        df[f'effectiveness_{attack_type}'] = np.random.uniform(0.3, 0.95, n_strategies)

    return df

# Visualize attack patterns
def visualize_attacks(attacks):
    """Create visualizations of attack patterns"""
    plt.figure(figsize=(15, 10))

    # Plot 1: Attack types distribution
    plt.subplot(2, 2, 1)
    attack_counts = attacks['attack_type'].value_counts()
    sns.barplot(x=attack_counts.index, y=attack_counts.values, hue=attack_counts.index, legend=False, palette='viridis')
    plt.title('Distribution of Attack Types')
    plt.xlabel('Attack Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # Plot 2: Severity vs Expected Damage
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=attacks, x='severity', y='expected_damage',
                    hue='attack_type', size='frequency', sizes=(20, 200),
                    alpha=0.7, palette='viridis')
    plt.title('Severity vs Expected Damage by Attack Type')
    plt.xlabel('Severity')
    plt.ylabel('Expected Damage')

    # Plot 3: Detection Difficulty vs Success Rate
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=attacks, x='detection_difficulty', y='success_rate',
                    hue='attack_type', size='target_value', sizes=(20, 200),
                    alpha=0.7, palette='viridis')
    plt.title('Detection Difficulty vs Success Rate')
    plt.xlabel('Detection Difficulty')
    plt.ylabel('Success Rate')

    # Plot 4: Expected Damage Distribution
    plt.subplot(2, 2, 4)
    sns.histplot(data=attacks, x='expected_damage', bins=20, kde=True)
    plt.axvline(attacks['expected_damage'].mean(), color='red', linestyle='--',
                label=f'Mean: {attacks["expected_damage"].mean():.2f}')
    plt.title('Distribution of Expected Damage')
    plt.xlabel('Expected Damage')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig('attack_visualizations.png')
    plt.close()

# Visualize defense strategies
def visualize_defenses(defenses):
    """Create visualizations of defense strategies"""
    plt.figure(figsize=(15, 10))

    # Plot 1: Implementation Cost vs Maintenance Cost
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=defenses, x='implementation_cost', y='maintenance_cost',
                    size='resource_usage', sizes=(50, 300), alpha=0.7)
    plt.title('Implementation Cost vs Maintenance Cost')
    plt.xlabel('Implementation Cost')
    plt.ylabel('Maintenance Cost')

    # Plot 2: Effectiveness comparison across attack types
    # Melt the dataframe to get all effectiveness values in one column
    eff_columns = [col for col in defenses.columns if 'effectiveness_' in col]
    eff_data = defenses.melt(id_vars=['strategy_name'],
                            value_vars=eff_columns,
                            var_name='attack_type',
                            value_name='effectiveness')
    # Clean up attack type names
    eff_data['attack_type'] = eff_data['attack_type'].str.replace('effectiveness_', '')

    plt.subplot(2, 2, 2)
    sns.boxplot(data=eff_data, x='attack_type', y='effectiveness')
    plt.title('Effectiveness Distribution by Attack Type')
    plt.xlabel('Attack Type')
    plt.ylabel('Effectiveness')
    plt.xticks(rotation=45)

    # Plot 3: Effectiveness heatmap
    plt.subplot(2, 1, 2)
    # Prepare data for heatmap
    heatmap_data = defenses.copy()
    heatmap_data = heatmap_data.set_index('strategy_name')
    heatmap_data = heatmap_data[eff_columns]
    # Rename columns for better readability
    heatmap_data.columns = [col.replace('effectiveness_', '') for col in heatmap_data.columns]

    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Effectiveness'})
    plt.title('Defense Strategies Effectiveness Heatmap')
    plt.xlabel('Attack Type')
    plt.ylabel('Defense Strategy')

    plt.tight_layout()
    plt.savefig('defense_visualizations.png')
    plt.close()

# Create payoff matrix for game theory analysis
def create_payoff_matrix(attacks, defenses):
    """Create a payoff matrix for game theory analysis"""
    n_attacks = len(attacks)
    n_defenses = len(defenses)

    # Initialize payoff matrix
    payoff_matrix = np.zeros((n_attacks, n_defenses))

    # Calculate payoffs for each attack-defense pair
    for i, (_, attack) in enumerate(attacks.iterrows()):
        attack_type = attack['attack_type']
        expected_damage = attack['expected_damage']

        for j, (_, defense) in enumerate(defenses.iterrows()):
            # Get defense effectiveness against this attack type
            effectiveness = defense[f'effectiveness_{attack_type}']

            # Defense cost factors
            defense_cost = defense['implementation_cost'] / 1000 + defense['maintenance_cost'] / 2000

            # Calculate payoff from defender's perspective
            defender_utility = effectiveness * expected_damage - defense_cost
            attacker_utility = (1 - effectiveness) * expected_damage

            payoff_matrix[i, j] = defender_utility - attacker_utility

    return payoff_matrix

# Visualize payoff matrix
def visualize_payoff_matrix(payoff_matrix, attacks, defenses):
    """Visualize the payoff matrix"""
    plt.figure(figsize=(12, 10))

    # Use attack IDs and defense names for labels
    attack_labels = [f"A{attack['attack_id']}-{attack['attack_type']}" for _, attack in attacks.iloc[:10].iterrows()]
    defense_labels = defenses['strategy_name'].tolist()

    # Create heatmap (showing only first 10 attacks for readability)
    sns.heatmap(payoff_matrix[:10, :], annot=True, cmap='RdBu_r', center=0,
                xticklabels=defense_labels, yticklabels=attack_labels,
                cbar_kws={'label': 'Payoff (Defender Perspective)'})
    plt.title('Game Theory Payoff Matrix Sample (First 10 Attacks)')
    plt.xlabel('Defense Strategies')
    plt.ylabel('Attack Patterns')
    plt.tight_layout()
    plt.savefig('payoff_matrix.png')
    plt.close()

# Solve zero-sum game
def solve_zero_sum_game(payoff_matrix):
    """Solve a zero-sum game to find optimal mixed strategies"""
    num_rows, num_cols = payoff_matrix.shape

    try:
        # For defender (column player)
        c_col = -np.ones(num_cols)  # Objective: maximize minimum payoff
        A_col = payoff_matrix  # Constraints: expected payoff for each attack
        b_col = np.ones(num_rows)  # Strategy must sum to 1

        # Solve using linear programming
        res_col = linprog(
            c=c_col,
            A_ub=A_col,
            b_ub=b_col,
            bounds=(0, None),
            method='highs'
        )

        # Check if solution was found
        if res_col.success:
            # Normalize to get probability distribution
            defender_strategy = res_col.x / np.sum(res_col.x)
            game_value = -res_col.fun
        else:
            # Fallback to uniform distribution if solver fails
            defender_strategy = np.ones(num_cols) / num_cols
            game_value = 0

    except Exception as e:
        print(f"Error solving game: {e}")
        defender_strategy = np.ones(num_cols) / num_cols
        game_value = 0

    # For attacker, we use a uniform distribution as a fallback
    attacker_strategy = np.ones(num_rows) / num_rows

    return {
        'defender_strategy': defender_strategy,
        'attacker_strategy': attacker_strategy,
        'game_value': game_value
    }

# Visualize game theory solution
def visualize_game_solution(solution, defenses):
    """Visualize the game theory solution"""
    plt.figure(figsize=(10, 6))

    # Plot defender's mixed strategy
    defender_strategy_df = pd.DataFrame({
        'Strategy': defenses['strategy_name'],
        'Probability': solution['defender_strategy']
    })

    sns.barplot(data=defender_strategy_df, x='Strategy', y='Probability', hue='Strategy', palette='Blues_d', legend=False)
    plt.title(f'Optimal Defender Strategy (Game Value: {solution["game_value"]:.2f})')
    plt.xlabel('Defense Strategy')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    plt.axhline(y=1/len(defenses), color='red', linestyle='--',
                label=f'Uniform Strategy (1/{len(defenses)})')
    plt.legend()
    plt.tight_layout()
    plt.savefig('game_solution.png')
    plt.close()

# Cluster attacks
def perform_attack_clustering(attacks, n_clusters=5):
    """Cluster attacks based on their characteristics"""
    # Extract features for clustering
    features = attacks[['severity', 'frequency', 'success_rate',
                        'detection_difficulty', 'expected_damage']].copy()

    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Add cluster info to attacks
    attacks_with_clusters = attacks.copy()
    attacks_with_clusters['cluster'] = clusters

    return attacks_with_clusters, kmeans

# Visualize attack clusters
def visualize_clusters(attacks_with_clusters):
    """Visualize the attack clusters"""
    plt.figure(figsize=(15, 10))

    # Plot 1: Cluster distribution
    plt.subplot(2, 2, 1)
    cluster_counts = attacks_with_clusters['cluster'].value_counts().sort_index()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, hue=cluster_counts.index, palette='viridis', legend=False)
    plt.title('Distribution of Attacks by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')

    # Plot 2: Clusters by Expected Damage vs Detection Difficulty
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=attacks_with_clusters, x='expected_damage', y='detection_difficulty',
                    hue='cluster', palette='viridis', s=100, alpha=0.7)
    plt.title('Clusters by Expected Damage vs Detection Difficulty')
    plt.xlabel('Expected Damage')
    plt.ylabel('Detection Difficulty')

    # Plot 3: Clusters by Severity vs Success Rate
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=attacks_with_clusters, x='severity', y='success_rate',
                    hue='cluster', palette='viridis', s=100, alpha=0.7)
    plt.title('Clusters by Severity vs Success Rate')
    plt.xlabel('Severity')
    plt.ylabel('Success Rate')

    # Plot 4: Attack types within each cluster
    plt.subplot(2, 2, 4)
    cluster_attack_types = pd.crosstab(attacks_with_clusters['cluster'],
                                      attacks_with_clusters['attack_type'])
    cluster_attack_types.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Attack Types within Each Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='Attack Type')

    plt.tight_layout()
    plt.savefig('attack_clusters.png')
    plt.close()

# Implement dynamic threat response system
class DynamicThreatResponseSystem:
    def _init_(self, attacks, defenses):
        self.attacks = attacks
        self.defenses = defenses
        self.clusters = None
        self.kmeans = None
        self.cluster_models = {}
        self.threat_history = []
        self.response_history = []

    def build_model(self, n_clusters=5):
        """Build the complete model including clustering and game theory solutions"""
        # Cluster attacks - using the global function with a different name
        self.attacks, self.kmeans = perform_attack_clustering(self.attacks, n_clusters)
        self.clusters = n_clusters

        # Build game theory models for each cluster
        for cluster in range(self.clusters):
            # Get attacks in this cluster
            cluster_attacks = self.attacks[self.attacks['cluster'] == cluster]

            if len(cluster_attacks) == 0:
                continue

            # Create payoff matrix for this cluster
            payoff_matrix = create_payoff_matrix(cluster_attacks, self.defenses)

            # Solve the game
            solution = solve_zero_sum_game(payoff_matrix)

            # Store the model
            self.cluster_models[cluster] = {
                'payoff_matrix': payoff_matrix,
                'solution': solution,
                'attack_count': len(cluster_attacks)
            }

    def get_optimal_defense(self, attack):
        """Get the optimal defense strategy for a given attack"""
        # Convert attack to feature vector
        features = np.array([[
            attack['severity'],
            attack['frequency'],
            attack['success_rate'],
            attack['detection_difficulty'],
            attack['expected_damage']
        ]])

        # Normalize using the same scaler (approximate)
        scaler = StandardScaler()
        all_features = self.attacks[['severity', 'frequency', 'success_rate',
                                    'detection_difficulty', 'expected_damage']].values
        scaler.fit(all_features)
        scaled_features = scaler.transform(features)

        # Predict cluster
        cluster = self.kmeans.predict(scaled_features)[0]

        # Get defense strategy for this cluster
        if cluster in self.cluster_models:
            defense_strategy = self.cluster_models[cluster]['solution']['defender_strategy']

            # Choose defense according to the probability distribution
            defense_idx = np.random.choice(len(self.defenses), p=defense_strategy)
            defense = self.defenses.iloc[defense_idx]
        else:
            # Fallback to random defense
            defense_idx = np.random.randint(0, len(self.defenses))
            defense = self.defenses.iloc[defense_idx]

        # Record the attack and response
        self.threat_history.append(attack)
        response = {
            'attack': attack,
            'cluster': cluster,
            'defense_strategy': defense['strategy_name'],
            'defense_id': defense['strategy_id'],
            'effectiveness': defense[f"effectiveness_{attack['attack_type']}"]
        }
        self.response_history.append(response)

        return response

# Simulate attack-defense scenario
def simulate_security_scenario(system, simulation_steps=30):
    """Simulate a series of attacks and defense responses"""
    results = []

    for step in range(simulation_steps):
        # Generate a random attack
        attack = {
            'attack_id': 1000 + step,
            'attack_type': np.random.choice(['DDoS', 'Malware', 'Phishing', 'SQLi', 'XSS']),
            'severity': np.random.randint(1, 11),
            'frequency': np.random.randint(1, 101),
            'success_rate': np.random.uniform(0.05, 0.95),
            'target_value': np.random.randint(1000, 10001),
            'detection_difficulty': np.random.uniform(0.1, 1.0),
            'mitigation_cost': np.random.randint(100, 2001)
        }
        attack['expected_damage'] = attack['severity'] * attack['success_rate'] * attack['target_value']

        # Get system response
        response = system.get_optimal_defense(attack)

        # Calculate outcome
        effectiveness = response['effectiveness']
        mitigated_damage = attack['expected_damage'] * effectiveness
        remaining_damage = attack['expected_damage'] - mitigated_damage

        # Record result
        results.append({
            'step': step,
            'attack_type': attack['attack_type'],
            'cluster': response['cluster'],
            'expected_damage': attack['expected_damage'],
            'mitigated_damage': mitigated_damage,
            'remaining_damage': remaining_damage,
            'defense_strategy': response['defense_strategy'],
            'effectiveness': effectiveness
        })

    return pd.DataFrame(results)

# Visualize simulation results
def visualize_simulation(results):
    """Visualize the simulation results"""
    plt.figure(figsize=(15, 15))

    # Plot 1: Damage over time
    plt.subplot(3, 1, 1)
    plt.plot(results['step'], results['expected_damage'], 'r-', label='Expected Damage')
    plt.plot(results['step'], results['mitigated_damage'], 'g-', label='Mitigated Damage')
    plt.plot(results['step'], results['remaining_damage'], 'b-', label='Remaining Damage')
    plt.xlabel('Simulation Step')
    plt.ylabel('Damage Value')
    plt.title('Attack Mitigation Performance Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Defense effectiveness over time
    plt.subplot(3, 1, 2)
    effectiveness_by_step = results.set_index('step')['effectiveness']
    plt.bar(results['step'], results['effectiveness'], color='purple', alpha=0.7)
    plt.axhline(y=effectiveness_by_step.mean(), color='k', linestyle='--',
                label=f'Avg Effectiveness: {effectiveness_by_step.mean():.2f}')
    plt.xlabel('Simulation Step')
    plt.ylabel('Defense Effectiveness')
    plt.title('Defense Effectiveness by Simulation Step')
    plt.legend()

    # Plot 3: Effectiveness by attack type and cluster
    plt.subplot(3, 2, 5)
    attack_effectiveness = results.groupby('attack_type')['effectiveness'].mean().sort_values()
    sns.barplot(x=attack_effectiveness.index, y=attack_effectiveness.values,
                hue=attack_effectiveness.index, palette='viridis', legend=False)
    plt.title('Average Defense Effectiveness by Attack Type')
    plt.xlabel('Attack Type')
    plt.ylabel('Average Effectiveness')
    plt.xticks(rotation=45)

    plt.subplot(3, 2, 6)
    cluster_effectiveness = results.groupby('cluster')['effectiveness'].mean().sort_index()
    sns.barplot(x=cluster_effectiveness.index, y=cluster_effectiveness.values,
                hue=cluster_effectiveness.index, palette='viridis', legend=False)
    plt.title('Average Defense Effectiveness by Attack Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Effectiveness')

    plt.tight_layout()
    plt.savefig('simulation_results.png')
    plt.close()

    # Create a second figure for defense performance
    plt.figure(figsize=(12, 10))

    # Plot 1: Defense strategy distribution
    plt.subplot(2, 1, 1)
    defense_counts = results['defense_strategy'].value_counts()
    sns.barplot(x=defense_counts.index, y=defense_counts.values,
                hue=defense_counts.index, palette='viridis', legend=False)
    plt.title('Distribution of Defense Strategies Used')
    plt.xlabel('Defense Strategy')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # Plot 2: Defense strategy effectiveness
    plt.subplot(2, 1, 2)
    strategy_performance = results.groupby('defense_strategy').agg({
        'expected_damage': 'sum',
        'mitigated_damage': 'sum',
        'effectiveness': 'mean'
    }).reset_index()
    strategy_performance['efficiency'] = strategy_performance['mitigated_damage'] / strategy_performance['expected_damage']

    sns.barplot(x='defense_strategy', y='efficiency',
                hue='defense_strategy', data=strategy_performance, palette='viridis', legend=False)
    plt.title('Defense Strategy Efficiency (Mitigated/Expected Damage)')
    plt.xlabel('Defense Strategy')
    plt.ylabel('Efficiency')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('defense_performance.png')
    plt.close()

# Main function to run the simulation
def main():
    print("Starting cybersecurity game theory simulation...")

    # Generate attack patterns and defense strategies
    print("Generating attack dataset...")
    attacks = generate_attack_dataset(100)
    print(f"Generated {len(attacks)} attack patterns")

    print("Generating defense strategies...")
    defenses = generate_defense_strategies(10)
    print(f"Generated {len(defenses)} defense strategies")

    # Visualize the datasets
    print("Creating visualizations of attack patterns...")
    visualize_attacks(attacks)

    print("Creating visualizations of defense strategies...")
    visualize_defenses(defenses)

    # Create payoff matrix for the complete dataset
    print("Creating payoff matrix...")
    payoff_matrix = create_payoff_matrix(attacks, defenses)
    visualize_payoff_matrix(payoff_matrix, attacks, defenses)

    # Solve the game
    print("Solving the game theory model...")
    solution = solve_zero_sum_game(payoff_matrix)
    visualize_game_solution(solution, defenses)

    # Cluster attacks
    print("Clustering attack patterns...")
    attacks_with_clusters, _ = perform_attack_clustering(attacks)
    visualize_clusters(attacks_with_clusters)

    # Initialize the dynamic threat response system
    print("Initializing dynamic threat response system...")
    system = DynamicThreatResponseSystem(attacks, defenses)
    system.build_model(5)

    # Run simulation
    print("Running security simulation...")
    results = simulate_security_scenario(system, 30)

    # Visualize simulation results
    print("Creating simulation visualizations...")
    visualize_simulation(results)

    # Calculate and display efficiency metrics
    total_potential_damage = results['expected_damage'].sum()
    total_mitigated_damage = results['mitigated_damage'].sum()
    efficiency = total_mitigated_damage / total_potential_damage
    print(f"Security Efficiency: {efficiency:.2%}")

    print("Simulation complete. Visualization files have been saved.")
    attacks.to_csv('attacks_data.csv', index=False)
    defenses.to_csv('defenses_data.csv', index=False)
    results.to_csv('simulation_results.csv', index=False)

    print("Data saved to CSV files.")



    return {
        'attacks': attacks,
        'defenses': defenses,
        'system': system,
        'results': results
    }

if _name_ == "_main_":
    results = main()
