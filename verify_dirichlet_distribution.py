import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Parameters
NUM_SIMULATIONS = 100000
NUM_TABLE_COMPARISONS = 10
STARTING_ALPHA = [0.3,.2,.5]    # Parameters to Dirichlet distribution

def generate_board_result():
    curr_alpha = STARTING_ALPHA.copy()     
    outcomes = [0, 0.5, 1]  # Matchpoint outcomes of a single table comparison
    individual_board_res = []
    for _ in range(NUM_TABLE_COMPARISONS):
        outcome_probabilities = np.random.dirichlet(curr_alpha, 1)[0]
        individual_outcome = np.random.choice(outcomes, p= outcome_probabilities)
        individual_board_res.append(individual_outcome)
        curr_alpha[outcomes.index(individual_outcome)] += .8
    return np.mean(individual_board_res)


if __name__ == "__main__":
    board_results = []
    for _ in range(NUM_SIMULATIONS):
        board_results.append(generate_board_result())
    print(np.mean(board_results))
    
    sample_mean = np.mean(board_results)
    std_err = stats.sem(board_results)
    confidence_interval = stats.norm.interval(0.95, loc=sample_mean, scale=std_err)
    
    result_str = (f'mean = {sample_mean}, std_error = {std_err}, 95% CI = {confidence_interval}')
    print(result_str)

    plt.hist(board_results, bins=4, edgecolor='black')
    plt.title('Histogram of Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
