import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Parameters
NUM_SIMULATIONS = 500000
NUM_TABLE_COMPARISONS = 10
STARTING_SCORE = 0.2    # Your starting bad score percentage
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

def run_trial(res):
    cum_score = STARTING_SCORE
    num_boards_played = 1
    while num_boards_played < 10:
        cum_score += generate_board_result()
        num_boards_played += 1
        if cum_score >= 0.7 * num_boards_played:
            res[num_boards_played] += 1

if __name__ == "__main__":
    res = {i: 0 for i in range(1,11)}
    for _ in range(NUM_SIMULATIONS):
        run_trial(res)
    
    for num_boards_played, freq in res.items():
        sample_proportion = freq / NUM_SIMULATIONS
        std_err = np.sqrt(sample_proportion * (1 - sample_proportion) / NUM_SIMULATIONS)
        confidence_interval = stats.norm.interval(0.95, loc=sample_proportion, scale=std_err)
        print(f'{num_boards_played}: proportion = {sample_proportion}, 95% CI = {confidence_interval}')
    



