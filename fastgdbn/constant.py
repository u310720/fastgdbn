OUT_OF_WAFER = 0  # Dice without corresponding test result
GOOD = 1  # Dice that passed the applied tests
BAD = 2  # Dice that failed the applied tests
DUI = 3  # Dice Under Investigation
UNKNOWN = 4  # Masked dice
TEST_ESCAPE = 5  # Dice that passed application testing but were defective

# Ground truth labels
INVALID = -1
ACCEPT = 0
REJECT = 1
