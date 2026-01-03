import os
from energy_net.consumption_prediction.predicting_consumption_model import save_predictions_with_train_test_split

#THIS IS HERE JUST TO SEE HOW CONSUMPTION_PREDICTIONS.CSV GOT CREATED - NOT NECESSARY TO
#RUN SINCE FILE HAS ALREADY BEEN CREATED
# =================================================================
# EXECUTION SCRIPT
# =================================================================

def main():
    """
    Executes the prediction pipeline using the default parameters
    defined in the source file.
    """
    print("Starting prediction pipeline...")

    try:
        # Calling the last function with its default parameters
        output_file, test_file = save_predictions_with_train_test_split(data_file ="synthetic_household_consumption.csv",
                                                                        output_file='consumption_predictions.csv')

        print("-" * 30)
        print(f"SUCCESS!")
        print(f"Predictions saved to: {os.path.abspath(output_file)}")
        print(f"Test split saved to:  {os.path.abspath(test_file)}")
        print("-" * 30)

    except FileNotFoundError:
        print("Error: The synthetic data file was not found at the default path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()