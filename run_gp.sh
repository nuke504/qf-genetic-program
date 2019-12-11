# Run genetic program over a few parameters many times
# Conditions for running the GP
# Fitness: Accuracy
# Init Population 200
# With ['or','and']
# Run the genetic program 5 times for different settings
# Modify as required
FITNESS_METRIC="accuracy"
OPERATOR_LIST="['or','and']"
INIT_POPULATION=200

for NUMBER_ARRAYS in 20 30; do
    for ((i=0;i<5;i++)); do
        python3 run_gp.py --fitness_metric=$FITNESS_METRIC --operator_list=$OPERATOR_LIST --init_population=$INIT_POPULATION --number_of_training_arrays=$NUMBER_ARRAYS
        echo "Iteration $i completed"
    done
    echo "Completed 5 iterations for number_of_training_arrays=$NUMBER_ARRAYS"
done