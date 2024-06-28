for N in {0..1000}
do
    echo "Running python script with -N $N"
    python3 run_experiments.py -N $N
done