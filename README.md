## Structure:
- (Not here) I stored data as csv for easier use, `train.csv` and `test.csv`.
- `/Notebooks`: Contains `EDA.ipynb` and `final_results.ipynb`.
- `requirements.txt`: Python packages used.
- `utils.py`: Helper function codifying data processing pre-modeling. Used in `train_and_predict.py`.
- `train_and_predict.py`: Trains model and makes prediction, expects path to data csv as arguments. Main component of `run.sh`.
- `run.sh`: Allows you to 
- `/prophet-mod-v1`: Executing `run.sh` reproduces these results.

## Thought process chain (ordered):
- After creating `EDA.ipynb` to answer #1, I made some decisions about how to approach modeling (see file)
- I created `train_and_predict.py` to provide a reproducible and somewhat systematic way to perform model training, prediction, and evaluation.
- I created `run.sh` so that you could replicate my results:
    - I'm using macOS, YMMV on non-unix based machine.
    - Make sure `run.sh` is executable: `chmod +x run.sh`
    - You need Python >=3.9 installed and accessible. If you alias Python in some (non-standard?) way, you may need to change line 15 to reference `python` properly.
    - You can execute in a zsh terminal like `./run.sh`
- `/prophet-mod-v1`: I included a copy of the results I generated using `run.sh`, in case you cannot.
- I then generated `final_results.ipynb` based on contents of `/prophet-mod-v1`.