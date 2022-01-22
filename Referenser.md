# installera neural prophet
https://github.com/ourownstory/neural_prophet 
conda create --name time_series_test_nu python=3.9 
conda activate time_series_test_nu
conda install -c anaconda ipykernel 
pip install neuralprophet 
conda install -c conda-forge imageio

# ta bort miljö
conda env remove -n time_series_test_nu