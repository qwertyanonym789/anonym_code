Our code requires the following:
- Python 3.5
- TensorFlow 1.10.0


To run our code, 
python -m baselines.ppo2.run_mujoco_ensemble --env Walker2d-v2 --r-ex-coef 1 --r-in-coef 0.05 --K-model-num 2 --regularize 0.01 --seed 1
