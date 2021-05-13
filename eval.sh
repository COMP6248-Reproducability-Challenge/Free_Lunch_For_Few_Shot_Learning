# 5-way 1-shot w/o normalization
python test_DC.py --n_shot 1 --n_episode 2000 --requries_finetune --requries_hallu  --deterministic
python test_DC.py --n_shot 1 --n_episode 2000 --requries_finetune --requries_repeat --deterministic 
python test_DC.py --n_shot 1 --n_episode 2000 --requries_finetune --requries_repeat --deterministic --beta 1 

# 5-way 1-shot w/ normalization
python test_DC.py --n_shot 1 --n_episode 2000 --requries_finetune --requries_hallu  --deterministic --requries_normalize
python test_DC.py --n_shot 1 --n_episode 2000 --requries_finetune --requries_repeat --deterministic --requries_normalize 
python test_DC.py --n_shot 1 --n_episode 2000 --requries_finetune --requries_repeat --deterministic --beta 1 --requries_normalize

# 5-way 5-shot w/o normalization
python test_DC.py --n_shot 5 --n_episode 2000 --requries_finetune --requries_hallu  --deterministic
python test_DC.py --n_shot 5 --n_episode 2000 --requries_finetune --requries_repeat --deterministic 
python test_DC.py --n_shot 5 --n_episode 2000 --requries_finetune --requries_repeat --deterministic --beta 1

# 5-way 5-shot w/ normalization
python test_DC.py --n_shot 5 --n_episode 2000 --requries_finetune --requries_hallu  --deterministic --requries_normalize
python test_DC.py --n_shot 5 --n_episode 2000 --requries_finetune --requries_repeat --deterministic --requries_normalize
python test_DC.py --n_shot 5 --n_episode 2000 --requries_finetune --requries_repeat --deterministic --beta 1 --requries_normalize

