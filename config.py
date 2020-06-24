CONFIG = {
    "BUFFER_SIZE": int(1e6),     # replay buffer size
    "BATCH_SIZE": 512,           # minibatch size
    "GAMMA": 0.95,               # discount factor
    "TAU": 1e-2,                 # for soft update of target parameters
    "LR_ACTOR": 1e-3,            # learning rate of the actor
    "LR_CRITIC": 1e-3,           # learning rate of the critic
    "WEIGHT_DECAY": 0,           # L2 weight decay
    "SIGMA": 0.001,               # std of noise
    "CLIP_GRADS": True,          # Whether to clip gradients
    "CLAMP_VALUE": 0.5,          # Clip value
    "FC1": 64,                   # First linear layer size
    "FC2": 64,                   # Second linear layer size
    "WARMUP": 0,                 # number of warmup steps
}