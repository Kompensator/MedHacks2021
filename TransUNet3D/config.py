import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.embedding_size = 128
    config.num_heads = 8
    config.mlp_size = 768
    
    config.epoch = 100
    config.learning_rate = 0.01

    return config
