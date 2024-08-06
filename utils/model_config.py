CONFIGS_ = {
    # input_channel, n_class, hidden_dim, latent_dim
    'cmapss': ([6, 16, 'F'], 14, 126, 512, 32)
}

GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    'cmapss': (512, 256, 1, 126, 64),
}

RUNCONFIGS = {
    'cmapss':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 32,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'ensemble_eta': 1,      # diversity loss
            'unique_labels': 126,    # available labels
            'generative_alpha': 10, # used to regulate user training
            'generative_beta': 10, # used to regulate user training
            'weight_decay': 0.0001
        },
}

