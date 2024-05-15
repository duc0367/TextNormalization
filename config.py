class CFG:
    n_splits = 3
    n_epochs = 5
    batch_size = 16
    learning_rate = 0.001
    data_folder = 'data/text-normalization-challenge-english-language'
    max_length_encode = 1060
    max_length_decode = 3800
    embedding_hidden_size = 256
    hidden_size = 512
    use_autocast = False
    max_norm = 1000
    print_freq = 32
    output_model = 'text_normalization'
