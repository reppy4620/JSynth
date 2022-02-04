def add_prefix(loss_dict, mode):
    return {f'{mode}/{key}': val for key, val in loss_dict.items()}
