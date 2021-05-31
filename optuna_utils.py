def get_nc_args(trial, nc, idx):
    name = "nc_argv" + str(idx)
    if nc == "Conv":
        out_channel = trial.suggest_int(name + "/out_channel", 3, 64)
        kernel_size = trial.suggest_int(name + "/kernel_size", 1, 7, step=2)
        activation = trial.suggest_categorical(name + "/activation", ["ReLU", "ReLU6", "Hardswish"])
        groups = 1
        if kernel_size == 1:
            padding = None
        elif kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        return [out_channel, kernel_size, 1, padding, groups, activation]

    elif nc == "DWConv":
        out_channel = trial.suggest_int(name + "/out_channel", 3, 64)
        kernel_size = trial.suggest_int(name + "/kernel_size", 1, 7, step=2)
        activation = trial.suggest_categorical(name + "/activation", ["ReLU", "ReLU6", "Hardswish"])
        if kernel_size == 1:
            padding = None
        elif kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        return [out_channel, kernel_size, 1, padding, activation]

    elif nc == "Bottleneck":
        out_channel = trial.suggest_int(name + "/out_channel", 3, 64)
        activation = trial.suggest_categorical(name + "/activation", ["ReLU", "ReLU6", "Hardswish"])
        shortcut = trial.suggest_categorical(name + "/shortcut", [False, True])
        expansion = trial.suggest_categorical(name + "/expansion", [0.5, 1, 2, 3])
        # Bottlenect output, shortcut , groups, expansion , activation
        return [out_channel, shortcut, 1, expansion, activation]

    elif nc == "InvertedResidualv2":
        stride = 1
        out_channel = trial.suggest_int(name + "/out_channel", 3, 64)
        expand = 0.5
        # InvertedResidualv2 :  oup, expand_ratio, stride(1,2)
        return [out_channel, expand, stride]

    elif nc == "MBConv":
        out_channel = trial.suggest_int(name + "/out_channel", 3, 64)
        kernel_size = trial.suggest_int(name + "/kernel_size", 3, 5, step=2)
        expand = 0.5
        stride = 1
        # MBConv : output, expand, stride(1,2), kernel_size(3,5)
        return [out_channel, expand, stride, kernel_size]


def get_rc_args(trial, rc, idx):
    name = "rc_argv" + str(idx)
    if rc == "InvertedResidualv2":
        t = trial.suggest_int(name + "/t", 1, 6)
        out_channel = trial.suggest_int(name + "/out_channel", 3, 64)
        stride = 2
        return [out_channel, t, stride]

    elif rc == "InvertedResidualv3":
        t = trial.suggest_int(name + "/t", 1, 6)
        kernel_size = trial.suggest_int(name + "/kernel_size", 3, 5, step=2)
        out_channel = trial.suggest_int(name + "/out_channel", 3, 64)
        c = trial.suggest_categorical(name + "/c", [16, 24, 40, 80, 112, 160])
        SE = trial.suggest_int(name + "/SE", 0, 1)
        HS = trial.suggest_int(name + "/HS", 0, 1)
        stride = 2
        return [kernel_size, t, c, SE, HS, stride]

    elif rc == "MaxPool":
        stride = trial.suggest_int(name + "/stride", 2, 4, step=2)
        return [stride]

    elif rc == "AvgPool":
        stride = trial.suggest_int(name + "/stride", 2, 4, step=2)
        return [stride]


def sample_model(trial):
    cfg = {}
    cfg["input_channel"] = 3
    cfg["depth_multiple"] = 1.0
    cfg["width_multiple"] = 1.0
    n1 = trial.suggest_int("n1", 1, 2)
    n2 = trial.suggest_int("n2", 1, 3)
    n3 = trial.suggest_int("n3", 1, 3)

    nc1 = trial.suggest_categorical("nc1", ["DWConv", "InvertedResidualv2", "MBConv"])
    nc1_args = get_nc_args(trial, nc1, 1)
    nc2 = trial.suggest_categorical("nc2", ["Conv", "DWConv", "Bottleneck", "InvertedResidualv2", "MBConv"])
    nc2_args = get_nc_args(trial, nc2, 2)
    nc3 = trial.suggest_categorical("nc3", ["Conv", "DWConv", "Bottleneck", "InvertedResidualv2", "MBConv"])
    nc3_args = get_nc_args(trial, nc3, 3)
    rc1 = trial.suggest_categorical("rc1", ["InvertedResidualv2", "InvertedResidualv3", "MaxPool", "AvgPool"])
    rc1_args = get_rc_args(trial, rc1, 1)
    rc2 = trial.suggest_categorical("rc2", ["InvertedResidualv2", "InvertedResidualv3", "MaxPool", "AvgPool"])
    rc2_args = get_rc_args(trial, rc2, 2)
    back = []
    back.append([n1, nc1, nc1_args])
    back.append([1, rc1, rc1_args])
    back.append([n2, nc2, nc2_args])
    back.append([1, rc2, rc2_args])
    back.append([n3, nc3, nc3_args])
    back.append([1, "GlobalAvgPool", []])
    back.append([1, "Flatten", []])
    back.append([1, "Linear", [9]])
    cfg["backbone"] = back
    return cfg
