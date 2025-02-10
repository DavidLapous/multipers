import numpy as np


def save_with_axis(path: str, signed_measures):
    np.savez(
        path,
        **{
            f"{i}_{axis}_{degree}": np.c_[
                sm_of_degree[0], sm_of_degree[1][:, np.newaxis]
            ]
            for i, sm in enumerate(signed_measures)
            for axis, sm_of_axis in enumerate(sm)
            for degree, sm_of_degree in enumerate(sm_of_axis)
        },
    )


def save_without_axis(path: str, signed_measures):
    np.savez(
        path,
        **{
            f"{i}_{degree}": np.c_[sm_of_degree[0], sm_of_degree[1][:, np.newaxis]]
            for i, sm in enumerate(signed_measures)
            for degree, sm_of_degree in enumerate(sm)
        },
    )


def get_sm_with_axis(sms, idx, axis, degree):
    sm = sms[f"{idx}_{axis}_{degree}"]
    return (sm[:, :-1], sm[:, -1])


def get_sm_without_axis(sms, idx, degree):
    sm = sms[f"{idx}_{degree}"]
    return (sm[:, :-1], sm[:, -1])


def load_without_axis(sms):
    indices = np.array(
        [[int(i) for i in key.split("_")] for key in sms.keys()], dtype=int
    )
    num_data, num_degrees = indices.max(axis=0) + 1
    signed_measures_reconstructed = [
        [get_sm_without_axis(sms, idx, degree) for degree in range(num_degrees)]
        for idx in range(num_data)
    ]
    return signed_measures_reconstructed


# test : np.all([np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1]) and len(a) == len(b) == 2 for x,y in zip(signed_measures_reconstructed,signed_measures_reconstructed) for a,b in zip(x,y)])


def load_with_axis(sms):
    indices = np.array(
        [[int(i) for i in key.split("_")] for key in sms.keys()], dtype=int
    )
    num_data, num_axis, num_degrees = indices.max(axis=0) + 1
    signed_measures_reconstructed = [
        [
            [get_sm_with_axis(sms, idx, axis, degree) for degree in range(num_degrees)]
            for axis in range(num_axis)
        ]
        for idx in range(num_data)
    ]
    return signed_measures_reconstructed


def save(path: str, signed_measures):
    if isinstance(signed_measures[0][0], tuple):
        save_without_axis(path=path, signed_measures=signed_measures)
    else:
        save_with_axis(path=path, signed_measures=signed_measures)


def load(path: str):
    sms = np.load(path)
    item = None
    for i in sms.keys():
        item = i
        break
    n = len(item.split("_"))
    match n:
        case 2:
            return load_without_axis(sms)
        case 3:
            return load_with_axis(sms)
        case _:
            raise Exception("Invalid Signed Measure !")

