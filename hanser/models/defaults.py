

def set(keys, value):
    if isinstance(keys, str):
        keys = [keys]
    global DEFAULTS
    d = DEFAULTS
    for k in keys[:-1]:
        assert k in d
        d = d[k]
    k = keys[-1]
    assert k in d
    d[k] = value


def get(keys):
    if isinstance(keys, str):
        keys = [keys]
    global DEFAULTS
    d = DEFAULTS
    for k in keys:
        d = d.get(k)
        if d is None:
            return d
    return d