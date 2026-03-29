from tux import StreamingCheckpointer
_, params = StreamingCheckpointer.load_trainstate_checkpoint('params::lapa_checkpoints/params', disallow_trainstate=True, max_buffer_size=32 * 2 ** 30)
print('TYPE', type(params))
if hasattr(params, 'keys'):
    k = list(params.keys())
    print('TOP_KEYS', k[:20], 'COUNT', len(k))
    if 'params' in params and hasattr(params['params'], 'keys'):
        k2 = list(params['params'].keys())
        print('PARAMS_KEYS', k2[:30], 'COUNT', len(k2))
        if 'transformer' in params['params'] and hasattr(params['params']['transformer'], 'keys'):
            print('TRANSFORMER_KEYS', list(params['params']['transformer'].keys())[:20])
