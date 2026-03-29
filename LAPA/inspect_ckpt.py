from tux import StreamingCheckpointer
from flax.traverse_util import flatten_dict

_, params = StreamingCheckpointer.load_trainstate_checkpoint('params::lapa_checkpoints/params', disallow_trainstate=True, max_buffer_size=32 * 2 ** 30)
print(type(params))
if 'params' in params:
    p = params['params']
    print('top has params key')
else:
    p = params
    print('top direct params')
flat = flatten_dict(p, sep='/')
print('num keys', len(flat))
for k in list(flat.keys())[:120]:
    print(k)
print('has transformer/wte/embedding', 'transformer/wte/embedding' in flat)
print('has wte/embedding', 'wte/embedding' in flat)
print('has model/transformer/wte/embedding', 'model/transformer/wte/embedding' in flat)
