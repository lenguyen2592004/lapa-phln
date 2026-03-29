mods = [
    'jax','flax','optax','chex','transformers','datasets','ml_collections','wandb','gcsfs','sentencepiece','PIL','decord','tiktoken','tensorflow','scipy','albumentations','uvicorn','fastapi','tux'
]
missing=[]
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        missing.append((m, str(e)))
print('MISSING_COUNT', len(missing))
for m,e in missing:
    print(m, '::', e)
