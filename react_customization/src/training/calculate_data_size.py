import json
from tqdm import tqdm

total = 0

pbar = tqdm(range(986))
for i in pbar:
    stat_file = f'/mnt/mydata/cvinwild/react/imagenet_10m/{i:05d}_stats.json'
    with open(stat_file, 'r') as fp:
        stats = json.load(fp)
        
        total += stats['successes']
        pbar.set_description(f'Total: {total}')

print(total)