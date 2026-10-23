import json
from concurrent.futures import ThreadPoolExecutor

import prepare_magicbrush as prep


def main():
    root=prep.ROOT
    (root/'data').mkdir(parents=True,exist_ok=True)
    inventory=prep.repo_inventory(root)
    with ThreadPoolExecutor(max_workers=4) as pool:
        metadata=list(pool.map(lambda shard:prep.metadata_shard(root,shard),inventory['files']))
    selection=prep.freeze_selection(root,inventory,metadata)
    print(json.dumps({'status':'COMPLETE_FROZEN','workers':4,'counts':{k:len(v) for k,v in selection['selections'].items()},
        'model_calls':0,'target_columns_read':False}),flush=True)


if __name__=='__main__':main()
