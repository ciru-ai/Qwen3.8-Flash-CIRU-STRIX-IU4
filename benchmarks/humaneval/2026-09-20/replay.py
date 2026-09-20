"""Replay frozen inputs against an already running matching server; no retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:18090')
    p.add_argument('--inputs',type=Path,default=Path(__file__).with_name('answers.jsonl'))
    p.add_argument('--warmup',type=Path,required=True,help='Extracted evidence/iu4/full-fastest/warmup-prompt.txt')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=164)
    a=p.parse_args()
    inputs=[json.loads(x) for x in a.inputs.read_text().splitlines()]
    assert 1<=a.limit<=164 and len(inputs)==164
    a.out.mkdir(parents=True,exist_ok=False)
    gen=a.out/'generation';gen.mkdir()
    headers={'Content-Type':'application/json'}
    if os.environ.get('LLAMA_API_KEY'):
        headers['Authorization']='Bearer '+os.environ['LLAMA_API_KEY']

    def call(path,body):
        req=urllib.request.Request(a.url.rstrip('/')+path,data=json.dumps(body).encode(),headers=headers)
        with urllib.request.urlopen(req,timeout=180) as response:
            return response.read()

    warm={'prompt':a.warmup.read_text(),'n_predict':1024,'stream':True,'cache_prompt':False,'temperature':0,'seed':0,'top_p':0.95,'top_k':20,'min_p':0,'repeat_penalty':1,'presence_penalty':0,'frequency_penalty':0}
    (a.out/'warmup-response.raw').write_bytes(call('/completion',warm))
    rows=[]
    for i,item in enumerate(inputs[:a.limit]):
        assert item['task_id']==f'HumanEval/{i}'
        req=item['request']
        assert req['max_tokens']==1024 and req['temperature']==0 and req['cache_prompt'] is False
        case=gen/f'HumanEval-{i}';case.mkdir()
        (case/'request.json').write_text(json.dumps(req,indent=2)+'\n')
        started=time.perf_counter()
        raw=call('/v1/chat/completions',req)
        elapsed=time.perf_counter()-started
        (case/'response.json').write_bytes(raw)
        response=json.loads(raw);choice=response['choices'][0];msg=choice['message']
        content=msg.get('content') or ''
        (case/'completion.txt').write_text(content)
        assert content and not (msg.get('reasoning_content') or msg.get('reasoning'))
        assert response['usage']['completion_tokens']<=1024 and response['timings'].get('cache_n',0)==0
        rows.append({'task_id':item['task_id'],'wall_seconds':elapsed,'finish_reason':choice['finish_reason'],'usage':response['usage'],'timings':response['timings'],'output_sha256':hashlib.sha256(content.encode()).hexdigest()})
        (gen/'progress.json').write_text(json.dumps({'completed':len(rows),'rows':rows},indent=2)+'\n')
        print(item['task_id'],f'{elapsed:.3f}s',choice['finish_reason'],flush=True)
    times=[r['wall_seconds'] for r in rows]
    summary={'completed':len(rows),'total_wall_seconds':sum(times),'mean_wall_seconds':statistics.mean(times),'median_wall_seconds':statistics.median(times),'rows':rows}
    (gen/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))


if __name__=='__main__':
    main()
