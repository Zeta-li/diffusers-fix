import hashlib
import json
import os
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
PYTHON = '/workspace/venvs/qwen-edit-pilot/bin/python'
STATE = ROOT / 'workflow_status.json'


def update(phase, **fields):
    value = dict(phase=phase, time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **fields)
    STATE.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    print(json.dumps(value, ensure_ascii=False), flush=True)


def stage(script, arguments, log_name):
    env = dict(os.environ, PYTHONPATH='/workspace/diffusers-fix/src')
    log = ROOT / log_name
    with log.open('x') as stream:
        subprocess.run([PYTHON, str(ROOT/'scripts'/script), *arguments], cwd='/workspace/diffusers-fix', env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    update('waiting_primary')
    start = time.monotonic()
    while True:
        state = json.loads((ROOT/'status.json').read_text())
        if state['phase'] == 'failed':
            raise RuntimeError(f'Primary failed: {state}')
        if state['phase'] == 'complete':
            break
        if time.monotonic()-start > 7200:
            raise TimeoutError('Primary did not finish')
        time.sleep(5)
    primary = (ROOT/'data/runs.jsonl').read_bytes()
    rows = [json.loads(line) for line in primary.splitlines()]
    assert len(rows) == 102 and len({r['run_id'] for r in rows}) == 102 and all(r['status']=='complete' for r in rows)
    (ROOT/'data/primary_completion_status.json').write_text(json.dumps(state,indent=2))
    update('safe_diagnostic', primary_requests=102, primary_sha256=hashlib.sha256(primary).hexdigest())
    stage('run_local_pilot.py', ['--resume','--safe','--repeats','3'], 'fallback.log')
    full = (ROOT/'data/runs.jsonl').read_bytes()
    assert full[:len(primary)] == primary
    rows = [json.loads(line) for line in full.splitlines()]
    assert len(rows)==138 and len({r['run_id'] for r in rows})==138 and all(r['status']=='complete' for r in rows)
    update('operator_profile', ledger_requests=138, ledger_dit_calls=sum(r['dit_calls'] for r in rows))
    stage('profile_local_step.py', [], 'profile.log')
    update('metrics_analysis')
    stage('analyze_local_pilot.py', ['--device','cuda'], 'analysis.log')
    metrics=json.loads((ROOT/'data/summary.json').read_text())
    assert metrics['status']=='complete' and metrics['counts']['primary_quality_rows_complete']==15
    profiles=[json.loads(line) for line in (ROOT/'diagnostics/profiler_requests.jsonl').read_text().splitlines()]
    assert len(profiles)==2 and all(r['status']=='complete' for r in profiles)
    assert sum(r['actual_dit_calls'] for r in profiles)==16
    update('complete',ledger_requests=138,ledger_dit_calls=sum(r['dit_calls'] for r in rows),extra_profile_requests=2,
           extra_profile_dit_calls=16,total_real_model_requests=140,total_real_model_dit_calls=1120,
           primary_prefix_preserved_sha256=hashlib.sha256(primary).hexdigest())


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        update('failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise
