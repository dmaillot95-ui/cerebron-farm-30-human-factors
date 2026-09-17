import json, os
from pathlib import Path
from gradio_client import Client

ROLE = os.getenv('ROLE', 'HUMAN_FACTORS_ANALYST')
MODEL = os.getenv('MODEL', 'huggingface-projects/llama-3.2-3B-Instruct')
mission = Path('MISSION.md').read_text(encoding='utf-8')
prompt = f'''You are {ROLE} in CEREBRON Farm 30 Human Factors.
Model endpoint: {MODEL}.

Mission:
{mission}

Produce a concise, evidence-disciplined assessment. Include:
1. task/context/population boundaries
2. cognitive or human-performance mechanisms
3. likely failure modes and confounders
4. measurable variables and metrics
5. human-system interaction risks
6. field-vs-lab validity limits
7. mitigation/design options
8. validation or experiment path
9. unknowns and contradictions
10. final claim ledger: ESTABLISHED / DERIVED / HYPOTHESIS / UNKNOWN

Never infer intent from behavior without evidence. Never treat a simulation, model, benchmark, or lab study as direct field validation.'''

Path('results').mkdir(exist_ok=True)
out = {'role': ROLE, 'model': MODEL, 'success': False, 'text': None, 'error': None}
try:
    c = Client(MODEL)
    out['text'] = c.predict(message=prompt, api_name='/chat')
    out['success'] = True
except Exception as e:
    out['error'] = repr(e)
Path(f'results/{ROLE}.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'role': ROLE, 'success': out['success'], 'error': out['error']}, ensure_ascii=False))