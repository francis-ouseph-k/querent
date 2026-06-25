import json
import re

jsonl_file = r'D:\work\CONFIDENTAIL\KREUPASANAM\digital-evaluation_ai\nl_to_sql\data\output\batch-run-output-20260622_125008_607809.jsonl'

errors = []
successes = []

with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data['Result'] == 'Success':
            successes.append(data)
        else:
            errors.append(data)

with open('summary.txt', 'w', encoding='utf-8') as out:
    out.write(f'Total Errors: {len(errors)}\n')
    out.write(f'Total Successes: {len(successes)}\n\n')
    
    out.write('--- ERRORS ---\n')
    for e in errors:
        out.write(f"Q{e['QNum']}: {e['Question']}\n")
        out.write(f"Error: {e['Error Message']}\n\n")

    out.write('--- SUCCESSES (Sample) ---\n')
    for s in successes[:20]:
        out.write(f"Q{s['QNum']}: {s['Question']}\n")
        out.write(f"Query: {s['Generated query']}\n\n")
