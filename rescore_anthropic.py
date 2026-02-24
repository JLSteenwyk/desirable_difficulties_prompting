import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

INPUTS = [
    Path('final_results/hlegold_full/anthropic__claude-opus-4-6/all_model_predictions.csv'),
    Path('final_results/hlegold_full/anthropic__claude-sonnet-4-6/all_model_predictions.csv'),
]
OUT_DIR = Path('final_results/hlegold_full/audit_outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_ROW_PATH = OUT_DIR / 'anthropic_rescored_rows.csv'
SUMMARY_PATH = OUT_DIR / 'anthropic_rescore_summary.txt'
AMBIG_PATH = OUT_DIR / 'anthropic_top20_ambiguity.csv'

MC_OPTIONS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def normalize_text(s: str) -> str:
    if s is None:
        return ''
    s = s.strip().lower()
    # Remove latex wrappers and markdown markers.
    s = s.replace('\\(', ' ').replace('\\)', ' ')
    s = s.replace('$', ' ').replace('`', ' ')
    s = s.replace('*', ' ')
    # Collapse whitespace and punctuation-ish separators.
    s = re.sub(r'[_\-–—]', ' ', s)
    s = re.sub(r'[^a-z0-9%,./+() ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def parse_mc_choice(row):
    parsed = (row.get('parsed_prediction') or '').strip()
    if re.fullmatch(r'[A-Z]', parsed):
        return parsed, 'parsed_single_letter'

    text = (row.get('model_output') or '').strip()
    if not text:
        return None, 'blank_output'

    patterns = [
        r'(?i)final\s+answer\s*[:\-]?\s*\(?\s*([A-Z])\s*\)?',
        r'(?i)answer\s*[:\-]\s*\(?\s*([A-Z])\s*\)?',
        r'(?i)option\s*([A-Z])\b',
        r'(?i)choose\s*\(?\s*([A-Z])\s*\)?',
        r'(?i)correct\s+answer\s*(?:is|:)\s*\(?\s*([A-Z])\s*\)?',
    ]

    captures = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            captures.append(m.group(1).upper())

    if captures:
        return captures[-1], 'regex_final_answer'

    # Last standalone letter on its own line near the end.
    tail = text[-400:]
    line_hits = re.findall(r'(?m)^\s*\(?([A-Z])\)?\s*$', tail)
    if line_hits:
        return line_hits[-1].upper(), 'tail_line_letter'

    return None, 'no_parse'


def numbers_equivalent(gold_norm: str, out_norm: str) -> bool:
    # Strict numeric equivalence for simple scalar answers.
    g = gold_norm.replace(' ', '')
    o = out_norm.replace(' ', '')
    if re.fullmatch(r'-?\d+(?:\.\d+)?%?', g):
        return g in o
    if re.fullmatch(r'-?\d+/\d+', g):
        return g in o
    return False


def exact_match_semantic(gold, output):
    out_raw = (output or '').strip()
    if not out_raw:
        return 0, 'low', 'Blank output', 'blank_output', 1.0

    gold_norm = normalize_text(gold)
    out_norm = normalize_text(out_raw)

    if not gold_norm:
        return 0, 'low', 'Missing gold', 'missing_gold', 1.0

    # Exact normalized equality.
    if out_norm == gold_norm:
        return 1, 'high', 'Exact normalized match', 'exact_norm_equal', 0.01

    # Gold appears verbatim after normalization.
    if gold_norm in out_norm:
        # Penalize if answer looks buried in long reasoning without explicit final tag.
        conf = 'high' if len(out_norm) < 400 else 'medium'
        return 1, conf, 'Gold string present in output', 'gold_substring', 0.15 if conf == 'high' else 0.3

    # Numeric-specific equivalence.
    if numbers_equivalent(gold_norm, out_norm):
        return 1, 'medium', 'Equivalent numeric answer present', 'numeric_equivalent', 0.35

    # Short yes/no gold answers require explicit standalone signal.
    if gold_norm in {'yes', 'no'}:
        standalone = re.findall(r'\b(yes|no)\b', out_norm)
        if standalone and standalone[-1] == gold_norm:
            return 1, 'medium', 'Final yes/no matches gold', 'yes_no_terminal', 0.4

    return 0, 'high', 'Gold answer not matched', 'no_semantic_match', 0.05


rows_out = []
summary = {
    'total_rows': 0,
    'total_points': 0,
}
model_totals = defaultdict(lambda: {'rows': 0, 'points': 0})
atype_totals = defaultdict(lambda: {'rows': 0, 'points': 0})
model_prompt = defaultdict(lambda: defaultdict(lambda: {'rows': 0, 'points': 0}))
ambiguity = []

for path in INPUTS:
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row['model_id']
            rid = row['id']
            prompt_category = row['prompt_category']
            answer_type = row['answer_type']
            gold = row['gold_answer']
            output = row['model_output'] or ''

            score = 0
            confidence = 'medium'
            reason = 'Unscored'
            rule = ''
            ambiguity_score = 0.5

            if answer_type == 'multipleChoice':
                choice, parse_rule = parse_mc_choice(row)
                if choice is None:
                    score = 0
                    confidence = 'high'
                    reason = 'No valid selected choice found'
                    rule = parse_rule
                    ambiguity_score = 0.2
                else:
                    if choice == (gold or '').strip():
                        score = 1
                        confidence = 'high'
                        reason = f'Selected {choice} equals gold'
                        ambiguity_score = 0.02
                    else:
                        score = 0
                        confidence = 'high'
                        reason = f'Selected {choice} not gold {gold.strip()}'
                        ambiguity_score = 0.02
                    rule = parse_rule
            elif answer_type == 'exactMatch':
                score, confidence, reason, rule, ambiguity_score = exact_match_semantic(gold, output)
            else:
                score = 0
                confidence = 'low'
                reason = 'Unsupported answer_type'
                rule = 'unsupported'
                ambiguity_score = 0.9

            reason_short = reason.strip()
            if len(reason_short.split()) > 20:
                reason_short = ' '.join(reason_short.split()[:20])

            rows_out.append({
                'model_id': model_id,
                'id': rid,
                'prompt_category': prompt_category,
                'answer_type': answer_type,
                'score': score,
                'confidence': confidence,
                'reason_short': reason_short,
                '_rule': rule,
                '_gold': gold,
                '_output': output,
                '_existing': (row.get('is_correct') or '').strip(),
                '_ambiguity': ambiguity_score,
            })

            summary['total_rows'] += 1
            summary['total_points'] += score
            model_totals[model_id]['rows'] += 1
            model_totals[model_id]['points'] += score
            atype_totals[answer_type]['rows'] += 1
            atype_totals[answer_type]['points'] += score
            model_prompt[model_id][prompt_category]['rows'] += 1
            model_prompt[model_id][prompt_category]['points'] += score

# Top likely ambiguity/misgraded rows.
for r in rows_out:
    # Priority: low/medium confidence, exactMatch no match on long output, or disagreement with existing label.
    disagree = False
    if r['_existing'] in {'0', '1'} and str(r['score']) != r['_existing']:
        disagree = True
    weight = r['_ambiguity']
    if r['confidence'] != 'high':
        weight += 0.3
    if r['answer_type'] == 'exactMatch' and r['score'] == 0 and len(r['_output']) > 300:
        weight += 0.35
    if disagree:
        weight += 0.5
    ambiguity.append((weight, disagree, r))

ambiguity.sort(key=lambda x: x[0], reverse=True)
top20 = ambiguity[:20]

# Write full per-row CSV.
with PER_ROW_PATH.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(
        f,
        fieldnames=['model_id', 'id', 'prompt_category', 'answer_type', 'score', 'confidence', 'reason_short']
    )
    w.writeheader()
    for r in rows_out:
        w.writerow({k: r[k] for k in w.fieldnames})

# Write top 20 ambiguity file.
with AMBIG_PATH.open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['model_id', 'id', 'prompt_category', 'answer_type', 'score', 'confidence', 'likely_misgraded', 'rationale'])
    for _, disagree, r in top20:
        rationale = r['reason_short']
        if disagree:
            rationale += '; differs from existing is_correct'
        w.writerow([
            r['model_id'], r['id'], r['prompt_category'], r['answer_type'], r['score'], r['confidence'],
            'yes' if disagree else 'no', rationale[:180]
        ])

# Human-readable summary file.
with SUMMARY_PATH.open('w', encoding='utf-8') as f:
    total_rows = summary['total_rows']
    total_points = summary['total_points']
    f.write(f'total_rows_audited: {total_rows}\n')
    f.write(f'total_points: {total_points}\n')
    f.write(f'total_possible_points: {total_rows}\n')
    f.write(f'overall_accuracy: {total_points/total_rows:.6f}\n\n')

    f.write('per_model\n')
    for m, d in sorted(model_totals.items()):
        acc = d['points'] / d['rows'] if d['rows'] else 0
        f.write(f'{m}: points={d["points"]}, total={d["rows"]}, accuracy={acc:.6f}\n')

    f.write('\nper_answer_type\n')
    for t, d in sorted(atype_totals.items()):
        acc = d['points'] / d['rows'] if d['rows'] else 0
        f.write(f'{t}: points={d["points"]}, total={d["rows"]}, accuracy={acc:.6f}\n')

    f.write('\nby_prompt_category_per_model\n')
    for m in sorted(model_prompt):
        f.write(f'{m}\n')
        for pc, d in sorted(model_prompt[m].items()):
            acc = d['points'] / d['rows'] if d['rows'] else 0
            f.write(f'  {pc}: points={d["points"]}, total={d["rows"]}, accuracy={acc:.6f}\n')

print('Wrote:', PER_ROW_PATH)
print('Wrote:', SUMMARY_PATH)
print('Wrote:', AMBIG_PATH)
print('rows:', summary['total_rows'], 'points:', summary['total_points'])
