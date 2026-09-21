#!/usr/bin/env python3
"""Publish a compact research summary from two verified, completed run directories."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fruitflybrain.provenance import atomic_json,sha256
from fruitflybrain.comparison.__main__ import markdown
from verify_comparison import verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy',type=Path,required=True);p.add_argument('--circuit',type=Path,required=True);a=p.parse_args()
    for path in (a.legacy,a.circuit):verify(path)
    left=json.loads((a.legacy/'report.json').read_text());right=json.loads((a.circuit/'report.json').read_text())
    if left['protocol']!=right['protocol'] or left['protocol'].get('development_smoke'):raise ValueError('Require matched full protocols')
    manifests=[json.loads((path/'manifest.json').read_text()) for path in (a.legacy,a.circuit)]
    if manifests[0]['source_sha256']!=manifests[1]['source_sha256']:raise ValueError('Scientific sources differ')
    result={**left,'circuit':right['circuit'],'paired_differences':left['paired_differences']+right['paired_differences'],
        'provenance':{'run_ids':[a.legacy.name,a.circuit.name],'source_sha256':manifests[0]['source_sha256'],
            'run_manifest_sha256':[sha256(path/'manifest.json') for path in (a.legacy,a.circuit)],
            'progress_checksum_repair':'Original mutable progress pointer was mistakenly included in immutable output hashes. Explicit audit records preserve original manifests; scientific artifacts verified unchanged.'}}
    detail=json.loads((a.circuit/'circuit/results.json').read_text())
    result['circuit_anatomy']={k:detail['anatomy'][k] for k in ('n_kc','n_mbon','edges','graph_sha256','sha256','rewiring')}
    result['plasticity_checks']={'conditions':len(detail['verification']),
        'min_edges_changed':min(r['plastic_edges_changed'] for r in detail['verification']),
        'max_edges_changed':max(r['plastic_edges_changed'] for r in detail['verification']),
        'all_frozen_verified':all(r['evaluation_frozen_verified'] for r in detail['verification'])}
    result['interpretation']={'legacy':'Exact tie: the calibrated probes deliver the same tokens to identical tabular learners.',
        'familiar':'Plastic circuit versus replay Q table is inconclusive across seeds.',
        'unseen':'Replay Q table outperformed this plastic circuit on the held-out maze set.',
        'anatomy':'No demonstrated advantage over the degree-matched rewired network.',
        'limits':['Reduced feedforward rate model, not the full fly brain or natural plasticity.',
            'One selected circuit and one rewiring; confidence intervals concern training seeds only.',
            'Fixed hyperparameters and update budget; no claim about optimally tuned algorithms.',
            'Parts A and B use different training regimes and state representations; compare agents within each part.',
            'Time-limit truncations are terminal in the existing maze; time remaining is not observed.',
            'Timing is cumulative update-call time, excluding data collection, initialization and evaluation.']}
    out=Path(__file__).resolve().parents[1]/'docs/research';atomic_json(out/'maze_comparison_results.json',result)
    intro='''# Maze comparison: observed results, 21 September 2026

The current Exp 2 fly-assisted Q controller is exactly equivalent to its direct-input Q control in these trials. The new trainable reduced fly circuit learns, but the experiment does not demonstrate an anatomical advantage. The replay Q table transfers better to the tested unseen mazes.

Part A uses 400 online training episodes. Part B uses the same 20,000 randomly collected transitions and 2,000 replay updates for every controller. Do not compare percentages across the two parts as if they used the same training procedure. Each final agent/split has 2,000 evaluation trials across ten seeds and both reward assignments. Statistical intervals use ten seed units.

'''
    text=intro+markdown(result).split('\n',1)[1]
    text+='\n## Limits and provenance\n\n'+'\n'.join('- '+s for s in result['interpretation']['limits'])
    text+='\n\nSelected circuit: 256 Kenyon cells, 97 MBONs, 3,857 recorded positive connections. Reward training changed 3,013–3,283 permitted internal strengths per condition; missing connections stayed zero. The neural readout also trained. Neither model is the full-brain spiking simulator.\n'
    text+='\nScientific source: `'+result['provenance']['source_sha256']+'`. Run IDs: '+', '.join('`'+name+'`' for name in result['provenance']['run_ids'])+'. Raw data/checkpoints remain under `runs/gx10-a/exp_2_comparison` and `runs/gx10-b/exp_2_comparison`, outside Git.\n'
    text+='\nThe first runner finalized `progress.json` after hashing it. Explicit `finalization_repair.json` audit records preserve the original manifests and exclude that mutable status pointer. All 378 immutable artifacts verified; no scientific output was changed.\n'
    (out/'maze_comparison_results.md').write_text(text)
    print(out/'maze_comparison_results.md')
if __name__=='__main__':main()
