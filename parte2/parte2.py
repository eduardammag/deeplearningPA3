"""Budget curves: matched ImageNet and ImageNet+SimCLR fine-tuning runs."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'parte0'))
from part0 import CameraTrapDataset, load_classes, MEAN, STD

CEILING = 0.686
SEEDS = (42, 43, 44)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def locate_ssl(args):
    if args.ssl_checkpoint:
        return args.ssl_checkpoint
    assert args.part1_zip and args.part1_zip.is_file(), 'Informe --part1-zip ou --ssl-checkpoint.'
    with zipfile.ZipFile(args.part1_zip) as archive:
        report = json.loads(archive.read('resultados/trilha_a_completa/relatorio_trilha_a.json'))
        assert not report.get('preliminary', True), 'A Parte 1 precisa estar completa.'
        crop = report['selected_augmentation']
        target = args.output_dir / 'source' / crop / 'imagenet_ssl.pt'
        target.parent.mkdir(parents=True, exist_ok=True)
        # Read one known member; never extract arbitrary ZIP paths.
        target.write_bytes(archive.read(f'resultados/trilha_a_completa/{crop}/imagenet_ssl.pt'))
        write_json(args.output_dir / 'source' / 'parte1_report.json', report)
    return target


@torch.inference_mode()
def evaluate(model, loader, device, class_count):
    model.eval()
    actual, predicted, confidence = [], [], []
    loss = 0.0
    for images, labels in loader:
        labels = labels.to(device)
        logits = model(images.to(device))
        loss += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
        conf, pred = logits.softmax(1).max(1)
        actual.extend(labels.cpu().tolist())
        predicted.extend(pred.cpu().tolist())
        confidence.extend(conf.cpu().tolist())
    return dict(loss=loss / len(actual),
        macro_f1=float(f1_score(actual, predicted, labels=list(range(class_count)), average='macro', zero_division=0)),
        accuracy=float(accuracy_score(actual, predicted)), actual=actual, predicted=predicted, confidence=confidence)


def run_one(args, datasets, classes, ssl_state, method, budget, seed, common):
    output = args.output_dir / method / f'rotulos_{budget}' / f'seed_{seed}'
    output.mkdir(parents=True, exist_ok=True)
    config = dict(**common, method=method, budget=budget, seed=seed)
    if (output / 'metrics.json').is_file():
        metrics = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
        if metrics['config'] != config:
            raise ValueError(f'Configuração diferente em {output}; use outra pasta de saída.')
        if (output / 'predictions.csv').is_file() and (output / 'best.pt').is_file():
            print('Já concluído:', output, flush=True)
            return metrics

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the classifier identically for each paired seed, independent of backbone loading.
    classifier = nn.Linear(512, len(classes))
    model = resnet18(weights=ResNet18_Weights.DEFAULT if method == 'piso' else None)
    if method == 'simclr':
        model.fc = nn.Identity()
        model.load_state_dict(ssl_state['backbone_state_dict'], strict=True)
    model.fc = classifier
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loaders = {split: DataLoader(ds, batch_size=args.batch_size, shuffle=split == 'train',
        num_workers=args.workers, pin_memory=device.type == 'cuda',
        generator=generator if split == 'train' else None) for split, ds in datasets.items()}
    history, best_f1, best_loss, best_epoch, start = [], -1.0, float('inf'), 0, 1
    resume_path = output / 'resume.pt'
    if resume_path.is_file():
        saved = torch.load(resume_path, map_location='cpu', weights_only=True)
        if saved['config'] != config:
            raise ValueError(f'Checkpoint incompatível: {resume_path}')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        history = saved['history']
        best_f1, best_loss, best_epoch = saved['best_f1'], saved['best_loss'], saved['best_epoch']
        start = saved['epoch'] + 1
        torch.set_rng_state(saved['torch_rng'])
        generator.set_state(saved['loader_rng'])
        if device.type == 'cuda' and saved['cuda_rng']:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print(f'Retomando {method} K={budget} seed={seed} época={start}', flush=True)

    for epoch in range(start, args.epochs + 1):
        model.train()
        train_loss, examples = 0.0, 0
        for images, labels in loaders['train']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            examples += len(labels)
        val = evaluate(model, loaders['val'], device, len(classes))
        history.append(dict(epoch=epoch, train_loss=train_loss / examples,
                            val_loss=val['loss'], val_macro_f1=val['macro_f1']))
        if (val['macro_f1'], -val['loss']) > (best_f1, -best_loss):
            best_f1, best_loss, best_epoch = val['macro_f1'], val['loss'], epoch
            torch.save(dict(model_state_dict=model.state_dict(), class_names=classes,
                image_size=args.image_size, mean=MEAN, std=STD, config=config, best_epoch=epoch), output / 'best.pt')
        write_json(output / 'history.json', history)
        temporary = output / 'resume.pt.tmp'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
            epoch=epoch, history=history, best_f1=best_f1, best_loss=best_loss, best_epoch=best_epoch,
            torch_rng=torch.get_rng_state(), loader_rng=generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []), temporary)
        temporary.replace(resume_path)
        print(f'{method} K={budget} seed={seed} época {epoch}/{args.epochs}: '
              f'loss treino={train_loss/examples:.4f}, F1 val={val["macro_f1"]:.4f}', flush=True)

    best = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(best['model_state_dict'])
    test = evaluate(model, loaders['test_public'], device, len(classes))
    predictions = [dict(image_id=image_id, actual=classes[actual], predicted=classes[pred], confidence=conf)
        for (image_id, _), actual, pred, conf in zip(datasets['test_public'].rows,
            test['actual'], test['predicted'], test['confidence'])]
    write_csv(output / 'predictions.csv', predictions)
    per_class = f1_score(test['actual'], test['predicted'], labels=list(range(len(classes))), average=None, zero_division=0)
    metrics = dict(method=method, budget=budget, seed=seed, macro_f1=test['macro_f1'],
        accuracy=test['accuracy'], best_epoch=best_epoch, val_macro_f1=best_f1,
        per_class_f1=per_class.tolist(), config=config)
    write_json(output / 'metrics.json', metrics)
    resume_path.unlink(missing_ok=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return metrics


def summarize(output, results, classes):
    write_csv(output / 'curva_orcamento.csv', [dict(metodo=r['method'], orcamento=r['budget'],
        numero_rotulos=r['budget'] * len(classes), seed=r['seed'], f1_macro=r['macro_f1'],
        acuracia=r['accuracy'], melhor_epoca=r['best_epoch']) for r in results])
    summary = []
    for method in ('piso', 'simclr'):
        for budget in (1, 5, 20):
            runs = [r for r in results if r['method'] == method and r['budget'] == budget]
            if not runs:
                continue
            summary.append(dict(metodo=method, orcamento=budget, numero_rotulos=budget * len(classes),
                seeds=len(runs), f1_media=float(np.mean([r['macro_f1'] for r in runs])),
                f1_desvio=float(np.std([r['macro_f1'] for r in runs], ddof=1)) if len(runs)>1 else 0.0,
                acuracia_media=float(np.mean([r['accuracy'] for r in runs])),
                acuracia_desvio=float(np.std([r['accuracy'] for r in runs], ddof=1)) if len(runs)>1 else 0.0))
    write_csv(output / 'resumo_orcamento.csv', summary)
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, label in [('piso', 'Piso: ImageNet + fine-tuning'), ('simclr', 'Método: ImageNet + SimCLR + fine-tuning')]:
        rows = [r for r in summary if r['metodo'] == method]
        ax.errorbar([r['numero_rotulos'] for r in rows], [r['f1_media'] for r in rows],
                    yerr=[r['f1_desvio'] for r in rows], marker='o', capsize=5, label=label)
    ax.axhline(CEILING, linestyle='--', color='gray', label='Teto de referência: 0,686')
    ax.set(xscale='log', xlabel='Número total de rótulos (escala log)', ylabel='F1 macro no test_public', ylim=(0, 1))
    ax.set_xticks([21,105,420], ['21 (1/classe)', '105 (5/classe)', '420 (20/classe)'])
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'curva_orcamento.png', dpi=180)
    plt.close(fig)
    paired = {method: {r['seed']: r for r in results if r['method'] == method and r['budget'] == 5}
              for method in ('piso', 'simclr')}
    seeds = sorted(set(paired['piso']) & set(paired['simclr']))
    if len(seeds) != 3:
        return
    piso = np.array([paired['piso'][s]['per_class_f1'] for s in seeds])
    simclr = np.array([paired['simclr'][s]['per_class_f1'] for s in seeds])
    delta = simclr - piso
    gains = [dict(classe=name, piso_f1_media=float(piso[:, i].mean()),
        metodo_f1_media=float(simclr[:, i].mean()), ganho_medio=float(delta[:, i].mean()),
        ganho_desvio=float(delta[:, i].std(ddof=1)),
        seeds_com_ganho=int((delta[:, i] > 0).sum())) for i, name in enumerate(classes)]
    gains.sort(key=lambda r:r['ganho_medio'], reverse=True)
    write_csv(output / 'ganho_por_classe.csv', gains)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([r['classe'] for r in gains], [r['ganho_medio'] for r in gains],
        xerr=[r['ganho_desvio'] for r in gains], color=['seagreen' if r['ganho_medio']>0 else 'indianred' for r in gains])
    ax.axvline(0, color='black', linewidth=1)
    ax.invert_yaxis()
    ax.set(xlabel='Ganho de F1 por classe: SimCLR − piso (média ± desvio, 3 seeds)', title='Orçamento de 5 rótulos por classe')
    fig.tight_layout()
    fig.savefig(output / 'ganho_por_classe.png', dpi=180)
    plt.close(fig)
    lines = ['# Parte 2 — Curva de orçamento', '',
        'Seeds 42, 43 e 44 no orçamento 5; seed 42 nos demais. Desvio amostral (ddof=1).',
        'As seeds alteram o treino, mantendo os CSVs oficiais e o checkpoint SimCLR fixos.',
        'Os dois métodos usam o mesmo pré-processamento, otimizador, taxa de aprendizado e seleção pelo val.',
        'Teto fornecido pelo monitor: 0,686; referência, não limite teórico.', '',
        '| Método | Rótulos/classe | F1 macro (média ± desvio) | Acurácia (média ± desvio) |',
        '|---|---:|---:|---:|']
    lines += [f"| {r['metodo']} | {r['orcamento']} | {r['f1_media']:.4f} ± {r['f1_desvio']:.4f} | {r['acuracia_media']:.4f} ± {r['acuracia_desvio']:.4f} |" for r in summary]
    positive = [r for r in gains if r['ganho_medio']>0]
    nonpositive = [r['classe'] for r in gains if r['ganho_medio']<=0]
    lines += ['', 'Maiores ganhos médios positivos: ' + (', '.join(f"{r['classe']} ({r['ganho_medio']:+.4f})" for r in positive[:5]) or 'nenhuma classe') + '.',
        '', 'Sem ganho médio positivo: ' + (', '.join(nonpositive) or 'nenhuma') + '.', '',
        'O ganho é calculado entre seeds pareadas. Três seeds não demonstram significância estatística; classes com poucos exemplos no teste podem variar bastante.',
        'As predições individuais e os F1 de todas as classes ficam nos CSVs/JSONs para auditoria.']
    (output / 'RELATORIO.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('pa3_dados'))
    parser.add_argument('--part1-zip', type=Path, default=Path('parte1trilhaA/resultados/pa3_resultados_completos.zip'))
    parser.add_argument('--ssl-checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('parte2/resultados'))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args()
    if args.epochs<1 or args.batch_size<2 or args.image_size<32 or args.workers<0 or args.lr<=0:
        parser.error('Argumentos de treino inválidos.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classes = load_classes(args.data_dir / 'classes.csv')
    ssl_path = locate_ssl(args)
    state = torch.load(ssl_path, map_location='cpu', weights_only=True)
    assert state['initialization'] == 'imagenet' and len(state['ssl_loss']) == state['epochs']
    assert state['image_size'] == args.image_size, 'Mantenha a resolução da Parte 1.'
    all_datasets = {}
    for budget in (1,5,20):
        ds = dict(train=CameraTrapDataset(args.data_dir/'images/train', args.data_dir/f'labels/rotulos_{budget}.csv', classes, args.image_size),
            val=CameraTrapDataset(args.data_dir/'images/val', args.data_dir/'labels/val.csv', classes, args.image_size),
            test_public=CameraTrapDataset(args.data_dir/'images/test_public', args.data_dir/'labels/test_public.csv', classes, args.image_size))
        assert Counter(label for _,label in ds['train'].rows) == Counter({i:budget for i in range(len(classes))})
        ids = [{row[0] for row in ds[split].rows} for split in ('train','val','test_public')]
        assert all(not ids[i] & ids[j] for i in range(3) for j in range(i+1,3)), 'Sobreposição entre splits.'
        all_datasets[budget] = ds
    common = dict(epochs=args.epochs, batch_size=args.batch_size, image_size=args.image_size,
        lr=args.lr, weight_decay=1e-4, ssl_sha256=sha256(ssl_path),
        label_sha256={p.name:sha256(p) for p in (args.data_dir/'labels').glob('*.csv')},
        classes=classes, selection='val_macro_f1_then_val_loss', seeds=list(SEEDS),
        ssl_crop=state['augmentation'], device='cuda' if torch.cuda.is_available() else 'cpu')
    write_json(args.output_dir/'config.json', common)
    print('Pré-checagem OK. GPU:', torch.cuda.is_available(), 'Checkpoint:', ssl_path, flush=True)
    print('10 treinos supervisionados; teto =', CEILING, flush=True)
    if args.check_only:
        return
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError('GPU ausente. Execute no Kaggle, ou use --allow-cpu explicitamente.')
    results = []
    for budget in (1,5,20):
        for seed in SEEDS if budget == 5 else (42,):
            for method in ('piso','simclr'):
                results.append(run_one(args, all_datasets[budget], classes, state, method, budget, seed, common))
                summarize(args.output_dir, results, classes)
    print('Concluído:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
