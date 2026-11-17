"""Grad-CAM gallery for three errors and five correct predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision.models import resnet18


def choose(rows, correct, count):
    candidates = sorted([r for r in rows if (r['actual']==r['predicted']) == correct],
                        key=lambda r: (-float(r['confidence']), r['image_id']))
    selected, classes = [], set()
    for row in candidates:
        if row['actual'] not in classes:
            selected.append(row)
            classes.add(row['actual'])
        if len(selected) == count:
            return selected
    for row in candidates:
        if row not in selected:
            selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError('Exemplos insuficientes para montar a galeria.')


def make_cam(model, tensor, target):
    captured = []
    hook = model.layer4.register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        logits = model(tensor)
        activation = captured[0]
        gradient, = torch.autograd.grad(logits[0, target], activation)
        weights = gradient.mean(dim=(2,3), keepdim=True)
        cam = (weights * activation).sum(1, keepdim=True).relu()
        cam = nn.functional.interpolate(cam, size=tensor.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam[0,0].detach().cpu().numpy()
        cam = (cam-cam.min()) / max(float(cam.max()-cam.min()), 1e-12)
        return cam, logits.detach().softmax(1)[0].cpu().numpy()
    finally:
        hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('parte2/resultados/simclr/rotulos_5/seed_42'))
    parser.add_argument('--data-dir', type=Path, default=Path('pa3_dados'))
    parser.add_argument('--output-dir', type=Path, default=Path('parte3/resultados'))
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    checkpoint = torch.load(args.model_dir/'best.pt', map_location='cpu', weights_only=True)
    classes = checkpoint['class_names']
    size = checkpoint['image_size']
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with (args.model_dir/'predictions.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    selection = [('erro',r) for r in choose(rows, False, 3)] + [('acerto',r) for r in choose(rows, True, 5)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(checkpoint['mean']).view(3,1,1)
    std = torch.tensor(checkpoint['std']).view(3,1,1)
    metadata = []
    panels = []
    for index, (kind,row) in enumerate(selection, 1):
        image_id = row['image_id']
        with Image.open(args.data_dir/'images/test_public'/f'{image_id}.jpg') as source:
            original = source.convert('RGB')
            input_image = ImageOps.pad(original, (size,size), method=Image.Resampling.BILINEAR)
        pixels = np.asarray(input_image, dtype=np.float32)/255
        tensor = ((torch.from_numpy(pixels).permute(2,0,1)-mean)/std).unsqueeze(0)
        target = classes.index(row['predicted'])
        cam, probabilities = make_cam(model,tensor,target)
        assert int(probabilities.argmax()) == target, 'Predição diferente do CSV.'
        assert abs(float(probabilities[target])-float(row['confidence'])) < 1e-4, 'Confiança diferente do CSV.'
        name = f'{index:02d}_{kind}_{image_id}'
        original.save(args.output_dir/f'{name}_original.png')
        np.save(args.output_dir/f'{name}_gradcam.npy', cam)
        fig, axes = plt.subplots(1,3,figsize=(12,4.8))
        axes[0].imshow(original)
        axes[0].set_title('Imagem original')
        axes[1].imshow(input_image)
        axes[1].set_title(f'Entrada do modelo ({size} × {size})')
        axes[2].imshow(input_image)
        axes[2].imshow(cam,cmap='jet',alpha=.48,vmin=0,vmax=1)
        axes[2].set_title('Grad-CAM da classe prevista')
        for ax in axes:
            ax.axis('off')
        title = f'{kind.upper()} | Real: {row["actual"]} | Prevista: {row["predicted"]} | Confiança: {probabilities[target]:.1%}'
        fig.suptitle(title,fontsize=10)
        fig.text(.5,.02,'Vermelho: maior saliência relativa; azul: menor. Mapa normalizado por imagem.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.94))
        fig.savefig(args.output_dir/f'{name}.png',dpi=160)
        plt.close(fig)
        overlay = np.clip(.55*pixels+.45*plt.get_cmap('jet')(cam)[...,:3],0,1)
        panels.append((kind,row,overlay,cam))
        metadata.append(dict(numero=index,tipo=kind,**{**row,'confidence':float(probabilities[target])},
            original_width=original.width,original_height=original.height,
            image_size=size,painel=f'{name}.png',original=f'{name}_original.png',mapa=f'{name}_gradcam.npy'))
        print(title, flush=True)
    for kind, title in [('erro','Três erros'),('acerto','Cinco acertos')]:
        subset = [p for p in panels if p[0]==kind]
        fig,axes = plt.subplots(len(subset),2,figsize=(9,3.8*len(subset)),squeeze=False)
        for pair, (_,row,overlay,cam) in zip(axes,subset):
            with Image.open(args.data_dir/'images/test_public'/f'{row["image_id"]}.jpg') as source:
                pair[0].imshow(source)
            pair[1].imshow(overlay)
            pair[0].set_title(f'Real: {row["actual"]}',fontsize=10)
            pair[1].set_title(f'Prevista: {row["predicted"]} ({float(row["confidence"]):.1%})',fontsize=10)
            for ax in pair:
                ax.axis('off')
        fig.suptitle(title+' — imagem e Grad-CAM',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.97))
        fig.savefig(args.output_dir/f'galeria_{kind}s.png',dpi=160)
        plt.close(fig)
    (args.output_dir/'exemplos.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    with (args.output_dir/'exemplos.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    provenance = dict(checkpoint=str(args.model_dir/'best.pt'),
        checkpoint_sha256=hashlib.sha256((args.model_dir/'best.pt').read_bytes()).hexdigest(),
        config=checkpoint['config'], best_epoch=checkpoint['best_epoch'],
        selection='Highest-confidence predictions, distinct true classes within errors/correct cases; qualitative diagnostic only.',
        saliency='Grad-CAM of predicted class logit at ResNet layer4; per-image min-max normalization; bilinear resize from 7x7.',
        limitations='Softmax confidence is uncalibrated; saliency is coarse and does not establish causal dependence or animal segmentation.')
    (args.output_dir/'proveniencia.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf-8')
    diagnoses_path = Path(__file__).with_name('parte3_diagnosticos.json')
    diagnoses = json.loads(diagnoses_path.read_text(encoding='utf-8')) if diagnoses_path.is_file() else {}
    lines = ['# Parte 3 — Galeria de falhas e atalhos', '',
        f'Modelo analisado: `{args.model_dir}/best.pt`, melhor época {checkpoint["best_epoch"]}.',
        'ResNet-18, ImageNet + SimCLR com crop de 50%, fine-tuning com 5 rótulos por classe, seed 42.',
        'Mantemos a seed 42 predefinida, sem escolher o modelo pelo resultado do teste.', '',
        'Seleção: três erros e cinco acertos de maior confiança, evitando repetir a classe verdadeira dentro de cada grupo. '
        'A seleção serve ao diagnóstico qualitativo e não é uma amostra representativa do teste.', '',
        'Cada painel mostra a foto original, a entrada com padding e o Grad-CAM da classe prevista. '
        'Vermelho indica maior saliência relativa; azul, menor. O mapa é calculado em layer4 (7×7) e ampliado para 224×224. '
        'O padding preto também faz parte da entrada e pode receber saliência. As cores são normalizadas separadamente por imagem.', '',
        '## Três erros', '', '![Três erros](galeria_erros.png)', '']
    for item in metadata:
        if item['numero']==4:
            lines += ['## Cinco acertos: animal ou fundo?', '', '![Cinco acertos](galeria_acertos.png)', '']
        lines += [f'### {item["numero"]}. {item["tipo"].capitalize()} — {item["image_id"]}', '',
            f'Classe verdadeira: **{item["actual"]}**. Prevista: **{item["predicted"]}**. Confiança softmax: **{item["confidence"]:.2%}**.', '',
            f'![Painel {item["numero"]}]({item["painel"]})', '',
            diagnoses.get(item['image_id'], 'Diagnóstico visual pendente para este exemplo.'), '']
    lines += ['## O que os exemplos mostram', '',
        'Nos três erros, os picos de saliência aparecem sobretudo em elementos do cenário ou no limite inferior/padding, '
        'em vez de se concentrarem claramente no animal. Nos cinco acertos também há forte saliência em fundo e objetos do ambiente. '
        'Assim, acertar a classe não basta para afirmar que a rede reconheceu a anatomia do animal. '
        'Os casos são compatíveis com uso de atalhos de contexto, mas Grad-CAM isoladamente não prova dependência causal.', '',
        'A confiança softmax não foi calibrada: 96,69% em uma previsão errada não é uma probabilidade confiável de acerto. '
        'A resolução espacial do mapa é baixa; não delimitamos os animais nem estimamos sua área como medida objetiva. '
        'Oito exemplos não permitem concluir que todos os acertos ou erros seguem esse padrão.', '',
        '## Reproduzir', '', 'Na raiz do projeto: `python -u parte3/parte3.py`.', '',
        '`exemplos.csv` e `exemplos.json` registram classes, confiança e arquivos. '
        '`proveniencia.json` registra checkpoint, hash e regra de seleção. '
        'Os arquivos `_gradcam.npy` guardam os mapas numéricos e `_original.png` guardam as fotos.', '',
        'Os diagnósticos de inspeção visual são mantidos em `parte3_diagnosticos.json`, ao lado do script.']
    (args.output_dir/'RELATORIO.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Galeria salva em',args.output_dir)


if __name__ == '__main__':
    main()
