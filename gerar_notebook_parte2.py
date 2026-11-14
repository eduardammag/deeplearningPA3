"""Package the budget experiment as a standalone Kaggle notebook."""
import json
from pathlib import Path
import textwrap

root = Path(__file__).resolve().parent
cells = []


def add(kind, source):
    cell = dict(cell_type=kind, id=f'parte2-{len(cells):02d}', metadata={},
                source=textwrap.dedent(source).strip() + '\n')
    if kind == 'code':
        compile(cell['source'], cell['id'], 'exec')
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add('markdown', '''
# PA3 — Parte 2: curva de orçamento

1. Ative **GPU** e **Internet** no Kaggle.
2. Adicione o Dataset `pa3_dados` completo, como na Parte 1.
3. Adicione também `pa3_resultados_completos.zip` da Parte 1 como Input. Pode estar extraído:
   o notebook também aceita a pasta `resultados/trilha_a_completa`.
4. Execute **Run All**. Os scripts já estão incorporados.

São **10 treinos supervisionados**: piso ImageNet e método ImageNet+SimCLR,
com 1, 5 e 20 rótulos por classe. Seeds 42,43,44 para 5; seed 42 nos demais.
O checkpoint SimCLR é fixo; a variabilidade medida é a do fine-tuning.
Teto do enunciado: **0,686**, sem treino adicional.
Os CSVs oficiais são mantidos; val escolhe a época, test_public só fornece as métricas finais.

Salve os outputs no Kaggle. Se precisar retomar em outra sessão, anexe os outputs salvos e
preencha `RESTORE_FROM` abaixo. A retomada é por época e também pula treinos já concluídos.
''')
add('code', '''
from pathlib import Path
import os, sys, json, shutil, subprocess
import torch
assert torch.cuda.is_available(), 'Ative a GPU antes de continuar.'
print(torch.__version__, torch.cuda.get_device_name(0))
WORK = Path('/kaggle/working/pa3_parte2')
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
DATA_OVERRIDE = None
PART1_ZIP_OVERRIDE = None
RESTORE_FROM = None  # Pasta parte2 de outputs salvos de uma execução anterior.
INPUT = Path('/kaggle/input')
roots = sorted({p.parent for p in INPUT.rglob('classes.csv')
    if (p.parent/'labels/rotulos_20.csv').is_file() and (p.parent/'images/train').is_dir()})
if DATA_OVERRIDE:
    DATA = Path(DATA_OVERRIDE)
else:
    assert len(roots)==1, f'Preencha DATA_OVERRIDE. Pastas encontradas: {roots}'
    DATA = roots[0]
OUTPUT = WORK/'resultados/parte2'
if RESTORE_FROM:
    restore = Path(RESTORE_FROM)
    assert restore.is_dir() and restore.resolve()!=OUTPUT.resolve()
    shutil.copytree(restore, OUTPUT, dirs_exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
zips = list(INPUT.rglob('pa3_resultados_completos.zip'))
PART1_ZIP = Path(PART1_ZIP_OVERRIDE) if PART1_ZIP_OVERRIDE else (zips[0] if len(zips)==1 else None)
EXTRA_ARGS = []
if PART1_ZIP:
    EXTRA_ARGS = ['--part1-zip', str(PART1_ZIP)]
else:
    reports = list(INPUT.rglob('relatorio_trilha_a.json'))
    assert len(reports)==1, 'Adicione o ZIP da Parte 1 ou sua pasta extraída; se houver vários ZIPs, preencha PART1_ZIP_OVERRIDE.'
    report = json.loads(reports[0].read_text())
    assert not report.get('preliminary', True)
    checkpoint = reports[0].parent/report['selected_augmentation']/'imagenet_ssl.pt'
    assert checkpoint.is_file(), checkpoint
    EXTRA_ARGS = ['--ssl-checkpoint', str(checkpoint)]
print('Dados:', DATA, '\\nParte 1:', EXTRA_ARGS)
''')
scripts = {name:(root/name).read_text(encoding='utf-8') for name in ('part0.py','parte2.py')}
add('code', 'SCRIPTS = ' + repr(scripts) + '''
\nfor name, source in SCRIPTS.items():
    (WORK/name).write_text(source, encoding='utf-8')
COMMAND = [sys.executable, '-u', 'parte2.py', '--data-dir', str(DATA),
    '--output-dir', str(OUTPUT), '--epochs', '30', '--batch-size', '16',
    '--image-size', '224', '--workers', '2'] + EXTRA_ARGS
subprocess.run(COMMAND + ['--check-only'], check=True)
''')
add('markdown', '''
## Rodar os 10 treinos

Mesma configuração supervisionada nos dois métodos: AdamW, lr=1e-4, weight decay=1e-4,
batch 16 e 30 épocas. Todas as camadas são ajustadas. O pré-processamento é o da Parte 0.
O SimCLR não é repetido: reutilizamos o checkpoint selecionado na Parte 1 pela validação.
O progresso aparece após cada época. Esta execução refaz a seed 42 para manter o protocolo pareado.
''')
add('code', '''
with (OUTPUT/'execucao.log').open('a', encoding='utf-8') as log:
    process = subprocess.Popen(COMMAND, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        exit_code = process.wait()
        if exit_code:
            raise RuntimeError(f'Treino encerrou com código {exit_code}; veja execucao.log.')
    except BaseException:
        process.terminate()
        process.wait()
        raise
''')
add('code', '''
import pandas as pd
from IPython.display import display, Image, Markdown, FileLink
display(pd.read_csv(OUTPUT/'resumo_orcamento.csv'))
display(Image(filename=str(OUTPUT/'curva_orcamento.png')))
display(pd.read_csv(OUTPUT/'ganho_por_classe.csv'))
display(Image(filename=str(OUTPUT/'ganho_por_classe.png')))
display(Markdown((OUTPUT/'RELATORIO.md').read_text()))
''')
add('markdown', '''
## Exportar

`curva_orcamento.csv` tem as dez linhas individuais exigidas pelo PA. `resumo_orcamento.csv`
traz média e desvio amostral (ddof=1). `ganho_por_classe.csv` calcula a diferença entre métodos
para cada seed, depois resume as três diferenças. Ganho <= 0 significa sem ganho médio positivo.
O desvio descreve variação entre seeds, não intervalo de confiança.

O ZIP completo contém checkpoints, históricos e predições por imagem. Preserve-o também
para as próximas partes. O pacote de apresentação exclui os pesos.
''')
add('code', '''
import zipfile
light = Path('/kaggle/working/pa3_parte2_relatorio.zip')
with zipfile.ZipFile(light, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in OUTPUT.iterdir():
        if path.is_file() and path.suffix in {'.csv','.png','.md','.json'}:
            archive.write(path, path.name)
shutil.make_archive('/kaggle/working/pa3_parte2_completo', 'zip', OUTPUT)
display(FileLink(str(light)))
display(FileLink('/kaggle/working/pa3_parte2_completo.zip'))
''')
notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                  language_info=dict(name='python', version='3.11')))
target = root/'PA3_Parte2_Kaggle.ipynb'
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(target)
