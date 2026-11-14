# PA3 — Parte 0 e Trilha A

Implementação das duas baselines supervisionadas solicitadas no PDF:

1. CNN pequena, inicializada do zero.
2. ResNet-18 com pesos ImageNet e fine-tuning de todas as camadas.

Ambas usam somente `labels/rotulos_5.csv` para treino. A melhor época é escolhida
pelo F1 macro no `val`; o `test_public` é usado uma única vez para o relatório final,
sem selecionar época ou hiperparâmetros. O pré-processamento preserva a proporção
das imagens, adicionando preenchimento até o tamanho quadrado de entrada.

## Preparação

Extraia o pacote completo de dados para `pa3_dados/`, incluindo `labels/`,
`images/val/` e `images/test_public/`. Os dados permanecem locais e são ignorados
pelo Git.

Instale PyTorch e dependências em um ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

O primeiro uso do ResNet-18 baixa os pesos ImageNet oficiais do torchvision,
caso ainda não estejam no cache local.

## Treino e avaliação

Para treinar as duas versões:

```powershell
python part0.py --model both
```

Para executar uma só:

```powershell
python part0.py --model small_cnn
python part0.py --model resnet18
```

O padrão usa 30 épocas, batch size 16, entrada 224×224 e seed 42. Para ajustar
esses valores, use `--epochs`, `--batch-size`, `--image-size` e `--seed`.
Os dados devem incluir exatamente 105 linhas em `rotulos_5.csv`.

Os resultados vão para `resultados/parte0/`, ignorado pelo Git:

- por modelo: `best.pt`, `history.json` e `test_public_metrics.json`;
- ResNet-18: `curva_treino_validacao.png`, `matriz_confusao.png` e
  `matriz_confusao.csv`;
- resumo de métricas dos modelos executados: `resumo_test_public.json`.

O gráfico da ResNet-18 inclui perda e F1 macro de treino e validação por época.
O checkpoint salvo corresponde à época de maior F1 macro no `val` (desempate
pela menor perda de validação). F1 macro e acurácia de `test_public` são
calculados apenas após essa seleção.

## Dependências

Consulte [`requirements.txt`](requirements.txt).

## Parte 1 — Trilha A (SimCLR)

[`part1a.py`](part1a.py) executa o fluxo da Trilha A com ResNet-18:

1. Pré-treina com SimCLR em todas as 6.505 imagens de `images/train/`, sem usar
   os rótulos nessa etapa. O script treina duas inicializações: pesos ImageNet
   e pesos aleatórios.
2. Faz avaliação linear, congelando o backbone, com três variantes lado a lado:
   ImageNet puro, ImageNet + SimCLR e aleatório + SimCLR.
3. Faz fine-tuning da variante ImageNet + SimCLR usando apenas os 105 exemplos
   de `rotulos_5.csv`.
4. Compara duas políticas de crop: `standard` usa área entre 8% e 100% da
   imagem; `context_preserving` muda apenas o limite inferior para 50%. Essa
   restrição preserva mais contexto, mas não garante que o animal esteja no crop.
   Por padrão, escolhe a política pelo F1 macro do fine-tuning no `val`.

Antes de comparar com o piso, gere as métricas da ResNet-18 da Parte 0; em
seguida, rode o comando completo (duas políticas de crop, duas inicializações
SimCLR e as avaliações downstream):

```powershell
python part0.py --model resnet18
python part1a.py
```

Para executar uma só variante de crop:

```powershell
python part1a.py --augmentation standard
python part1a.py --augmentation context_preserving
```

O script requer o pacote completo: `images/train/` com 6.505 JPGs,
`images/val/`, `images/test_public/` e `labels/rotulos_5.csv`, `labels/val.csv`
e `labels/test_public.csv`. A primeira execução precisa baixar os pesos
ImageNet do torchvision. O pré-treino é computacionalmente intensivo; GPU é
recomendada. É possível ajustar épocas, batch size, dimensão e workers com
`--ssl-epochs`, `--linear-epochs`, `--finetune-epochs`, `--batch-size`,
`--image-size` e `--workers`.
O estado do pré-treino SimCLR é salvo ao fim de cada época; se o processo for
interrompido, execute novamente com os mesmos argumentos para retomá-lo.

Resultados e checkpoints são salvos em `resultados/trilha_a/` (ignorado pelo
Git). `resultados_trilha_a.csv` contém as métricas de avaliação linear e
fine-tuning; `relatorio_trilha_a.json` registra as políticas comparadas e a
escolha feita pelo `val`. O CSV também inclui o piso da Parte 0 quando o arquivo
`resultados/parte0/resnet18/test_public_metrics.json` existe; use
`--floor-metrics` para indicar outro caminho. Sem esse arquivo, o script avisa e
executa a Trilha A sem a linha de comparação. As métricas do `test_public` são
somente para relatório; use o `val` para escolher épocas e aumentação.

## Pasta de resultados para apresentação

Depois de concluir os treinos da Parte 0 e da Trilha A, gere a pasta consolidada
com tabelas, gráficos e relatório:

```powershell
python gerar_relatorio.py
```

O pacote fica em `resultados/entrega/RELATORIO.md`, com métricas em CSV e figuras
copiadas ou geradas a partir dos resultados, incluindo a curva de treino e
validação, matriz de confusão, comparação linear e experimento de aumentação.
`resultados/` fica ignorada pelo Git por conter artefatos e checkpoints grandes.
