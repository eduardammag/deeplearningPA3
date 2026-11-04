# PA3 — Parte 0

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
