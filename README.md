# PA3 — classificação de imagens

Repositório do trabalho de classificação de imagens de armadilhas fotográficas em 21 classes. O modelo entregue é uma ResNet-18; a inferência funciona em CPU e sem acesso à internet.

## Inferência do modelo entregue

Requer Python 3.10 ou superior. Instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

Para classificar uma pasta de imagens:

```powershell
python predict.py --imagens PASTA --saida preds.csv
```

O script carrega `modelo_entregue.pt` e `classes.csv` da raiz do repositório, imprime o número de parâmetros e grava as colunas `image_id,label`. A monitoria pode passar ao argumento `--imagens` a pasta do `test_private`; esse conjunto é mantido pela monitoria e não precisa estar disponível localmente.

## Organização

- `parte0/`: baselines supervisionadas; resultados em `parte0/resultados/`.
- `parte1trilhaA/`: notebook da trilha SimCLR; relatórios e arquivos de execução em `parte1trilhaA/resultados/`.
- `parte2/`: experimento da curva de orçamento; resultados em `parte2/resultados/`.
- `parte3/`: diagnósticos Grad-CAM; resultados em `parte3/resultados/`.
- `pa3_dados/`: dados locais do trabalho (não incluídos no Git).
- `entrega/curva_orcamento.csv`: cópia do CSV da curva para entrega.

## Treino e avaliação

Para reproduzir as baselines da Parte 0 com os dados completos:

```powershell
python parte0/part0.py --data-dir pa3_dados
```

Para reproduzir os dez treinos da Parte 2, é necessário o checkpoint SimCLR produzido na Parte 1A. Pode-se fornecer o ZIP completo em `parte1trilhaA/resultados/pa3_resultados_completos.zip`:

```powershell
python parte2/parte2.py --data-dir pa3_dados --part1-zip parte1trilhaA/resultados/pa3_resultados_completos.zip --output-dir parte2/resultados
```

O treinamento da Parte 2 requer PyTorch com CUDA para um tempo de execução razoável; GPU é recomendada. Para verificar os arquivos sem treinar:

```powershell
python parte2/parte2.py --check-only
```

Os arquivos ZIP e checkpoints de experimentos não são versionados por serem grandes. Se você clonou o repositório, obtenha o ZIP da execução completa da Parte 1A ou indique diretamente o checkpoint SimCLR com `--ssl-checkpoint CAMINHO/imagenet_ssl.pt`.

Para reproduzir a galeria da Parte 3, com resultados da Parte 2 e dados disponíveis:

```powershell
python parte3/parte3.py
```

O relatório das Partes 0 e 1A está em `parte1trilhaA/resultados/relatorio/RELATORIO.md`; a Parte 2 tem `parte2/PARTE2.md` e `parte2/resultados/RELATORIO.md`; a Parte 3 tem `parte3/resultados/RELATORIO.md`. `AI_LOG.md` registra o uso de IA; `ROTEIRO_APRESENTACAO.md` contém o texto da apresentação.
