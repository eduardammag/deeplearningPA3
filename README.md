# Entrega PA3

Este repositório contém o código, a curva de orçamento e o modelo final para classificação das 21 classes. O modelo entregue é ResNet-18, treinado no orçamento de 20 rótulos por classe (seed 42); a melhor época foi escolhida pelo F1 macro de validação. Os resultados de `test_public` são apresentados para auditoria e não foram usados para selecionar a época.

## Inferência do modelo entregue

Requer Python 3.10+ e os pacotes listados em `requirements.txt`. O checkpoint inclui todos os pesos e nomes de classe; inferência usa CPU e não acessa a internet.

```powershell
python -m pip install -r requirements.txt
python predict.py --imagens PASTA --saida preds.csv
```

O CSV resultante tem as colunas `image_id,label`. `image_id` é o nome do arquivo sem extensão e `label` é o nome da classe. O comando imprime o número de parâmetros. O padrão usa `modelo_entregue.pt` ao lado do script; pode-se passar outro checkpoint compatível com `--pesos`.

## Treino e avaliação

O experimento de orçamento pode ser executado com o conjunto de dados completo em `pa3_dados/` e o checkpoint SimCLR indicado abaixo. Os dez treinos levam tempo significativo; GPU é recomendada. O script seleciona checkpoints por F1 macro de validação e avalia `test_public` ao terminar.

```powershell
python -m pip install -r requirements.txt
python parte2.py --data-dir pa3_dados --ssl-checkpoint resultados/parte2/source/context_preserving/imagenet_ssl.pt --output-dir resultados/parte2
```

O checkpoint SimCLR e os dados são artefatos locais, ignorados pelo Git; disponibilize-os junto ao ambiente de execução para repetir o treinamento. O resultado agregado é `resultados/parte2/curva_orcamento.csv`. A cópia da curva usada na entrega está em `entrega/curva_orcamento.csv`.

Para avaliar o modelo entregue em um diretório de imagens sem rótulos, execute o comando `predict.py` acima. Esse é o monitor CPU/offline correspondente ao checkpoint entregue.

## Arquivos

- `predict.py`: inferência por pasta de imagens.
- `modelo_entregue.pt`: checkpoint final, com arquitetura/configuração e nomes das classes.
- `entrega/curva_orcamento.csv`: método, orçamento, seed, F1 macro e acurácia dos dez treinos.
- `AI_LOG.md`: registro de uso de IA.
- `parte2.py`, `part0.py`: código de treinamento e avaliação.

Os experimentos da Parte 2 comparam fine-tuning iniciado de ImageNet (`piso`) e de ImageNet + SimCLR (`simclr`), em orçamentos de 1, 5 e 20 rótulos por classe. A curva e os detalhes das execuções ficam em `resultados/parte2/` durante o trabalho local.
