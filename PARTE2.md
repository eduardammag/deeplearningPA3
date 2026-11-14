# Parte 2 — execução

Importe `PA3_Parte2_Kaggle.ipynb` no Kaggle, ative GPU e Internet e adicione:

- O Dataset com `pa3_dados` completo e extraído.
- `parte1trilhaA/pa3_resultados_completos.zip`, ou os arquivos extraídos desse ZIP.

Execute Run All. Se houver múltiplas entradas, preencha os caminhos nas variáveis
de configuração do notebook. O notebook contém os scripts necessários.

Alternativamente, na raiz deste projeto, com PyTorch CUDA disponível:

```powershell
python -u parte2.py --data-dir pa3_dados --part1-zip parte1trilhaA/pa3_resultados_completos.zip --output-dir resultados/parte2
```

Para apenas validar os arquivos sem treinar:

```powershell
python parte2.py --check-only
```

O treino usa ResNet-18, 224px, batch 16, AdamW com lr 1e-4 e weight decay 1e-4,
30 épocas e seleção pelo F1 macro no val (desempate pela perda). Todas as camadas
são ajustadas. O método parte do pré-treino SimCLR selecionado pela validação da
Parte 1. O piso parte de ImageNet. Não se reutilizam classificadores já ajustados
aos 105 rótulos para iniciar o orçamento de 1, evitando vazamento de rótulos.

Os CSVs oficiais são fixos: seeds 42, 43 e 44 em 5 rótulos/classe; 42 em 1 e 20.
A variabilidade reportada é de fine-tuning, condicionada ao mesmo pré-treino
SimCLR. A cabeça inicial e a ordem dos lotes são pareadas por seed entre métodos.
O teto de 0,686 é fornecido no enunciado, não precisa ser treinado.

Saídas em `resultados/parte2/`:

- `curva_orcamento.csv`: resultados individuais dos dez treinos.
- `resumo_orcamento.csv`: média e desvio amostral (ddof=1).
- `curva_orcamento.png`: F1 macro contra total de rótulos em escala log, com teto.
- `ganho_por_classe.csv` e `.png`: diferenças pareadas de F1 no orçamento 5.
- `RELATORIO.md`: resumo e classes com/sem ganho médio positivo.
- `config.json`: parâmetros e hashes dos CSVs/checkpoint.
- Por método/orçamento/seed: `best.pt`, `history.json`, `metrics.json`,
  `predictions.csv`; durante treinos incompletos, também `resume.pt`.

Reexecute o mesmo comando para retomar; não mude a configuração. Para outra sessão
Kaggle, preserve os outputs, anexe-os e configure `RESTORE_FROM`. O teste público
não é usado para escolher época, crop ou hiperparâmetros.

Validação local: pré-checagem dos dados reais e do checkpoint; teste de integração
com dez treinos sintéticos, reutilização de execução concluída, tabelas e gráficos.
Esses testes não são resultados do PA e não substituem o treino na GPU.
