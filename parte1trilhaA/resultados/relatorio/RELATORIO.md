# Resultados PA3 — Parte 0 e Parte 1A

Esta pasta reúne métricas, gráficos e arquivos auxiliares das execuções. As métricas do `test_public` são para relato, não para seleção de época ou hiperparâmetros.

## Estado da execução

**Execução registrada como completa pelo script.** Confira as configurações e os arquivos de histórico antes de apresentar.

## Parte 0 — piso supervisionado

| Modelo | Melhor época (val) | F1 macro test_public | Acurácia test_public |
|---|---:|---:|---:|
| small_cnn | 30 | 0.1782 | 0.2343 |
| resnet18 | 22 | 0.4302 | 0.4550 |

A curva de treino/validação da ResNet-18 e sua matriz de confusão estão nesta pasta quando a execução da Parte 0 foi concluída.

## Parte 1A — Trilha A

- Backbone: SimCLR com ResNet-18.
- Aumentação escolhida pelo F1 macro no `val`: context_preserving.
- F1 macro de validação selecionado: 0.3734.

### Avaliação linear

| Inicialização | F1 macro test_public | Acurácia test_public |
|---|---:|---:|
| imagenet_only | 0.4775 | 0.5117 |
| imagenet_ssl | 0.3655 | 0.4032 |
| random_ssl | 0.3643 | 0.4094 |
| imagenet_only | 0.4775 | 0.5117 |
| imagenet_ssl | 0.1916 | 0.2219 |
| random_ssl | 0.2596 | 0.2984 |

Os artefatos permitem comparar ImageNet puro, ImageNet + SimCLR e aleatório + SimCLR. Uma conclusão sobre ganho deve usar a avaliação e as configurações do experimento efetivamente executado.

### Aumentação

O crop padrão permite recortes com área mínima de 8%; a variante `context_preserving` aumenta esse mínimo para 50%. Essa mudança preserva mais contexto, mas não garante que o animal permaneça no recorte. Compare os F1 de validação; o `test_public` é apenas para relatório.

Com crops padrão, o SimCLR pode formar pares positivos sem o animal. Nesses casos, a função de contraste ainda aproxima as duas vistas e pode favorecer elementos compartilhados como fundo, iluminação, vegetação e textura, em vez da espécie. A variante de contexto tenta reduzir esse risco, mas a conclusão deve ser limitada às métricas medidas; o tamanho do crop não garante que o animal apareça.
- standard: F1 macro val 0.3614, F1 macro test_public 0.3929, acurácia 0.4291.
- context_preserving: F1 macro val 0.3734, F1 macro test_public 0.3407, acurácia 0.3822.

Comparação com o piso ResNet-18 da Parte 0: F1 macro 0.4302, acurácia 0.4550.

## Arquivos

- `metricas_resumo.csv`: tabela consolidada de métricas.
- `parte0_metricas.png`: tabela visual da Parte 0.
- `curva_treino_validacao.png`: curvas da ResNet-18, se disponíveis.
- `matriz_confusao.png` e `matriz_confusao.csv`: matriz da ResNet-18.
- `trilha_a_avaliacao_linear.png`: comparação das três avaliações lineares.
- `trilha_a_aumentacao.png`: comparação de aumentações.
- `resultados_trilha_a.csv`: métricas detalhadas por variante.

Os checkpoints e históricos completos ficam nos diretórios `parte0/` e `trilha_a/` dentro de `resultados/`.
