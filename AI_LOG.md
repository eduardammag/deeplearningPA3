# Registro de uso de IA

Usei ferramentas de IA como apoio para compreender o enunciado, implementar e revisar partes do código e organizar a entrega. As decisões e resultados apresentados são de responsabilidade do grupo; o código e as métricas devem ser compreendidos e conferidos por nós.

- **Parte 0:** pedi ajuda para estruturar as baselines supervisionadas, seleção por validação e geração de avaliações e gráficos. O código está em `parte0/part0.py` e os resultados ficam em `parte0/resultados/`.
- **Parte 1A:** pedi ajuda na implementação da trilha SimCLR, avaliação linear, fine-tuning e comparação de políticas de crop. O notebook está em `parte1trilhaA/` e os relatórios e arquivos de execução ficam em `parte1trilhaA/resultados/`.
- **Parte 2:** pedi ajuda para organizar o experimento dos orçamentos, parear seeds, gerar métricas e curva e documentar como reproduzir as execuções. O código está em `parte2/parte2.py`; os resultados ficam em `parte2/resultados/` e a cópia da curva entregue em `entrega/curva_orcamento.csv`.
- **Entrega:** pedi ajuda para revisar os entregáveis, preparar `predict.py`, descrever os comandos no README e organizar o checkpoint final `modelo_entregue.pt`. O checkpoint corresponde ao método `piso`, orçamento de 20 rótulos por classe, seed 42, selecionado pela melhor época de F1 macro em `val`. `test_public` não foi usado para essa seleção.
- **Parte 3:** pedi ajuda para organizar a inspeção Grad-CAM de erros e acertos. O código está em `parte3/parte3.py` e a galeria, os mapas e o relatório ficam em `parte3/resultados/`.
- **Revisão final:** pedi ajuda para reorganizar os arquivos por parte, revisar o README e este registro e esclarecer o papel do `test_private`. O conjunto privado é mantido pela monitoria; não está incluído nos dados locais e a monitoria executa o monitor de predição nele.

Revisei os arquivos e descrições gerados com assistência de IA. A reprodução de treinos depende dos dados completos e dos artefatos da Parte 1A descritos no README; a presença de uma curva ou checkpoint no repositório, por si só, não demonstra que todos os experimentos foram reexecutados a partir do zero nesta cópia.
