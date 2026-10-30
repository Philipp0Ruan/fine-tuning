# Avaliação e limites de uso

## Métricas registradas

- **Accuracy:** proporção total de acertos; pode esconder erros da classe minoritária.
- **Precision por classe:** entre as mensagens previstas para a classe, quantas realmente pertenciam a ela.
- **Recall por classe:** entre as mensagens reais da classe, quantas foram encontradas.
- **F1 por classe:** média harmônica entre precision e recall.
- **F1 macro:** média simples dos dois F1; dá peso igual a suporte e venda e seleciona o checkpoint.
- **Support:** quantidade de exemplos reais da classe.
- **Matriz de confusão:** linhas são classes reais e colunas são previstas, na ordem `suporte`, `venda`.

```text
                         previsto suporte    previsto venda
real suporte                  correto          suporte perdido
real venda               venda desviada              correto
```

O relatório `metrics.json` contém `validation`, `test`, `baseline_tfidf`, `baseline_majority_test` e o melhor F1 macro de validação. Classes sem previsão recebem zero nas métricas indefinidas (`zero_division=0`). Compare os números juntamente com o tamanho da amostra: uma mudança de um exemplo pode alterar muito o resultado de um teste pequeno.

## Baselines

A baseline TF-IDF usa n-gramas de caracteres de tamanho 3 a 5 e regressão logística com balanceamento de classes. A classe majoritária é determinada somente pela partição de treino. São referências para entender se o custo do Transformer trouxe benefício. As métricas dessas referências são salvas; seus estimadores não são exportados nem usados como fallback pela API.

O treinamento Transformer usa a perda padrão de classificação do modelo. Não há peso de classe aplicado nessa perda. Se houver desbalanceamento forte, analise recall por classe e experimente ajuste de perda ou amostragem usando apenas a validação. Não faça esse ajuste com base no teste final.

## Confiança e revisão humana

A saída contém as duas probabilidades softmax. A maior define o rótulo, e `needs_review` é verdadeiro quando essa probabilidade é menor que `CONFIDENCE_THRESHOLD`. O valor padrão de 0,75 é uma configuração operacional inicial, não um limiar comprovado.

Redes neurais podem ter confiança alta em frases fora do domínio, mensagens ambíguas ou entradas sem significado. Para usar a confiança como critério, faça calibração em validação e meça cobertura e erro dos casos encaminhados automaticamente. A aplicação cliente deve definir como os casos de revisão serão tratados. Não há classe “outros” nem detector de intenção múltipla.

## Como aceitar uma versão

1. Defina o custo de confundir suporte com venda e de perder uma oportunidade comercial.
2. Estabeleça metas de F1 e recall por classe antes de abrir o teste final.
3. Compare BERT, TF-IDF e classe majoritária nas mesmas partições.
4. Revise amostras de erros, ambiguidades, mensagens longas e linguagem informal.
5. Verifique diferentes produtos, regiões, erros de digitação e mensagens fora do catálogo.
6. Valide com exemplos representativos que não compartilhem templates ou conversas com o treino.
7. Registre a decisão e preserve o artefato aprovado junto com seu manifesto.

Este projeto não define uma meta fictícia de 95% ou outra taxa. Bons resultados em dados didáticos não demonstram prontidão para atendimento real.

## Reprodutibilidade e rastreabilidade

O manifesto salva semente, configuração, hashes dos JSONL, versões das bibliotecas principais, plataforma, dispositivo, revisão solicitada e hash de revisão resolvido quando disponível no Hub. Para modelos locais, a revisão resolvida pode ser nula. Não há promessa de reprodutibilidade bit a bit entre hardware, drivers ou bibliotecas diferentes.

As dependências diretas estão fixadas no `pyproject.toml`, exceto a faixa compatível de Pydantic e o backend de build. Dependências transitivas não têm lock universal entre Windows, Linux e CUDA. Para uma entrega aprovada, congele o ambiente específico e use imagens Docker por digest conforme a política do seu projeto.

## Ensaio sintético

O modo `smoke` do notebook usa 48 exemplos artificiais no arquivo de treino e 8 no teste, um BERT de uma camada com estados de dimensão 32 e pesos iniciais aleatórios. Executa duas épocas e permite salvar/recarregar o modelo e exercitar o contrato. O roteiro local de testes, ignorado pelo Git, também verifica a alteração dos pesos da cabeça de classificação.

Isso é uma verificação de integração do treinamento, e não fine-tuning de um modelo pré-treinado útil em português. O manifesto é marcado `purpose=smoke`, e a API o recusa sem autorização de demonstração. `production_quality_measured=false` é salvo no resumo. O caminho de fine-tuning efetivo carrega o BERTimbau pré-treinado e permanece disponível no comando `train` e no notebook.

As verificações temporárias não são commitadas, conforme solicitado. Elas cobrem dados inválidos, vazamento, configuração, treinamento, bloqueio do modelo artificial, validação da API e gateway HTTP. Os números de métricas do ensaio não são publicados como resultado do desafio.
