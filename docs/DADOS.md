# Dados, rótulos e prevenção de vazamento

## Formato aceito

Use arquivos locais UTF-8 (BOM é aceito), um objeto JSON por linha. Linhas em branco são ignoradas. As propriedades obrigatórias são `prompt` (texto) e `completion` (`suporte` ou `venda`, exatamente em minúsculas). Propriedades adicionais são ignoradas. Não há remoção de stopwords, acentos ou pontuação, nem conversão do texto do BERTimbau para minúsculas.

O prompt deve ter de 1 a 4.000 caracteres após remoção de espaços externos. O treinamento trunca em tokens conforme `--max-length`, e não em caracteres. Uma mensagem de 4.000 caracteres pode perder seu final na tokenização. Escolha o comprimento por análise dos comprimentos do treino e pelo orçamento de memória.

| Mensagem artificial | Rótulo | Justificativa |
|---|---|---|
| Quero comprar uma lavadora | venda | intenção explícita de aquisição |
| Qual o preço desse forno? | venda | consulta comercial anterior à compra |
| Como conecto minha TV ao Wi-Fi? | suporte | dúvida de funcionamento |
| Minha lavadora não centrifuga | suporte | problema com produto |

Mensagens como “quero trocar minha geladeira” são ambíguas: podem indicar pós-venda ou nova compra. A implementação não cria uma terceira classe e não corrige rótulos automaticamente. Defina uma política de anotação, revise esses casos e use encaminhamento humano. O sinal de revisão por probabilidade não resolve por si só as ambiguidades.

## Auditoria automática

1. O leitor verifica os campos de cada linha e informa arquivo e número da linha ao encontrar erro. Não inclui o conteúdo da mensagem no erro.
2. Para comparar textos, aplica Unicode NFKC, `casefold()` e colapso de espaços. O texto usado pelo modelo mantém grafia e caixa originais.
3. Duplicatas internas com o mesmo rótulo ficam com a primeira ocorrência e entram no contador `duplicates_removed`.
4. Um texto normalizado com dois rótulos diferentes interrompe a execução.
5. Os dois arquivos precisam conter ambas as classes após deduplicação.
6. Qualquer interseção normalizada entre os arquivos interrompe a execução; não há exclusão silenciosa do teste.
7. O treino original precisa ter pelo menos duas amostras únicas por classe. A proporção de validação precisa permitir ao menos duas classes nas duas partições.

O `data_audit.json` informa hashes SHA-256 dos arquivos originais, contagens, duplicatas removidas e tamanho de cada divisão. Não contém prompts. O manifesto inclui essa auditoria.

## Divisão correta

O treino original é dividido com `train_test_split(..., stratify=labels, random_state=seed)`. O valor padrão de validação é 20%. O tokenizer é pré-treinado e a baseline ajusta seu vocabulário somente à partição de treino. No ensaio offline, o vocabulário artificial é construído do arquivo de treino sintético, sem consultar o teste; esse modo não representa uma avaliação independente de qualidade.

A validação é usada para selecionar o checkpoint por F1 macro. O arquivo `teste.jsonl` é reservado para medir o modelo final. Sua leitura inicial só verifica estrutura e interseção; não calcula métricas nem orienta a seleção. Após escolher o melhor checkpoint, o programa calcula os resultados finais do teste para o modelo, a baseline e a classe majoritária.

Se você repetir muitos treinos olhando o teste para escolher o melhor, o teste deixa de ser independente. Use validação para experimentos e mantenha outro conjunto intocado para a avaliação definitiva.

## Limitações da deduplicação

A normalização não identifica paráfrases, erros ortográficos semelhantes, conversas do mesmo cliente nem padrões gerados do mesmo template. Em datasets sintéticos, uma divisão aleatória pode compartilhar templates e inflar as métricas. O gerador de demonstração usa templates diferentes no teste, mas ainda é pequeno e artificial.

Em um projeto real, agrupe por conversa, cliente ou template de origem antes de dividir e mantenha grupos inteiros na mesma partição. Se houver mudança de catálogo ou sazonalidade, considere um teste temporal. Esses grupos não existem no contrato mínimo do desafio, por isso não são inferidos pelo código.

## Arquivos do desafio e privacidade

Obtenha os JSONL usando os links fornecidos pelo organizador. Eles não são baixados automaticamente: URLs assinadas têm prazo e não devem ficar no histórico do Git. Nos ensaios de desenvolvimento não são usados esses arquivos nem o classificador pronto da resolução.

O `.gitignore` cobre `data/`, artefatos e relatórios. Dados sensíveis colocados fora dessas pastas ainda exigem cuidado; confira `git status` antes de adicionar arquivos. A API não envia mensagens a provedores de IA: o processamento ocorre no modelo local. Bibliotecas podem acessar o Hugging Face no primeiro download do modelo público durante o treino; a inferência usa apenas arquivos locais.
