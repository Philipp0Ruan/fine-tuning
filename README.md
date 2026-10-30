# Fine-tuning: classificação de atendimento

Classificador de mensagens em português com duas classes: **suporte = 0** (dúvidas, funcionamento ou problemas) e **venda = 1** (intenção de compra). O projeto usa BERTimbau com Hugging Face e PyTorch, oferece uma API Python, um gateway Node.js e um notebook Jupyter para treinamento local ou no Google Colab.

> O código permite treinar com os JSONL do desafio. Os ensaios locais usam exclusivamente dados sintéticos e um BERT minúsculo criado localmente. Esse ensaio verifica a execução do pipeline; ele não demonstra qualidade comercial nem substitui o fine-tuning do BERTimbau com dados representativos. Não há API paga, credencial oficial, modelo oficial treinado ou métrica de produção incluídos.

## 1. O que será construído

```text
JSONL de treino -> validação/deduplicação -> divisão estratificada
                                              |
                             treino + validação -> melhor checkpoint
                                              |
JSONL de teste (reservado) --------------------> avaliação final
                                              |
                                     modelo + tokenizer + manifesto
                                              |
cliente -> Node.js :3000 -> FastAPI :8000 -> classificação + probabilidades
```

- Python executa validação, baseline TF-IDF, fine-tuning, avaliação, exportação e inferência.
- Node.js oferece um contrato HTTP estável, valida a entrada e controla tamanho e prazo das requisições.
- O notebook chama os mesmos comandos do projeto; não mantém outra implementação de treinamento.
- Docker Compose é opcional para servir um modelo já treinado. Treinar no Colab com GPU costuma ser mais conveniente.

## 2. Pré-requisitos e instalação no Windows

Use Python **3.11 ou 3.12**, Node.js **22.9+** e Git. Reserve espaço para dependências, cache do Hugging Face e checkpoints. O BERTimbau Base tem cerca de 110 milhões de parâmetros; a memória necessária depende do batch e do comprimento de entrada. CPU funciona, mas o treinamento completo pode ser demorado. Não é preciso login no Hugging Face para o modelo público padrão.

Abra PowerShell e execute:

```powershell
cd C:\Users\fruan\projetos\fine-tunig
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[notebook]"
Copy-Item .env.example .env
```

Se o `.env` já existir, preserve suas configurações. Os exemplos usam o Python da `.venv` diretamente, sem exigir a ativação do ambiente. Em Linux/Colab, substitua `.\.venv\Scripts\python.exe` pelo interpretador instalado. Para GPU local, instale a variante PyTorch/CUDA compatível com seu equipamento conforme o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/).

## 3. Arquivos e responsabilidades

```text
src/retail_intent/
  config.py       configuração e validação de ambiente
  data.py         leitura rigorosa de JSONL e prevenção de vazamento
  metrics.py      accuracy, F1, matriz de confusão e relatório por classe
  train.py        baseline, treinamento e exportação
  predict.py      inferência com modelo salvo
  api.py          FastAPI e healthcheck
  cli.py          comandos train, predict e validate
gateway/          serviço HTTP em Node.js, sem dependências npm externas
notebooks/        guia executável para Jupyter e Colab
docs/             dados, arquitetura, operação e limitações
Dockerfile        imagem da API Python
compose.yaml      API Python + gateway Node.js
.env.example      configuração compartilhável, sem segredos
CHECKLIST.md      roteiro local com tópicos e subtópicos; ignorado pelo Git
```

`data/`, `artifacts/`, `reports/`, `.env`, ambientes virtuais e testes locais são ignorados. O notebook é entregue com saídas vazias. Não são versionados datasets do desafio, pesos, resultados inventados ou testes.

## 4. Primeira execução: dados artificiais

Para exercitar todo o pipeline sem baixar modelos ou dados oficiais, abra o notebook e mantenha `MODE = "smoke"`:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab notebooks/fine_tuning.ipynb
```

Esse modo do notebook cria mensagens artificiais, um tokenizer local e um BERT pequeno com pesos iniciais aleatórios, executa treino real com atualização de parâmetros, avalia e recarrega o artefato para inferência. O manifesto identifica `purpose=smoke`. A API rejeita esse artefato por padrão. Métricas desse modo são diagnósticas, sem significado de desempenho no desafio.

Os JSONL artificiais ficam na pasta `data/` da execução e podem ser auditados pela CLI. Por exemplo, para o ensaio local já executado nesta entrega:

```powershell
.\.venv\Scripts\python.exe -m retail_intent.cli validate --train-file artifacts/smoke/data/treino.jsonl --test-file artifacts/smoke/data/teste.jsonl
```

## 5. Usar os dados do desafio

1. Obtenha `treino.jsonl` e `teste.jsonl` de uma fonte autorizada e salve-os em `data/raw/`.
2. Links assinados do Notion podem expirar. Se isso acontecer, solicite links novos ou exporte os arquivos; não coloque URLs assinadas no Git.
3. Cada linha precisa conter exatamente um objeto JSON com `prompt` e `completion`. Outras propriedades são permitidas e ignoradas.

```json
{"prompt":"Quero comprar uma cafeteira", "completion":"venda"}
{"prompt":"Como configuro o timer da cafeteira?", "completion":"suporte"}
```

4. Não use um único array JSON, objetos concatenados na mesma linha ou rótulos fora de `suporte` e `venda`.
5. Execute a auditoria antes do treino:

```powershell
.\.venv\Scripts\python.exe -m retail_intent.cli validate --train-file data/raw/treino.jsonl --test-file data/raw/teste.jsonl
```

Textos vazios, JSON inválido, classes ausentes, rótulos contraditórios e sobreposição entre treino e teste interrompem a execução. Duplicatas internas com o mesmo rótulo são contadas e removidas, sem mudar o texto utilizado pelo tokenizer. A comparação normaliza Unicode, caixa e espaços. Mensagens apenas parecidas ainda exigem inspeção humana. Consulte [o contrato de dados](docs/DADOS.md).

## 6. Treinar o BERTimbau

```powershell
.\.venv\Scripts\python.exe -m retail_intent.cli train --train-file data/raw/treino.jsonl --test-file data/raw/teste.jsonl --output-dir artifacts/bertimbau --epochs 5 --batch-size 8 --learning-rate 0.00002 --validation-size 0.2 --max-length 128 --seed 42
```

O programa separa a validação **somente do treino**, de modo estratificado e reprodutível, escolhe o checkpoint por **F1 macro da validação** e usa o teste reservado apenas na avaliação final. Registra também as métricas de uma baseline TF-IDF + regressão logística para comparação, treinada sem observar o teste. O teste não deve ser usado para escolher hiperparâmetros; repetições orientadas por seus resultados transformam-no em validação.

Parâmetros úteis:

| Opção | Padrão | Efeito |
|---|---|---|
| `--model-name` | `MODEL_NAME` do `.env` | checkpoint público inicial |
| `--revision` | `MODEL_REVISION` | tag ou hash de revisão do modelo |
| `--epochs` | 5 | passagens pelo treino |
| `--batch-size` | 8 | amostras por dispositivo |
| `--learning-rate` | 0.00002 | taxa de aprendizado |
| `--validation-size` | 0.2 | proporção separada do treino |
| `--max-length` | 128 | limite de tokens, com truncamento |
| `--seed` | 42 | semente para divisão e treino |
| `--output-dir` | `artifacts/model` | destino novo para o artefato |

O diretório de saída precisa estar vazio ou não existir para impedir a mistura de execuções. O treino salva tokenizer, pesos `safetensors`, configuração de classes, checkpoints, `manifest.json`, `metrics.json` e `data_audit.json`. Os hashes SHA-256 dos arquivos e a revisão efetivamente resolvida do modelo ajudam a rastrear a execução. Fixe `--revision` em um hash para repetir a origem exata; `main` pode mudar. Sementes não garantem igualdade bit a bit entre GPUs e versões de bibliotecas.

Os relatórios incluem accuracy, precision, recall e F1 por classe, F1 macro, matriz de confusão (linhas reais, colunas previstas, ordem `suporte`, `venda`) e baseline majoritária. Não existe pontuação mínima universal: estabeleça metas conforme o custo dos erros. Veja [avaliação e limitações](docs/AVALIACAO.md).

## 7. Prever e iniciar os serviços

Depois de treinar, ajuste `.env` para `MODEL_DIR=artifacts/bertimbau`. Para inferência direta:

```powershell
.\.venv\Scripts\python.exe -m retail_intent.cli predict --model-dir artifacts/bertimbau --text "Gostaria de comprar uma geladeira"
```

Em um terminal na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe -m retail_intent.api
```

Em outro terminal na mesma pasta:

```powershell
npm --prefix gateway start
```

O Node lê o `.env` da raiz. As duas portas escutam apenas em `127.0.0.1` por padrão. Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). O processo Python falha explicitamente ao iniciar se o modelo não estiver pronto; nunca responde usando heurística ou pesos substitutos.

```powershell
Invoke-RestMethod http://127.0.0.1:3000/health
$body = @{texts=@('Quero adquirir um aspirador','Meu forno parou de aquecer')} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:3000/classify -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

Contrato de `POST /classify`: objeto com somente `texts`, uma lista de 1 a 32 mensagens, cada uma com 1 a 4.000 caracteres após remoção de espaços externos. O gateway limita o corpo a 512 KiB, exige `application/json`, rejeita propriedades extras e usa prazo de 30 segundos. A resposta tem `predictions`, cada item com `label`, `confidence`, `probabilities` e `needs_review`. A mensagem original não é devolvida nem registrada pelo código.

`confidence` é a maior probabilidade softmax, não uma garantia calibrada. `needs_review=true` indica que o valor ficou abaixo do limiar configurado, mas não suspende automaticamente o fluxo do bot. O consumidor deve decidir o encaminhamento humano. `/health` retorna estado e propósito do modelo, sem dados de clientes.

| Status | Significado |
|---|---|
| 200 | requisição concluída |
| 400 | JSON malformado ou requisição inválida |
| 404 / 405 | caminho ou método incorreto |
| 413 | corpo maior que o limite |
| 415 | Content-Type incorreto |
| 422 | campos inválidos |
| 502 | serviço Python indisponível ou resposta inválida |
| 504 | prazo de resposta excedido |

Para servir um artefato de demonstração, use explicitamente o caminho exibido pelo notebook em `MODEL_DIR` e `ALLOW_SMOKE_MODEL=true`. O ensaio local desta entrega fica em `artifacts/smoke/model`. Essa opção é apenas para ensaio local; devolva-a a `false` para uso normal.

## 8. Jupyter e Google Colab

```powershell
.\.venv\Scripts\python.exe -m jupyter lab notebooks/fine_tuning.ipynb
```

No Colab, envie o notebook, selecione GPU em **Ambiente de execução > Alterar tipo de ambiente de execução**, execute a célula de obtenção do projeto e escolha o modo (`smoke` inicialmente; `challenge` após enviar os JSONL). O notebook verifica os arquivos antes de treinar e oferece download de um ZIP com o modelo. Não solicita token para repositório/modelo público. O armazenamento da sessão Colab é temporário: baixe o ZIP antes de encerrar. O projeto não faz upload automático de dados nem de pesos para o Hub.

## 9. Docker Compose opcional

Instale Docker Desktop com suporte Linux e treine um modelo antes. O Compose usa CPU e monta o diretório escolhido como somente leitura. O treinamento não ocorre na inicialização.

```powershell
docker compose up --build -d
docker compose ps
docker compose logs --tail 50
docker compose down
```

O `.env` precisa apontar `MODEL_DIR` para o artefato treinado. Somente o gateway é publicado em `127.0.0.1:3000`. A API Python fica na rede interna, e o gateway só inicia após seu healthcheck. O `.dockerignore` exclui datasets, `.env`, pesos e testes do contexto de build. Não há banco de dados, Redis ou serviço pago.

## 10. Configurações e operação

| Variável | Uso |
|---|---|
| `MODEL_NAME`, `MODEL_REVISION` | origem do modelo no treinamento |
| `MODEL_DIR` | artefato local servido |
| `MAX_LENGTH` | tokens usados no treino padrão; o manifesto prevalece na inferência |
| `CONFIDENCE_THRESHOLD` | limiar de encaminhamento, entre 0 e 1 |
| `DEVICE` | `cpu`, `cuda` ou `auto` |
| `ALLOW_SMOKE_MODEL` | autoriza artefato de ensaio, padrão `false` |
| `PYTHON_HOST`, `PYTHON_PORT` | endereço da API Python |
| `UPSTREAM_URL` | endereço HTTP da API visto pelo gateway |
| `GATEWAY_HOST`, `GATEWAY_PORT` | endereço do gateway |
| `REQUEST_TIMEOUT_MS` | prazo do gateway em milissegundos |

O `.env` local contém configurações funcionais, sem credenciais inventadas. Nenhuma chave de API é necessária. Para acesso externo adicione proxy HTTPS, autenticação, limite de requisições e políticas de retenção da sua organização antes de alterar o endereço de escuta. Consulte [operação e solução de problemas](docs/OPERACAO.md).

## 11. Git e entrega

Destino local solicitado: `C:\Users\fruan\projetos\fine-tunig`. Repositório remoto: [Philipp0Ruan/fine-tuning](https://github.com/Philipp0Ruan/fine-tuning). A diferença de grafia entre pasta e repositório é intencional.

O `CHECKLIST.md` local contém o roteiro detalhado e o registro das verificações. Ele, os testes temporários, o `.env` real e os dados não entram nos commits. Antes de publicar, confira:

```powershell
git status --short
git check-ignore .env CHECKLIST.md
git ls-files
```

Versione somente os arquivos de implementação, notebook limpo, configuração de exemplo e documentação. Não use `git add -f` para incluir dados ou credenciais. Os testes simulados validam contratos, dados, treino, serialização e comunicação entre serviços; sua execução não permite concluir que o classificador funciona bem com clientes reais.

## 12. Verificações desta entrega

Verificações realizadas em 30/10/2026 usando somente mensagens artificiais, sem API key e sem os JSONL oficiais do desafio:

| Verificação | Resultado |
|---|---|
| Dados, configuração, modelo e API Python | 22 testes locais aprovados, incluindo compatibilidade com pesos `.bin` |
| Gateway Node.js | 11 testes locais aprovados, incluindo falha de conexão, timeout e limites |
| Treino sintético | pesos atualizados, checkpoint exportado e modelo recarregado |
| Notebook em modo smoke | células executadas; artefato e ZIP gerados |
| Integração HTTP local | Node.js → FastAPI → BERT sintético aprovada |
| Docker Compose | duas imagens construídas, dois serviços saudáveis e classificação aprovada; ambiente temporário encerrado |

Os testes e seus registros permanecem apenas na pasta local ignorada `.local-tests/`; não fazem parte do código versionado. O notebook demonstra o pipeline e é entregue sem saídas. A qualidade do BERTimbau com os dados do desafio ainda precisa ser medida pelo treinamento completo: nenhuma métrica do ensaio foi apresentada como resultado comercial.

## 13. Referências

- [BERTimbau: modelo público para português e licença do checkpoint](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
- [Hugging Face: classificação de sequências (versão usada pelo projeto)](https://huggingface.co/docs/transformers/v4.57.1/en/tasks/sequence_classification)
- [Trainer e argumentos de treinamento](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer)
- [PyTorch: instalação](https://pytorch.org/get-started/locally/)
- [Node.js: carregamento de arquivo de ambiente](https://nodejs.org/api/cli.html#--env-file-if-existsconfig)
- [Desafio de referência da Rocketseat](https://github.com/rocketseat-education/ia-para-devs/tree/master/Desafio%20fine-tuning)

O projeto implementa sua própria solução; o modelo já ajustado da resolução não é utilizado como atalho de treinamento.
