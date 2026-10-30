# Arquitetura e operação

## Fluxo de execução

O cliente chama o gateway Node.js em `/classify`. Ele valida o envelope e o tamanho, encaminha ao FastAPI e verifica a resposta. O FastAPI carrega uma única instância de `IntentClassifier` na inicialização. O classificador usa tokenizer e pesos locais, modo `eval()` e `torch.inference_mode()`.

Um lock serializa o acesso ao modelo dentro do processo Python. A API aceita lotes de até 32 mensagens; não cria vários modelos por requisição. O processamento roda no pool de threads dos endpoints síncronos do FastAPI. Se houver muitos clientes, será necessário controle de concorrência/fila e dimensionamento: o lock não é uma fila durável nem um limitador de requisições. Cada processo adicional de servidor carrega outra cópia dos pesos.

O gateway tem prazo para a chamada ao Python. Um timeout interrompe a espera HTTP, mas não garante cancelamento de uma inferência que já começou no processo Python. Ajuste os limites considerando latência medida e recursos disponíveis.

## Artefatos

```text
artifacts/bertimbau/
  config.json
  model.safetensors
  tokenizer.json e arquivos auxiliares
  manifest.json
  data_audit.json
  metrics.json
  training_args.bin
  checkpoints/
```

`manifest.json` é escrito por último, com estado `ready`. A inferência exige manifesto compatível, duas classes conhecidas e pesos `safetensors`. Ela não abre `training_args.bin` nem executa código remoto de modelos. Checkpoints incluem estado necessário ao Trainer e são artefatos locais confiáveis, não entradas de usuários.

O checkpoint original do BERTimbau distribui pesos PyTorch em formato `.bin`. No carregamento inicial do treinamento, o código permite esse formato usando `weights_only=True` e PyTorch 2.7.1; não exige uma conversão remota para safetensors. A exportação do modelo ajustado usa `safetensors`, exigido no serviço de inferência. Sempre use modelos de origem confiável.

Para empacotar apenas o modelo de inferência, preserve configuração, tokenizer, pesos e manifesto. Os relatórios auxiliam auditoria; checkpoints não precisam acompanhar a API. Não inclua datasets nem `.env` no pacote compartilhável. Use artefatos de origem confiável; o manifesto não é uma assinatura digital.

## Configuração local

Os comandos devem ser executados da raiz. Python carrega `.env` da pasta atual, sem sobrepor variáveis já presentes no processo. O gateway resolve o `.env` pela localização do seu arquivo, na raiz do projeto. Variáveis herdadas do processo prevalecem. `.env.example` contém valores compartilháveis; `.env` real e checklist são ignorados.

O treinamento usa `DEVICE` (`cpu` por padrão no exemplo). Para Colab/GPU use `--device auto` ou `--device cuda`. A inferência respeita o comprimento gravado no manifesto, mantendo a regra usada pelo modelo. Alterar `MAX_LENGTH` no `.env` afeta novos treinos, sem mudar um artefato já exportado.

Não é necessária API key. A API não implementa autenticação e escuta somente na máquina local por padrão. Antes de expor o serviço em outra rede, configure autenticação, HTTPS, limite de requisições, limites de concorrência e monitoramento. Não use um `.env` com dados fictícios fingindo ser uma credencial válida.

## Docker

O Compose usa um container Python em CPU e outro Node.js, ambos sem usuário root. Publica apenas o gateway em `127.0.0.1`. A origem do modelo é `MODEL_DIR`, montada em `/model` como somente leitura. `create_host_path=false` evita criar silenciosamente uma pasta vazia quando o caminho está errado.

A API Python escuta em `0.0.0.0` somente dentro da rede do Compose. Isso é necessário para o outro container conectar. O `.env` não é copiado para as imagens; somente variáveis explicitamente listadas no Compose são injetadas. Não há download do modelo no startup. O healthcheck precisa passar antes de iniciar o gateway.

Docker é opcional e exige o engine do Docker Desktop em execução. `docker compose config --quiet` verifica a configuração, mas não comprova build nem funcionamento dos containers. Não trate esse comando como teste de implantação.

## Trocar o modelo e reverter

1. Treine em um diretório novo, preservando a versão anterior.
2. Inspecione relatórios, rótulos, manifesto e origem dos dados.
3. Execute inferência e uma chamada HTTP com o novo artefato em ambiente local.
4. Atualize `MODEL_DIR` no `.env` e reinicie Python e gateway; no Compose use `docker compose up -d --force-recreate`.
5. Confira `/health` e a classificação de exemplos de operação aprovados.
6. Para reverter, aponte `MODEL_DIR` para a pasta anterior e reinicie novamente.

O código não promove automaticamente um modelo treinado, nem faz upload dos pesos para o Hugging Face. O processo de aprovação do seu ambiente continua necessário para uso comercial.

## Solução de problemas

| Sintoma | O que fazer |
|---|---|
| Python não encontra retail_intent | Use o Python da `.venv` e execute `python -m pip install -e .` na raiz. |
| Falha ao importar bibliotecas no Colab | Reinicie a sessão após instalação; bibliotecas de visão/áudio incompatíveis com a versão de Torch podem ser removidas no ambiente temporário de texto. |
| Modelo não exportado/manifesto ausente | Termine um treino e aponte `MODEL_DIR` à pasta que contém `manifest.json`. |
| Modelo de ensaio bloqueado | Use BERTimbau treinado; para demonstração local habilite conscientemente `ALLOW_SMOKE_MODEL=true`. |
| CUDA indisponível | Use CPU ou habilite GPU no Colab e confira instalação PyTorch compatível. |
| Falta de memória | Reduza batch e comprimento de entrada; encerre outros processos com modelo carregado. |
| No space left on device | Reserve vários GB em um disco com espaço antes de instalar pesos/dependências. A entrega local reutiliza o Torch instalado para evitar uma segunda cópia. Não apague arquivos pessoais automaticamente. |
| Saída de treino já existe | Escolha outro `--output-dir`; não misture execuções nem apague artefatos úteis sem backup. |
| Vazamento treino/teste | Revise a origem e faça uma divisão independente. Não remova amostras do teste silenciosamente para melhorar a nota. |
| Dados insuficientes para estratificação | Adicione exemplos únicos de ambas as classes e ajuste a fração de validação. |
| Gateway responde 502 | Verifique Python, `UPSTREAM_URL`, modelo carregado e logs sem incluir mensagens pessoais. |
| Gateway responde 504 | Meça latência, reduza lote e revise capacidade antes de aumentar `REQUEST_TIMEOUT_MS`. |
| Porta ocupada | Escolha portas livres no `.env` e atualize também `UPSTREAM_URL`. |
| Node não reconhece loadEnvFile/fetch | Atualize para Node.js 22.9 ou superior. |
| Link Notion expirado | Solicite novo link ao organizador ou use os arquivos baixados anteriormente. |
| Docker não conecta | Inicie Docker Desktop e confira se containers Linux estão habilitados. |

## Encerramento e observabilidade

Localmente use Ctrl+C em cada terminal. No Compose use `docker compose down` (não apaga o modelo montado). O Node tenta encerrar conexões e força o fechamento após 10 segundos. A API desativa logs de acesso no comando fornecido. Os logs do programa não incluem o texto das mensagens nem valores de ambiente.

Para operação contínua, acompanhe disponibilidade, latência, proporção de revisão e erros por classe com exemplos rotulados de forma autorizada. Guarde somente o necessário e siga a política de retenção aplicável. Não coloque dados de clientes no Git para investigar incidentes.
