# Resolvedor ARC-AGI-1 com Gemini 3.5 Flash Lite

Este repositório contém a implementação completa, modular e educacional de um resolvedor para o benchmark **ARC-AGI-1** (*Abstraction and Reasoning Corpus*), desenvolvido em **Python** e gerenciado via **uv**.

O resolvedor avalia o modelo de fronteira **Gemini 3.5 Flash Lite (`gemini-3.5-flash-lite`)** com **raciocínio avançado ativado (`thinking_level="high"`)** em duas abordagens fundamentais:
1. **Fim a Fim (*End-to-End*)**: Inferência direta da grade de saída a partir da formulação em linguagem natural e representação bidimensional das matrizes.
2. **Geração de Código (*Code Generation*)**: Síntese de uma função pura em Python `transform(grid: list[list[int]]) -> list[list[int]]`, executada em um ambiente isolado (*sandbox*) com controle estrito de tempo limite (*timeout*).

---

## 📁 Estrutura do Resolvedor

```text
mike_carvalho_borges/
│
├── .env                       # Chave de API (GOOGLE_API_KEY)
│
├── solver/                    # Código-fonte do resolvedor
│   ├── solver.py              # Ponto de acesso principal e lógica do resolvedor
│   └── README.md              # Guia de execução e documentação do resolvedor
│
└── resultados/                # Artefatos consolidados de avaliação
    ├── experiment_summary.txt # Resumo legível em texto com tabelas comparativas
    ├── experiment_summary.json# Resumo estruturado do experimento em JSON
    ├── relatorio.tex          # Código-fonte do relatório acadêmico em LaTeX
    └── relatorio.pdf          # Relatório acadêmico compilado (2 páginas)
```

---

## 🚀 Requisitos e Configuração

### 1. Pré-requisitos
- **Python 3.10+**
- **uv** (gerenciador e executor ultrarrápido de ambientes Python)

Se ainda não possuir o `uv`, instale via PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Configuração de Chave de API
O resolvedor busca automaticamente a variável `GOOGLE_API_KEY` no arquivo `.env` localizado na raiz do projeto (`mike_carvalho_borges/.env`):
```env
GOOGLE_API_KEY="sua_chave_aqui"
```

---

## 💻 Como Executar o Resolvedor

O resolvedor pode ser executado diretamente com o `uv run`, garantindo que todas as dependências (`google-genai`, `python-dotenv`) sejam instaladas em um ambiente isolado sem poluir seu ambiente global.

### Execução Completa (Ambas as Abordagens: Fim a Fim e Código)
```powershell
uv run --with google-genai --with python-dotenv python solver/solver.py --mode both
```

### Executar Apenas Fim a Fim (*End-to-End*)
```powershell
uv run --with google-genai --with python-dotenv python solver/solver.py --mode end_to_end
```

### Executar Apenas Geração de Código (*Code Generation*)
```powershell
uv run --with google-genai --with python-dotenv python solver/solver.py --mode code
```

### Opções de Linha de Comando Disponíveis

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--mode` | `both` | Abordagem de resolução: `end_to_end`, `code` ou `both`. |
| `--label` | `arc_agi1_gemini35_high` | Rótulo identificador do experimento. |
| `--output-dir` | `../resultados` | Pasta onde serão gravados os relatórios e arquivos JSON. |
| `--model` | `gemini-3.5-flash-lite` | Identificador do modelo Gemini utilizado. |
| `--thinking-level` | `high` | Nível de esforço de raciocínio interno do Gemini (`minimal`, `low`, `medium`, `high`). |
| `--delay` | `1.0` | Intervalo em segundos entre duas chamadas sucessivas à API. |
| `--timeout` | `5.0` | Limite de tempo (em segundos) para execução de código gerado no sandbox. |
| `--max-attempts` | `3` | Número máximo de tentativas do laço de correção (*reflection loop*) com diagnóstico. |

### Laço de Correção com Diagnóstico (*Correction Loop*)
Na abordagem de **Geração de Código**, caso a função gerada falhe em um ou mais exemplos de treino, o resolvedor ativa automaticamente o **laço de correção**:
- Executa a função em cada par de treino no sandbox e extrai o diagnóstico de erro (grade produzida vs grade esperada ou mensagem de exceção).
- Envia o histórico e a discrepância visual de volta ao Gemini em um prompt estruturado de correção reflexiva.
- Repete até que todos os exemplos de treino passem ou até atingir o limite `--max-attempts`.
- Registra se a tarefa foi resolvida com o laço, a média de tentativas e se houve casos de sobreajuste ao treino com falha no teste.

---

## 🔍 Saída no Terminal e Métricas Coletadas

Durante a execução de cada tarefa, o resolvedor exibe no terminal:
- **Identificador da Tarefa**: Ex.: `[01/20] Task: 3aa6fb7a.json`
- **Tokens de Entrada (*Input Tokens*)**: Quantidade de tokens enviados no prompt.
- **Tokens de Raciocínio (*Reasoning Tokens*)**: Tokens dedicados à cadeia de pensamento (*CoT*) interna do modelo.
- **Tokens de Resposta (*Answer Tokens*)**: Quantidade de tokens dedicados à resposta final (grade predita ou código Python).
- **Tempo de Execução (s)**: Tempo de chamada da API somado ao tempo de teste e execução no sandbox.
- **Status e Nota (*Grade*)**: `CORRECT [PASS] (Grade: 1.0)` ou `INCORRECT [FAIL] (Grade: 0.0)`.
- **Acurácia Parcial (*Running Score*)**: Progresso cumulativo de acertos sobre as tarefas processadas.

---

## 📊 Arquivos de Saída Gerados

Ao término da execução, os seguintes artefatos são atualizados automaticamente na pasta `resultados/`:
1. `experiment_summary.txt`: Relatório em texto puro contendo a tabela de todas as tarefas com colunas para acerto, tokens e tempo, seguido da tabela comparativa entre Fim a Fim e Geração de Código.
2. `experiment_summary.json`: Estrutura JSON com todos os dados brutos, tempos e predições.
3. `relatorio.tex`: Artigo científico de 2 páginas preenchido conforme as diretrizes do IJCAI, com foco na arquitetura e avaliação do resolvedor.
4. `relatorio.pdf`: Documento PDF único compilado com a formatação final de conferência.
