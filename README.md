# Logo Vectorizer

Vetoriza logos (imagens planas, com ou sem texto) em **SVG**: quantização de cores,
remoção de fundo e curvas Bézier com preservação de cantos. Determinístico (mesma
entrada + mesma seed = mesma saída).

## Instalação

```powershell
pip install -r requirements.txt
```

## Uso

```powershell
python main.py logo.png -o logo.svg                 # básico (detecta cores e remove fundo)
python main.py ./imagens --batch --out-dir saida    # pasta inteira
python main.py logo.jpg --scale 2 --denoise 2       # imagem pequena ou JPEG com ruído
python main.py logo.png --trim                      # recorta margens do fundo
python main.py logo.png --palette '#1F4E9C,#F5B301' # paleta fixa
python main.py logo.png --no-transparent            # mantém o fundo como rect
```

## Opções

| Opção | Default | Descrição |
|---|---|---|
| `--colors N` | 6 | Nº de cores (K-Means) |
| `--palette` | — | Cores fixas `#RRGGBB,...` |
| `--bg` | `auto` | `auto` \| `none` \| `#RRGGBB` \| nome CSS |
| `--no-transparent` | off | Fundo vira `<rect>` em vez de transparente |
| `--scale N` | 1 | Upscale antes do traço (imagens pequenas) |
| `--denoise N` | 1 | 0=off, 1=mediana, 2=mediana+bilateral |
| `-t --tolerance` | 0.6 | Erro máx. do Bézier em px (menor = mais fiel) |
| `--corners` | 110 | Ângulo p/ preservar canto (menor = mais suave) |
| `--min-area` | 0.0003 | Área mínima de um blob (remove sujeira) |
| `--trim` | off | Recorta margens uniformes |
| `--max-size` | 0 | Limita o maior lado em px |
| `--seed` | 12345 | Semente do K-Means |

## Estrutura

```
main.py            → CLI
vectorizer/
  pipeline.py      → orquestração
  quantize.py      → preprocess, K-Means, fundo
  geometry.py      → cantos + fitting Bézier
  trace.py         → máscara → regiões
  svg.py           → escritor SVG
tests/             → gerador de amostras + validador
```

Limitações: fotos com gradientes não viram logos perfeitos (a paleta é reduzida —
aumente `--colors`); texto muito pequeno depende da resolução de origem (`--scale 2`).