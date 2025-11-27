# Detecção de Impedimento

Este documento descreve a funcionalidade de detecção automática de impedimento implementada no sistema de análise de futebol.

## Visão Geral

O módulo de detecção de impedimento (`sports.common.offside`) implementa uma lógica simplificada para identificar jogadores em posição de impedimento durante uma partida de futebol, baseada nas regras oficiais do jogo.

## Funcionamento

### Condições de Impedimento

Um jogador é considerado em impedimento quando **todas** as seguintes condições são satisfeitas:

1. **Metade adversária**: O jogador está na metade do campo adversária
2. **Além dos defensores**: O jogador está mais próximo da linha de gol que o **penúltimo defensor** adversário
3. **Além da bola**: O jogador está mais próximo da linha de gol que a bola

### Características Principais

#### 1. Debounce Temporal
Para evitar falsos positivos causados por ruído na detecção, o sistema usa um **buffer temporal**:
- Cada jogador possui um histórico de N frames (padrão: 5)
- Impedimento é confirmado apenas quando 70% dos frames recentes indicam impedimento
- Isso reduz flutuações frame-a-frame

#### 2. Transformação de Coordenadas
O sistema transforma coordenadas da tela para coordenadas reais do campo:
- Usa homografia calculada a partir dos keypoints do campo
- Posições em centímetros (cm) no sistema de coordenadas do campo padrão FIFA
- Campo padrão: 120m x 70m (12000cm x 7000cm)

#### 3. Determinação do Time Atacante
O sistema identifica qual time está atacando baseado em:
- **Primária**: Proximidade com a bola (jogador mais próximo da bola)
- **Secundária**: Posição média dos times no campo

#### 4. Anotação Visual
Jogadores em impedimento são marcados com:
- Círculo vermelho ao redor do jogador
- Label "IMP" (Impedimento) acima do jogador
- Cor configurável (padrão: vermelho #FF0000)

## Uso

### 1. Uso Básico na Análise Combinada

A detecção de impedimento está integrada automaticamente no modo `COMBINED_ANALYSIS`:

```bash
python main.py \
    --source_video_path input/jogo.mp4 \
    --target_video_path output/jogo_analise.mp4 \
    --device cuda \
    --mode COMBINED_ANALYSIS
```

### 2. Uso Programático com Estado

```python
from sports.common.offside import OffsideDetector, OffsideConfig
from sports.configs.soccer import SoccerPitchConfiguration

# Configuração personalizada
config = OffsideConfig(
    debounce_frames=5,           # Número de frames para debounce
    min_defenders=2,              # Mínimo de defensores necessários
    depth_axis='x',               # Eixo de profundidade ('x' ou 'y')
    position_tolerance_cm=50.0,   # Tolerância em cm para posições
    enable_annotations=True,      # Habilitar anotações visuais
    offside_color=(0, 0, 255),    # Cor em BGR (vermelho)
    circle_radius=30,             # Raio do círculo de marcação
    circle_thickness=3            # Espessura da linha
)

# Criar detector (mantém estado entre frames)
detector = OffsideDetector(config=config)
pitch_config = SoccerPitchConfiguration()

# Para cada frame do vídeo
for frame in video_frames:
    # ... obter detections, team_ids, ball_detections, transformer ...
    
    # Detectar impedimento
    offside_ids = detector.detect(
        detections=players,
        players_team_id=team_ids,
        ball_detections=ball_detections,
        transformer=transformer,
        config=pitch_config
    )
    
    # Anotar frame
    annotated_frame = detector.annotate(
        frame=frame,
        detections=players,
        offside_ids=offside_ids
    )
```

### 3. Uso Simples sem Estado

```python
from sports.common.offside import detect_and_annotate_offside

# Função de conveniência que não mantém estado
annotated_frame, offside_ids = detect_and_annotate_offside(
    frame=frame,
    detections=players,
    players_team_id=team_ids,
    ball_detections=ball_detections,
    transformer=transformer,
    config=pitch_config
)

print(f"Jogadores em impedimento: {offside_ids}")
```

## Configurações

### OffsideConfig

Classe de configuração com os seguintes parâmetros:

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `debounce_frames` | int | 5 | Frames consecutivos necessários para confirmar impedimento |
| `min_defenders` | int | 2 | Número mínimo de defensores para avaliar impedimento |
| `depth_axis` | str | 'x' | Eixo de profundidade do campo ('x' horizontal, 'y' vertical) |
| `position_tolerance_cm` | float | 50.0 | Tolerância em cm para considerar posições próximas |
| `enable_annotations` | bool | True | Habilitar anotações visuais no frame |
| `offside_color` | tuple | (0, 0, 255) | Cor BGR para marcação de impedimento |
| `circle_radius` | int | 30 | Raio do círculo de marcação (pixels) |
| `circle_thickness` | int | 3 | Espessura da linha do círculo |

## Testes

### Executar Testes Unitários

```bash
# Executar todos os testes
python -m pytest tests/test_offside.py -v

# Executar teste específico
python -m pytest tests/test_offside.py::TestOffsideDetector::test_basic_offside_detection -v

# Executar com cobertura
python -m pytest tests/test_offside.py --cov=sports.common.offside --cov-report=html
```

### Cenários de Teste Implementados

1. **test_basic_offside_detection**: Detecção básica com atacante além dos defensores
2. **test_no_offside_when_aligned**: Sem impedimento quando jogador alinhado
3. **test_no_offside_in_own_half**: Sem impedimento na própria metade
4. **test_debounce_mechanism**: Verificação do mecanismo de debounce
5. **test_insufficient_defenders**: Não avalia com poucos defensores
6. **test_offside_behind_ball**: Sem impedimento se atrás da bola
7. **test_annotation**: Verificação de anotações visuais

## Limitações e Considerações

### Limitações Conhecidas

1. **Simplificação das Regras**: 
   - Não considera se o jogador está "participando ativamente da jogada"
   - Não considera situações especiais (lateral, escanteio, tiro de meta)
   
2. **Dependências**:
   - Requer detecção precisa de keypoints do campo
   - Necessita pelo menos 4 keypoints válidos para homografia
   - Depende da qualidade do rastreamento de jogadores

3. **Performance**:
   - Processamento adicional por frame (~5-10ms)
   - Requer memória para buffer temporal por jogador

### Casos Especiais

- **Bola não detectada**: Usa última posição conhecida da bola
- **Poucos defensores**: Não avalia impedimento (retorna lista vazia)
- **Keypoints insuficientes**: Pula detecção no frame (sem erro)

## Ajustes e Otimização

### Para Reduzir Falsos Positivos

```python
config = OffsideConfig(
    debounce_frames=10,          # Aumentar para mais estabilidade
    position_tolerance_cm=100.0  # Aumentar tolerância
)
```

### Para Aumentar Sensibilidade

```python
config = OffsideConfig(
    debounce_frames=3,           # Reduzir para resposta mais rápida
    position_tolerance_cm=25.0   # Reduzir tolerância
)
```

### Para Diferentes Orientações de Campo

```python
# Campo horizontal (padrão)
config = OffsideConfig(depth_axis='x')

# Campo vertical
config = OffsideConfig(depth_axis='y')
```

## Integração com Pipeline Existente

O módulo se integra perfeitamente com o pipeline existente:

```
Frame → Pitch Detection → Player Detection → Tracking → Team Classification
                    ↓                                           ↓
              View Transformer                            Team IDs
                    ↓                                           ↓
                Ball Detection  ────────────────────────────────┤
                                                                ↓
                                                      Offside Detection
                                                                ↓
                                                        Annotated Frame
```

## Exemplos de Saída

### Console Output
```
Jogadores em impedimento no frame 245: [3, 7]
Jogadores em impedimento no frame 246: [3]
Jogadores em impedimento no frame 247: []
```

### Anotação Visual
- Jogador em posição normal: Elipse colorida com ID e velocidade
- Jogador em impedimento: Elipse + Círculo vermelho + Label "IMP"

## Troubleshooting

### Problema: Muitos falsos positivos

**Solução**: Aumentar `debounce_frames` e `position_tolerance_cm`

### Problema: Impedimentos não detectados

**Solução**: Verificar se há keypoints suficientes do campo, verificar qualidade do tracking

### Problema: Erro na transformação de coordenadas

**Solução**: Garantir que pelo menos 4 keypoints do campo são detectados corretamente

### Problema: Performance lenta

**Solução**: 
- Reduzir `debounce_frames`
- Desabilitar anotações: `enable_annotations=False`
- Processar a cada N frames

## Referências

- [Regras do Futebol - Lei 11 (Impedimento)](https://www.theifab.com/laws/latest/offside/)
- [Supervision Library](https://github.com/roboflow/supervision)
- [OpenCV Homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)

## Contribuindo

Para melhorias ou correções:

1. Adicione testes para novos cenários em `tests/test_offside.py`
2. Documente mudanças neste arquivo
3. Verifique que todos os testes passam: `pytest tests/test_offside.py`
4. Mantenha compatibilidade com o pipeline existente

## Changelog

### v1.0.0 (2025-11-25)
- Implementação inicial da detecção de impedimento
- Suporte a debounce temporal
- Testes unitários completos
- Integração com `run_combined_analysis`
- Documentação completa



