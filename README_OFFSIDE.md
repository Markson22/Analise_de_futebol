# Guia Rápido - Detecção de Impedimento

## 🎯 O que foi implementado?

Uma funcionalidade completa de **detecção automática de impedimento** em vídeos de futebol, integrada ao pipeline de análise existente.

## 🚀 Uso Rápido

### 1. Executar com Impedimento (Padrão)

```bash
python examples/soccer/main.py \
    --source_video_path examples/soccer/input/jogo_real.mp4 \
    --target_video_path examples/soccer/output/jogo_com_impedimento.mp4 \
    --device cuda \
    --mode COMBINED_ANALYSIS
```

O vídeo de saída incluirá:
- ✅ Detecção de jogadores com velocidade
- ✅ Classificação de times
- ✅ Rastreamento de bola
- ✅ **Marcação de impedimento** (círculo vermelho + label "IMP")

### 2. Executar Testes

```bash
# Navegar para a raiz do projeto
cd "C:\Users\markson.machado\Desktop\VISION COMPUTER\sports-main"

# Executar testes unitários
python -m pytest tests/test_offside.py -v

# Executar teste específico
python -m pytest tests/test_offside.py::TestOffsideDetector::test_basic_offside_detection -v
```

## 📁 Arquivos Criados

```
sports-main/
├── sports/
│   └── common/
│       └── offside.py          # ⭐ Módulo principal de detecção
├── tests/
│   └── test_offside.py         # ⭐ Testes unitários completos
├── docs/
│   └── OFFSIDE_DETECTION.md    # ⭐ Documentação detalhada
└── README_OFFSIDE.md           # ⭐ Este guia rápido
```

## 🎨 Visualização

### Antes (sem impedimento)
```
Jogador: [Elipse Azul] #3 15.2km/h
```

### Depois (com impedimento)
```
Jogador: [Círculo Vermelho] [Elipse Azul] #3 15.2km/h
         [Label] IMP ←
```

## ⚙️ Configuração Personalizada

```python
from sports.common.offside import OffsideDetector, OffsideConfig

# Criar configuração customizada
config = OffsideConfig(
    debounce_frames=10,          # Mais estável (padrão: 5)
    min_defenders=2,              # Mínimo de defensores
    position_tolerance_cm=100.0,  # Tolerância maior (padrão: 50)
    enable_annotations=True,      # Mostrar marcações
    offside_color=(0, 0, 255),    # Vermelho em BGR
    circle_radius=30,             # Tamanho do círculo
)

detector = OffsideDetector(config=config)
```

## 🧪 Cenários de Teste

Os testes cobrem:

1. ✅ **Detecção básica**: Atacante além dos defensores
2. ✅ **Sem impedimento alinhado**: Jogador na mesma linha do defensor
3. ✅ **Própria metade**: Sem impedimento na própria metade
4. ✅ **Debounce**: Confirmação após N frames consecutivos
5. ✅ **Poucos defensores**: Não avalia com menos de 2 defensores
6. ✅ **Atrás da bola**: Sem impedimento se atrás da bola
7. ✅ **Anotação visual**: Verifica marcações no frame

## 📊 Condições de Impedimento

Um jogador é marcado em impedimento quando **TODAS** as condições são verdadeiras:

```
✓ Está na metade adversária
✓ Está além do penúltimo defensor adversário
✓ Está além da bola
✓ Condição confirmada por N frames consecutivos (debounce)
```

## 🔧 Integração no Código

### Opção 1: Usando a função integrada

```python
# Já integrado em run_combined_analysis()
# Basta executar com mode COMBINED_ANALYSIS
```

### Opção 2: Uso manual

```python
from sports.common.offside import detect_and_annotate_offside

# Para cada frame
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

### Opção 3: Com estado persistente (recomendado)

```python
from sports.common.offside import OffsideDetector, OffsideConfig

# Inicializar uma vez
detector = OffsideDetector(config=OffsideConfig())

# Para cada frame do vídeo
for frame in video_frames:
    # Detectar
    offside_ids = detector.detect(
        detections=players,
        players_team_id=team_ids,
        ball_detections=ball_detections,
        transformer=transformer,
        config=pitch_config
    )
    
    # Anotar
    frame = detector.annotate(frame, players, offside_ids)
```

## 🐛 Troubleshooting

### Muitos falsos positivos?
```python
# Aumentar debounce e tolerância
config = OffsideConfig(
    debounce_frames=10,
    position_tolerance_cm=100.0
)
```

### Impedimentos não detectados?
1. Verificar se keypoints do campo são detectados (>= 4 pontos)
2. Verificar qualidade do tracking de jogadores
3. Reduzir debounce: `debounce_frames=3`

### Performance lenta?
```python
# Desabilitar anotações
config = OffsideConfig(enable_annotations=False)

# Ou processar a cada N frames no main.py
```

## 📖 Documentação Completa

Para detalhes técnicos, algoritmos e exemplos avançados:
- Ver: `docs/OFFSIDE_DETECTION.md`

## 🎯 Próximos Passos (Opcionais)

### Melhorias Futuras Sugeridas:

1. **Situações especiais**:
   - Detectar laterais, escanteios, tiro de meta
   - Não marcar impedimento nessas situações

2. **Participação ativa**:
   - Verificar se jogador está interferindo na jogada
   - Considerar trajetória da bola

3. **Visualização melhorada**:
   - Desenhar linha de impedimento no campo
   - Mostrar radar com posições

4. **Histórico e estatísticas**:
   - Contar impedimentos por jogador/time
   - Exportar relatório JSON

5. **Otimização**:
   - Processar apenas quando há mudança de posse
   - Cache de transformações

## 📝 Exemplo Completo

```python
import cv2
import supervision as sv
from ultralytics import YOLO

from sports.common.offside import OffsideDetector, OffsideConfig
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Configurar modelos
player_model = YOLO("football-player-detection.pt")
pitch_model = YOLO("football-pitch-detection.pt")
ball_model = YOLO("football-ball-detection.pt")

# Configurar detector de impedimento
offside_config = OffsideConfig(debounce_frames=5)
offside_detector = OffsideDetector(config=offside_config)
pitch_config = SoccerPitchConfiguration()

# Processar vídeo
cap = cv2.VideoCapture("jogo.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detectar pitch e criar transformer
    pitch_result = pitch_model(frame)[0]
    keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    
    if mask.sum() >= 4:
        transformer = ViewTransformer(
            source=keypoints.xy[0][mask].astype(float),
            target=np.array(pitch_config.vertices)[mask].astype(float)
        )
        
        # Detectar jogadores e times
        player_result = player_model(frame)[0]
        players = sv.Detections.from_ultralytics(player_result)
        # ... classificar times, detectar bola ...
        
        # Detectar impedimento
        offside_ids = offside_detector.detect(
            players, team_ids, ball_detections, transformer, pitch_config
        )
        
        # Anotar
        frame = offside_detector.annotate(frame, players, offside_ids)
        
        if offside_ids:
            print(f"⚠️ Impedimento detectado: Jogadores {offside_ids}")
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 💡 Dicas

1. **Melhor qualidade**: Use `device='cuda'` se tiver GPU disponível
2. **Debugging**: Configure `enable_annotations=True` para visualizar
3. **Produção**: Configure `enable_annotations=False` para melhor performance
4. **Ajuste fino**: Experimente diferentes valores de `debounce_frames` e `position_tolerance_cm`

## 📞 Suporte

Para dúvidas ou problemas:
1. Verificar `docs/OFFSIDE_DETECTION.md` para detalhes técnicos
2. Executar testes: `pytest tests/test_offside.py -v`
3. Verificar exemplos de uso acima

---

**Status**: ✅ Implementação completa e testada
**Versão**: 1.0.0
**Data**: 25/11/2025



