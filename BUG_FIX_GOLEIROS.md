# 🐛 Bug Corrigido - Anotação de Goleiros

## ❌ Problema

**Erro**: `AttributeError: 'tuple' object has no attribute 'xyxy'`

**Localização**: Linha 786 em `run_offside_detection()`

```python
# CÓDIGO INCORRETO (causava erro)
for i, gk in enumerate(goalkeepers):
    team_id = goalkeepers_team_id[i]
    color = sv.Color.from_hex(COLORS[team_id])
    ellipse_gk = sv.EllipseAnnotator(color=color, thickness=2)
    annotated_frame = ellipse_gk.annotate(annotated_frame, gk)  # ❌ ERRO AQUI
```

### Por que deu erro?

- ❌ Tentou iterar sobre `sv.Detections` com `enumerate(goalkeepers)`
- ❌ `gk` virou uma tupla ao invés de um objeto `Detections`
- ❌ Não é possível anotar uma tupla diretamente

---

## ✅ Solução Aplicada

Seguir o mesmo padrão usado para jogadores: **separar por time usando máscaras**.

```python
# CÓDIGO CORRETO (funciona!)
if len(goalkeepers) > 0 and len(goalkeepers_team_id) > 0:
    # 1. Criar máscaras por time
    goalkeepers_team_0_mask = goalkeepers_team_id == 0
    goalkeepers_team_1_mask = goalkeepers_team_id == 1
    
    # 2. Filtrar goleiros por time
    goalkeepers_team_0 = goalkeepers[goalkeepers_team_0_mask]
    goalkeepers_team_1 = goalkeepers[goalkeepers_team_1_mask]
    
    # 3. Anotar time 0
    if len(goalkeepers_team_0) > 0:
        labels_gk_0 = [f"#{tid}" for tid in goalkeepers_team_0.tracker_id]
        ellipse_gk_0 = sv.EllipseAnnotator(color=sv.Color.from_hex(COLORS[0]), thickness=2)
        label_gk_0 = sv.LabelAnnotator(
            color=sv.Color.from_hex(COLORS[0]),
            text_color=sv.Color.from_hex("#FFFFFF"),
            text_padding=5,
            text_thickness=1,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        annotated_frame = ellipse_gk_0.annotate(annotated_frame, goalkeepers_team_0)
        annotated_frame = label_gk_0.annotate(annotated_frame, goalkeepers_team_0, labels=labels_gk_0)
    
    # 4. Anotar time 1
    if len(goalkeepers_team_1) > 0:
        labels_gk_1 = [f"#{tid}" for tid in goalkeepers_team_1.tracker_id]
        ellipse_gk_1 = sv.EllipseAnnotator(color=sv.Color.from_hex(COLORS[1]), thickness=2)
        label_gk_1 = sv.LabelAnnotator(
            color=sv.Color.from_hex(COLORS[1]),
            text_color=sv.Color.from_hex("#FFFFFF"),
            text_padding=5,
            text_thickness=1,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        annotated_frame = ellipse_gk_1.annotate(annotated_frame, goalkeepers_team_1)
        annotated_frame = label_gk_1.annotate(annotated_frame, goalkeepers_team_1, labels=labels_gk_1)
```

---

## 📊 O que mudou?

| Antes (Errado) | Depois (Correto) |
|----------------|------------------|
| ❌ `for i, gk in enumerate(goalkeepers)` | ✅ Separar por time com máscaras |
| ❌ Anotar goleiro individual | ✅ Anotar grupo de goleiros |
| ❌ Uma cor por goleiro | ✅ Uma cor por time |
| ❌ Loop manual | ✅ Operações vetorizadas |

---

## 🎯 Por que essa é a solução correta?

1. **Mantém consistência**: Usa o mesmo padrão dos jogadores (linhas 741-771)
2. **Eficiente**: Usa máscaras NumPy ao invés de loops Python
3. **Correto**: Trabalha com objetos `sv.Detections` nativamente
4. **Cores por time**: Goleiros ficam da mesma cor do time deles

---

## 🔍 Como Verificar se Funcionou?

Execute o comando:
```bash
cd examples/soccer
python main.py \
    --source_video_path "input/08fd33_0.mp4" \
    --target_video_path "output/teste_impedimento.mp4" \
    --mode OFFSIDE_DETECTION \
    --device cuda
```

**Resultado esperado**: 
- ✅ Sem erros
- ✅ Goleiros anotados com elipse da cor do time
- ✅ Labels com IDs dos goleiros
- ✅ Detecção de impedimento funcionando

---

## 📝 Lições Aprendidas

### ❌ Não fazer:
```python
# Não iterar diretamente sobre sv.Detections
for detection in detections:
    annotator.annotate(frame, detection)  # ❌ ERRO!
```

### ✅ Fazer:
```python
# Usar máscaras e filtrar grupos
mask = condition == True
filtered = detections[mask]
annotator.annotate(frame, filtered)  # ✅ CORRETO!
```

---

## 🚀 Status

- ✅ Bug identificado
- ✅ Correção aplicada em `run_offside_detection()`
- ✅ Código testado
- ✅ Documentado

**Data**: 27/11/2025  
**Status**: ✅ RESOLVIDO  
**Arquivo**: `examples/soccer/main.py` linha 773-804

