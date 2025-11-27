# Analise_de_futebol
Programa de visão computacional inteligente de futebol

# 🎯 Modos de Uso - Sistema de Análise de Futebol

## ✅ MODO ADICIONADO: OFFSIDE_DETECTION

O modo **OFFSIDE_DETECTION** foi adicionado à classe `Mode`!

---

## 📋 Todos os Modos Disponíveis

### 1. PITCH_DETECTION
**Detecta pontos-chave do campo** (linhas, áreas, círculo central)

```bash
python main.py \
    --source_video_path input/jogo_real.mp4 \
    --target_video_path output/campo_detectado.mp4 \
    --mode PITCH_DETECTION
```

### 2. PLAYER_DETECTION
**Detecta pessoas em campo** (jogadores, goleiros, árbitros)

```bash
python main.py \
    --source_video_path input/jogo_real.mp4 \
    --target_video_path output/jogadores_detectados.mp4 \
    --mode PLAYER_DETECTION
```

### 3. BALL_DETECTION
**Detecta e rastreia a bola**

```bash
python main.py \
    --source_video_path input/jogo_real.mp4 \
    --target_video_path output/bola_detectada.mp4 \
    --mode BALL_DETECTION
```

### 4. PLAYER_TRACKING
**Rastreia jogadores** com IDs únicos entre frames

```bash
python main.py \
    --source_video_path input/jogo_real.mp4 \
    --target_video_path output/jogadores_rastreados.mp4 \
    --mode PLAYER_TRACKING
```

### 5. TEAM_CLASSIFICATION
**Classifica jogadores por time** usando cores de uniforme

```bash
python main.py \
    --source_video_path input/jogo_real.mp4 \
    --target_video_path output/times_classificados.mp4 \
    --mode TEAM_CLASSIFICATION
```

### 6. RADAR
**Desenha radar do campo** com posições dos jogadores

```bash
python main.py \
    --source_video_path input/jogo_real.mp4 \
    --target_video_path output/com_radar.mp4 \
    --mode RADAR
```

### 7. PLAYER_SPEED_ESTIMATION
**Estima velocidade dos jogadores** em km/h

```bash
python main.py \
    --source_video_path input/jogo_real.mp4 \
    --target_video_path output/com_velocidade.mp4 \
    --mode PLAYER_SPEED_ESTIMATION
```

### 8. COMBINED_ANALYSIS ⭐
**Análise completa**: jogadores + velocidade + times + bola + impedimento

```bash
python main.py \
    --source_video_path input/jogo_real.mp4 \
    --target_video_path output/analise_completa.mp4 \
    --device cuda \
    --mode COMBINED_ANALYSIS
```

**Inclui**:
- ✅ Detecção de jogadores
- ✅ Rastreamento com IDs
- ✅ Classificação de times (cores)
- ✅ Velocidade em km/h
- ✅ Rastreamento de bola
- ✅ **Detecção de impedimento** (integrada)

### 9. OFFSIDE_DETECTION 🆕⚽
**Foco em detecção de impedimento** com visualização dedicada

```bash
python main.py \
    --source_video_path input/jogo_real.mp4 \
    --target_video_path output/impedimento.mp4 \
    --device cuda \
    --mode OFFSIDE_DETECTION
```

**Características especiais**:
- ⚠️ **Marcação destacada de impedimento** (círculo vermelho maior + "IMP")
- 📊 **Alerta no frame**: "IMPEDIMENTO! Jogadores: [IDs]"
- 🎯 **Foco exclusivo** em detecção de impedimento
- ✓ Jogadores classificados por time
- ✓ Bola rastreada
- ✓ Sem velocidade (para clareza visual)

---

## 🚀 Comparação: COMBINED_ANALYSIS vs OFFSIDE_DETECTION

| Característica | COMBINED_ANALYSIS | OFFSIDE_DETECTION |
|----------------|-------------------|-------------------|
| **Jogadores detectados** | ✅ | ✅ |
| **Times classificados** | ✅ | ✅ |
| **Velocidade** | ✅ Sim | ❌ Não (para foco) |
| **Bola** | ✅ | ✅ |
| **Impedimento** | ✅ Integrado | ✅ **Destacado** |
| **Marcação impedimento** | Padrão | **Maior e mais visível** |
| **Alerta no frame** | ❌ | ✅ **"IMPEDIMENTO!"** |
| **Uso** | Análise geral | Foco em impedimento |

---

## 📊 Quando Usar Cada Modo?

### COMBINED_ANALYSIS
Use quando quiser:
- Análise completa do jogo
- Ver velocidades dos jogadores
- Ter todas as informações em um só vídeo
- Análise geral para treinamento

### OFFSIDE_DETECTION
Use quando quiser:
- **Focar especificamente em impedimentos**
- Validar decisões de árbitro
- Analisar lances polêmicos
- Visualização mais clara de impedimentos
- Treinar jogadores sobre posicionamento

---

## ⚙️ Parâmetros Comuns

### Device (GPU vs CPU)
```bash
# Com GPU (mais rápido)
--device cuda

# Com CPU
--device cpu
```

### Vídeos de Exemplo
```bash
# Vídeos disponíveis em examples/soccer/input/
- 08fd33_0.mp4
- 0bfacc_0.mp4
- 2e57b9_0.mp4
- 573e61_0.mp4
- jogo_real.mp4
- jogo_real2.mp4
```

---

## 🎯 Exemplos Práticos

### 1. Análise Rápida de Impedimento
```bash
python main.py \
    --source_video_path input/jogo_real.mp4 \
    --target_video_path output/impedimento_analise.mp4 \
    --device cuda \
    --mode OFFSIDE_DETECTION
```

### 2. Análise Completa para Treino
```bash
python main.py \
    --source_video_path input/jogo_real.mp4 \
    --target_video_path output/treino_completo.mp4 \
    --device cuda \
    --mode COMBINED_ANALYSIS
```

### 3. Apenas Velocidades
```bash
python main.py \
    --source_video_path input/jogo_real.mp4 \
    --target_video_path output/velocidades.mp4 \
    --device cuda \
    --mode PLAYER_SPEED_ESTIMATION
```

---

## 🔧 Solução de Problemas

### Erro: "No module named 'sports.common.offside'"
**Solução**:
```bash
cd "C:\Users\markson.machado\Desktop\VISION COMPUTER\sports-main"
python setup.py develop
```

### Erro de memória (GPU)
**Solução**: Use CPU ou reduza resolução
```bash
--device cpu
```

### Vídeo muito lento
**Solução**: Use GPU
```bash
--device cuda
```

---

## 📝 Notas Importantes

1. **OFFSIDE_DETECTION** requer:
   - ✅ Detecção adequada do campo (keypoints)
   - ✅ Pelo menos 4 pontos do campo visíveis
   - ✅ Mínimo de 2 defensores por time

2. **COMBINED_ANALYSIS** é o mais completo mas também o mais pesado

3. Para melhor performance, use GPU (`--device cuda`)

4. O modo **OFFSIDE_DETECTION** é novo e especializado em impedimento

---

## 🎉 Resumo

| Modo | Uso Principal |
|------|---------------|
| **PITCH_DETECTION** | Verificar detecção do campo |
| **PLAYER_DETECTION** | Verificar detecção de jogadores |
| **BALL_DETECTION** | Verificar detecção da bola |
| **PLAYER_TRACKING** | Ver rastreamento de IDs |
| **TEAM_CLASSIFICATION** | Ver classificação de times |
| **RADAR** | Visão tática do jogo |
| **PLAYER_SPEED_ESTIMATION** | Análise de velocidade |
| **COMBINED_ANALYSIS** | **Análise completa** ⭐ |
| **OFFSIDE_DETECTION** | **Foco em impedimento** 🆕 |

---

**Implementado em**: 27/11/2025  
**Versão**: 1.0.0  
**Status**: ✅ Totalmente funcional



