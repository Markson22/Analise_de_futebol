"""
Exemplo standalone para testar detecção de impedimento.

Este script pode ser executado independentemente para verificar
se a funcionalidade de impedimento está funcionando corretamente.
"""

import sys
import os

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import cv2
from sports.common.offside import OffsideDetector, OffsideConfig
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv


def create_demo_frame():
    """Cria um frame de demonstração para visualização."""
    # Frame HD de 1280x720
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    
    # Desenha um campo simplificado
    # Linha central vertical
    cv2.line(frame, (640, 0), (640, 720), (0, 255, 0), 2)
    
    # Linhas laterais
    cv2.line(frame, (0, 100), (1280, 100), (0, 255, 0), 2)
    cv2.line(frame, (0, 620), (1280, 620), (0, 255, 0), 2)
    
    # Linhas de fundo
    cv2.line(frame, (50, 100), (50, 620), (0, 255, 0), 2)
    cv2.line(frame, (1230, 100), (1230, 620), (0, 255, 0), 2)
    
    # Texto explicativo
    cv2.putText(
        frame,
        "DEMONSTRACAO DE DETECCAO DE IMPEDIMENTO",
        (300, 50),
        cv2.FONT_HERSHEY_BOLD,
        1,
        (0, 0, 0),
        2
    )
    
    return frame


def draw_player_on_field(frame, position, color, label):
    """Desenha um jogador no campo."""
    x, y = int(position[0]), int(position[1])
    
    # Desenha círculo para o jogador
    cv2.circle(frame, (x, y), 20, color, -1)
    
    # Desenha borda branca
    cv2.circle(frame, (x, y), 20, (255, 255, 255), 2)
    
    # Adiciona label
    cv2.putText(
        frame,
        label,
        (x - 10, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


def create_demo_scenario():
    """
    Cria um cenário de demonstração visual.
    
    Returns:
        frame, detections, team_ids, ball_detections, transformer
    """
    frame = create_demo_frame()
    
    # Posições dos jogadores na tela (coordenadas visuais)
    # Time 0 (Azul) - lado esquerdo, ataca para direita
    # Time 1 (Vermelho) - lado direito, defende
    
    player_positions = [
        # Time 0 (Azul) - Atacantes
        [200, 360],   # Jogador 1 - recuado
        [400, 360],   # Jogador 2 - meio-campo
        [600, 360],   # Jogador 3 - atacante avançado
        [1000, 360],  # Jogador 4 - IMPEDIMENTO (além dos defensores)
        
        # Time 1 (Vermelho) - Defensores
        [1100, 380],  # Jogador 5 - Goleiro
        [950, 380],   # Jogador 6 - Último defensor
        [850, 380],   # Jogador 7 - Penúltimo defensor (linha de impedimento)
    ]
    
    team_ids = np.array([0, 0, 0, 0, 1, 1, 1])
    tracker_ids = [1, 2, 3, 4, 5, 6, 7]
    
    # Desenha jogadores no frame
    colors = [
        (255, 0, 0),    # Azul para time 0
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (0, 0, 255),    # Vermelho para time 1
        (0, 0, 255),
        (0, 0, 255),
    ]
    
    for i, (pos, color, tid) in enumerate(zip(player_positions, colors, tracker_ids)):
        label = f"#{tid}"
        draw_player_on_field(frame, pos, color, label)
    
    # Desenha bola
    ball_position = [500, 360]
    cv2.circle(frame, tuple(ball_position), 10, (0, 255, 255), -1)
    cv2.circle(frame, tuple(ball_position), 10, (0, 0, 0), 2)
    
    # Desenha linha de impedimento (penúltimo defensor)
    offside_line_x = 850
    cv2.line(frame, (offside_line_x, 100), (offside_line_x, 620), (0, 165, 255), 2)
    cv2.putText(
        frame,
        "LINHA DE IMPEDIMENTO",
        (offside_line_x - 100, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 165, 255),
        2
    )
    
    # Adiciona legenda
    cv2.putText(frame, "Time 0 (Azul) - Ataca ->", (50, 680), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, "Time 1 (Vermelho) - Defende", (900, 680), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "Jogador #4 em IMPEDIMENTO!", (400, 650), 
                cv2.FONT_HERSHEY_BOLD, 0.8, (0, 0, 255), 2)
    
    # Cria detecções mock
    xyxy = np.array([[p[0] - 20, p[1] - 20, p[0] + 20, p[1] + 20] 
                     for p in player_positions])
    detections = sv.Detections(
        xyxy=xyxy,
        class_id=np.array([2] * len(player_positions)),
        tracker_id=np.array(tracker_ids)
    )
    
    # Cria detecção da bola
    ball_xyxy = np.array([[
        ball_position[0] - 10,
        ball_position[1] - 10,
        ball_position[0] + 10,
        ball_position[1] + 10
    ]])
    ball_detections = sv.Detections(
        xyxy=ball_xyxy,
        class_id=np.array([0]),
        tracker_id=np.array([99])
    )
    
    # Cria transformer (tela -> campo)
    source = np.array([
        [50, 100],
        [1230, 100],
        [1230, 620],
        [50, 620],
    ], dtype=np.float32)
    
    target = np.array([
        [0, 0],
        [12000, 0],
        [12000, 7000],
        [0, 7000],
    ], dtype=np.float32)
    
    transformer = ViewTransformer(source=source, target=target)
    
    return frame, detections, team_ids, ball_detections, transformer


def main():
    """Função principal de demonstração."""
    print("\n" + "="*70)
    print("  DEMONSTRAÇÃO - DETECÇÃO DE IMPEDIMENTO")
    print("="*70)
    
    # Cria cenário de demonstração
    print("\n1. Criando cenário de demonstração...")
    frame, detections, team_ids, ball_detections, transformer = create_demo_scenario()
    
    # Configura detector
    print("2. Configurando detector de impedimento...")
    config = OffsideConfig(
        debounce_frames=1,  # Sem debounce para demo imediata
        min_defenders=2,
        depth_axis='x',
        enable_annotations=True,
        offside_color=(0, 0, 255),  # Vermelho
        circle_radius=35,
        circle_thickness=3
    )
    
    detector = OffsideDetector(config=config)
    pitch_config = SoccerPitchConfiguration()
    
    # Detecta impedimento
    print("3. Detectando impedimento...")
    offside_ids = detector.detect(
        detections=detections,
        players_team_id=team_ids,
        ball_detections=ball_detections,
        transformer=transformer,
        config=pitch_config
    )
    
    print(f"\n📊 RESULTADO:")
    print(f"   Total de jogadores: {len(detections)}")
    print(f"   Time 0 (Azul): {sum(team_ids == 0)} jogadores")
    print(f"   Time 1 (Vermelho): {sum(team_ids == 1)} jogadores")
    
    if offside_ids:
        print(f"   ⚠️  IMPEDIMENTO DETECTADO!")
        print(f"   Jogadores em impedimento: {offside_ids}")
    else:
        print(f"   ✅ Nenhum impedimento detectado")
    
    # Anota frame
    print("\n4. Anotando frame...")
    annotated_frame = detector.annotate(
        frame=frame,
        detections=detections,
        offside_ids=offside_ids
    )
    
    # Salva e exibe resultado
    output_path = os.path.join(
        os.path.dirname(__file__),
        "output",
        "offside_demo.jpg"
    )
    
    # Cria diretório de output se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cv2.imwrite(output_path, annotated_frame)
    print(f"\n5. Frame salvo em: {output_path}")
    
    # Exibe frame
    print("\n6. Exibindo resultado...")
    print("   Pressione qualquer tecla para fechar a janela")
    
    cv2.imshow("Detecção de Impedimento - Demo", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("  ✅ DEMONSTRAÇÃO CONCLUÍDA!")
    print("="*70 + "\n")
    
    # Testa múltiplos frames com debounce
    print("\n" + "="*70)
    print("  TESTE DE DEBOUNCE (5 frames)")
    print("="*70 + "\n")
    
    detector_debounce = OffsideDetector(
        config=OffsideConfig(debounce_frames=3, min_defenders=2, depth_axis='x')
    )
    
    for frame_num in range(1, 6):
        offside_ids = detector_debounce.detect(
            detections, team_ids, ball_detections, transformer, pitch_config
        )
        
        if offside_ids:
            print(f"Frame {frame_num}: ⚠️  IMPEDIMENTO - Jogadores {offside_ids}")
        else:
            print(f"Frame {frame_num}: ⏳ Aguardando confirmação (debounce)...")
    
    print("\n" + "="*70)
    print("  ✅ TESTE DE DEBOUNCE CONCLUÍDO!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Demonstração interrompida pelo usuário")
    except Exception as e:
        print(f"\n\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()



