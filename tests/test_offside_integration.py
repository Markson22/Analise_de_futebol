"""
Teste de integração para detecção de impedimento.

Este teste demonstra o uso completo do módulo de impedimento
em um cenário realista com múltiplos frames.
"""

import numpy as np
import supervision as sv

from sports.common.offside import OffsideDetector, OffsideConfig
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration


def create_simple_transformer():
    """Cria um transformer simples para testes."""
    # Pontos de origem (coordenadas de tela simuladas)
    source = np.array([
        [100, 100],   # Canto superior esquerdo
        [900, 100],   # Canto superior direito
        [900, 500],   # Canto inferior direito
        [100, 500],   # Canto inferior esquerdo
    ], dtype=np.float32)
    
    # Pontos de destino (coordenadas de campo em cm)
    # Campo 120m x 70m = 12000cm x 7000cm
    target = np.array([
        [0, 0],           # Canto superior esquerdo
        [12000, 0],       # Canto superior direito
        [12000, 7000],    # Canto inferior direito
        [0, 7000],        # Canto inferior esquerdo
    ], dtype=np.float32)
    
    return ViewTransformer(source=source, target=target)


def create_match_scenario():
    """
    Cria um cenário de jogo realista para teste.
    
    Cenário: Time 0 (azul) ataca da esquerda para direita
             Time 1 (vermelho) defende do lado direito
             
    Posições no campo (em coordenadas de tela):
    - Time 0: Jogadores no lado esquerdo/centro
    - Time 1: Defensores no lado direito
    - Bola: No centro-direita
    - Atacante: Ultrapassou a defesa (IMPEDIMENTO)
    """
    
    # Posições dos jogadores em coordenadas de tela (pixels)
    # Serão transformadas para coordenadas de campo pelo transformer
    
    positions_screen = [
        # Time 0 (Azul) - Ataca para a direita
        [200, 250],  # ID 1: Defensor (lado esquerdo)
        [300, 250],  # ID 2: Meio-campo
        [400, 250],  # ID 3: Meio-campo avançado
        [700, 250],  # ID 4: ATACANTE EM IMPEDIMENTO (ultrapassou a defesa)
        
        # Time 1 (Vermelho) - Defende lado direito
        [750, 270],  # ID 5: Goleiro
        [600, 270],  # ID 6: Defensor mais recuado
        [550, 270],  # ID 7: Penúltimo defensor (linha de impedimento)
    ]
    
    tracker_ids = [1, 2, 3, 4, 5, 6, 7]
    team_ids = np.array([0, 0, 0, 0, 1, 1, 1])  # 4 do time 0, 3 do time 1
    
    # Criar detecções mock
    n = len(positions_screen)
    xyxy = np.array([[p[0] - 10, p[1] - 20, p[0] + 10, p[1] + 20] 
                     for p in positions_screen])
    
    detections = sv.Detections(
        xyxy=xyxy,
        class_id=np.array([2] * n),  # PLAYER_CLASS_ID = 2
        tracker_id=np.array(tracker_ids)
    )
    
    # Bola está no meio-campo, atrás do atacante em impedimento
    ball_position = [[450, 250]]
    ball_xyxy = np.array([[440, 240, 460, 260]])
    ball_detections = sv.Detections(
        xyxy=ball_xyxy,
        class_id=np.array([0]),  # BALL_CLASS_ID = 0
        tracker_id=np.array([99])
    )
    
    return detections, team_ids, ball_detections, positions_screen


def test_integration_full_scenario():
    """
    Teste de integração completo simulando múltiplos frames de um jogo.
    """
    print("\n" + "="*70)
    print("TESTE DE INTEGRAÇÃO - DETECÇÃO DE IMPEDIMENTO")
    print("="*70)
    
    # Configuração
    config = OffsideConfig(
        debounce_frames=3,
        min_defenders=2,
        depth_axis='x',
        position_tolerance_cm=50.0,
        enable_annotations=True
    )
    
    detector = OffsideDetector(config=config)
    transformer = create_simple_transformer()
    pitch_config = SoccerPitchConfiguration()
    
    # Simular 10 frames do jogo
    print("\nSimulando 10 frames de jogo...")
    print("-" * 70)
    
    for frame_num in range(1, 11):
        detections, team_ids, ball_detections, positions = create_match_scenario()
        
        # Detectar impedimento
        offside_ids = detector.detect(
            detections=detections,
            players_team_id=team_ids,
            ball_detections=ball_detections,
            transformer=transformer,
            config=pitch_config
        )
        
        # Mostrar resultado
        if offside_ids:
            print(f"Frame {frame_num:2d}: ⚠️  IMPEDIMENTO detectado - Jogadores: {offside_ids}")
        else:
            if frame_num < config.debounce_frames:
                print(f"Frame {frame_num:2d}: ⏳ Aguardando debounce ({frame_num}/{config.debounce_frames})...")
            else:
                print(f"Frame {frame_num:2d}: ✅ Sem impedimento")
        
        # Verificação por frame
        if frame_num < config.debounce_frames:
            assert len(offside_ids) == 0, f"Frame {frame_num}: Não deveria detectar (debounce)"
        else:
            assert 4 in offside_ids, f"Frame {frame_num}: Jogador 4 deveria estar em impedimento"
    
    print("-" * 70)
    print("✅ Teste de integração concluído com sucesso!")
    print("="*70 + "\n")


def test_integration_position_changes():
    """
    Testa detecção quando jogadores mudam de posição ao longo do tempo.
    """
    print("\n" + "="*70)
    print("TESTE - MUDANÇA DE POSIÇÃO DOS JOGADORES")
    print("="*70)
    
    config = OffsideConfig(debounce_frames=2, min_defenders=2, depth_axis='x')
    detector = OffsideDetector(config=config)
    transformer = create_simple_transformer()
    pitch_config = SoccerPitchConfiguration()
    
    # Cenário 1: Jogador em impedimento
    print("\nCenário 1: Jogador avança para impedimento")
    print("-" * 70)
    
    for i in range(3):
        positions = [
            [200, 250],  # Time 0
            [300 + i*50, 250],  # Time 0 - avançando
            [600, 270],  # Time 1
            [650, 270],  # Time 1
        ]
        
        xyxy = np.array([[p[0] - 10, p[1] - 20, p[0] + 10, p[1] + 20] 
                         for p in positions])
        detections = sv.Detections(
            xyxy=xyxy,
            class_id=np.array([2, 2, 2, 2]),
            tracker_id=np.array([1, 2, 3, 4])
        )
        
        team_ids = np.array([0, 0, 1, 1])
        ball_detections = sv.Detections(
            xyxy=np.array([[350, 240, 370, 260]]),
            class_id=np.array([0]),
            tracker_id=np.array([99])
        )
        
        offside_ids = detector.detect(
            detections, team_ids, ball_detections, transformer, pitch_config
        )
        
        if offside_ids:
            print(f"  Frame {i+1}: ⚠️  IMPEDIMENTO - Jogadores: {offside_ids}")
        else:
            print(f"  Frame {i+1}: ✅ Sem impedimento")
    
    # Cenário 2: Jogador volta para posição legal
    print("\nCenário 2: Jogador volta para posição legal")
    print("-" * 70)
    
    detector = OffsideDetector(config=config)  # Reset do estado
    
    for i in range(3):
        positions = [
            [200, 250],  # Time 0
            [400 - i*50, 250],  # Time 0 - recuando
            [600, 270],  # Time 1
            [650, 270],  # Time 1
        ]
        
        xyxy = np.array([[p[0] - 10, p[1] - 20, p[0] + 10, p[1] + 20] 
                         for p in positions])
        detections = sv.Detections(
            xyxy=xyxy,
            class_id=np.array([2, 2, 2, 2]),
            tracker_id=np.array([1, 2, 3, 4])
        )
        
        team_ids = np.array([0, 0, 1, 1])
        ball_detections = sv.Detections(
            xyxy=np.array([[350, 240, 370, 260]]),
            class_id=np.array([0]),
            tracker_id=np.array([99])
        )
        
        offside_ids = detector.detect(
            detections, team_ids, ball_detections, transformer, pitch_config
        )
        
        if offside_ids:
            print(f"  Frame {i+1}: ⚠️  IMPEDIMENTO - Jogadores: {offside_ids}")
        else:
            print(f"  Frame {i+1}: ✅ Sem impedimento")
    
    print("-" * 70)
    print("✅ Teste de mudança de posição concluído!")
    print("="*70 + "\n")


def test_integration_statistics():
    """
    Testa coleta de estatísticas de impedimento ao longo de múltiplos frames.
    """
    print("\n" + "="*70)
    print("TESTE - ESTATÍSTICAS DE IMPEDIMENTO")
    print("="*70)
    
    config = OffsideConfig(debounce_frames=2, min_defenders=2, depth_axis='x')
    detector = OffsideDetector(config=config)
    transformer = create_simple_transformer()
    pitch_config = SoccerPitchConfiguration()
    
    # Estatísticas
    total_frames = 20
    offside_count = 0
    offside_players_history = []
    
    print(f"\nProcessando {total_frames} frames...")
    print("-" * 70)
    
    for frame_num in range(total_frames):
        detections, team_ids, ball_detections, _ = create_match_scenario()
        
        offside_ids = detector.detect(
            detections, team_ids, ball_detections, transformer, pitch_config
        )
        
        if offside_ids:
            offside_count += 1
            offside_players_history.append(set(offside_ids))
        
        if (frame_num + 1) % 5 == 0:
            print(f"  Frames processados: {frame_num + 1}/{total_frames}")
    
    print("-" * 70)
    print(f"\n📊 ESTATÍSTICAS:")
    print(f"  Total de frames: {total_frames}")
    print(f"  Frames com impedimento: {offside_count}")
    print(f"  Percentual: {(offside_count/total_frames)*100:.1f}%")
    
    if offside_players_history:
        all_offside_players = set()
        for players in offside_players_history:
            all_offside_players.update(players)
        print(f"  Jogadores em impedimento: {sorted(all_offside_players)}")
    
    print("\n✅ Teste de estatísticas concluído!")
    print("="*70 + "\n")


def run_all_integration_tests():
    """Executa todos os testes de integração."""
    print("\n" + "="*70)
    print("  SUITE DE TESTES DE INTEGRAÇÃO - DETECÇÃO DE IMPEDIMENTO")
    print("="*70)
    
    try:
        test_integration_full_scenario()
        test_integration_position_changes()
        test_integration_statistics()
        
        print("\n" + "="*70)
        print("  ✅ TODOS OS TESTES DE INTEGRAÇÃO PASSARAM!")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ ERRO: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERRO INESPERADO: {e}")
        raise


if __name__ == "__main__":
    run_all_integration_tests()



