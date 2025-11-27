"""
Testes unitários para o módulo de detecção de impedimento.
"""

import numpy as np
import pytest
import supervision as sv

from sports.common.offside import OffsideConfig, OffsideDetector, detect_and_annotate_offside
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration


class MockTransformer:
    """Transformer mock para testes que retorna coordenadas de campo simuladas."""
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Simula transformação para coordenadas de campo.
        Multiplica por 10 para simular escala de cm.
        """
        return points * 10.0


def create_mock_detections(positions: list, tracker_ids: list, class_ids: list = None) -> sv.Detections:
    """
    Cria detecções mock para testes.
    
    Args:
        positions: Lista de posições [x, y] dos jogadores.
        tracker_ids: Lista de IDs de rastreamento.
        class_ids: Lista de IDs de classe (opcional).
    
    Returns:
        sv.Detections mock.
    """
    n = len(positions)
    if class_ids is None:
        class_ids = [2] * n  # PLAYER_CLASS_ID = 2
    
    # Cria bounding boxes mock (não usadas para impedimento)
    xyxy = np.array([[p[0] - 10, p[1] - 20, p[0] + 10, p[1] + 20] for p in positions])
    
    detections = sv.Detections(
        xyxy=xyxy,
        class_id=np.array(class_ids),
        tracker_id=np.array(tracker_ids)
    )
    
    return detections


class TestOffsideDetector:
    """Testes para a classe OffsideDetector."""
    
    def test_basic_offside_detection(self):
        """Testa detecção básica de impedimento."""
        config = OffsideConfig(
            debounce_frames=1,  # Sem debounce para teste simples
            min_defenders=2,
            depth_axis='x'
        )
        detector = OffsideDetector(config=config)
        pitch_config = SoccerPitchConfiguration()
        
        # Cenário: Atacante (time 0) ultrapassou os defensores (time 1)
        # Campo: 0 a 12000 cm, centro em 6000 cm
        # Time 0 ataca para a direita (+x), time 1 ataca para a esquerda (-x)
        
        # Posições em coordenadas de tela (serão multiplicadas por 10 pelo mock)
        # Após transformação: time 0 à esquerda, time 1 à direita
        positions = [
            [400, 200],  # Jogador 1, time 0, posição normal: 4000 cm
            [500, 200],  # Jogador 2, time 0, posição normal: 5000 cm
            [900, 200],  # Jogador 3, time 0, ATACANTE EM IMPEDIMENTO: 9000 cm
            [700, 210],  # Jogador 4, time 1, defensor: 7000 cm
            [800, 210],  # Jogador 5, time 1, defensor mais próximo do gol: 8000 cm
        ]
        
        tracker_ids = [1, 2, 3, 4, 5]
        team_ids = np.array([0, 0, 0, 1, 1])
        
        detections = create_mock_detections(positions, tracker_ids)
        
        # Bola próxima ao meio campo: 6000 cm
        ball_detections = create_mock_detections([[600, 200]], [99], [0])
        
        transformer = MockTransformer()
        
        offside_ids = detector.detect(
            detections=detections,
            players_team_id=team_ids,
            ball_detections=ball_detections,
            transformer=transformer,
            config=pitch_config
        )
        
        # Jogador 3 deve estar em impedimento
        assert 3 in offside_ids, "Jogador 3 deveria estar em impedimento"
        assert 1 not in offside_ids, "Jogador 1 não deveria estar em impedimento"
        assert 2 not in offside_ids, "Jogador 2 não deveria estar em impedimento"
    
    def test_no_offside_when_aligned(self):
        """Testa que não há impedimento quando atacante está alinhado com defensor."""
        config = OffsideConfig(
            debounce_frames=1,
            min_defenders=2,
            depth_axis='x',
            position_tolerance_cm=100.0
        )
        detector = OffsideDetector(config=config)
        pitch_config = SoccerPitchConfiguration()
        
        # Atacante alinhado com penúltimo defensor
        positions = [
            [400, 200],  # Time 0: 4000 cm
            [750, 200],  # Time 0, atacante alinhado: 7500 cm
            [700, 210],  # Time 1, defensor: 7000 cm
            [800, 210],  # Time 1, último defensor: 8000 cm
        ]
        
        tracker_ids = [1, 2, 3, 4]
        team_ids = np.array([0, 0, 1, 1])
        
        detections = create_mock_detections(positions, tracker_ids)
        ball_detections = create_mock_detections([[600, 200]], [99], [0])
        transformer = MockTransformer()
        
        offside_ids = detector.detect(
            detections=detections,
            players_team_id=team_ids,
            ball_detections=ball_detections,
            transformer=transformer,
            config=pitch_config
        )
        
        # Nenhum jogador deve estar em impedimento (dentro da tolerância)
        assert len(offside_ids) == 0, "Não deveria haver impedimento com jogadores alinhados"
    
    def test_no_offside_in_own_half(self):
        """Testa que não há impedimento na própria metade do campo."""
        config = OffsideConfig(debounce_frames=1, min_defenders=2, depth_axis='x')
        detector = OffsideDetector(config=config)
        pitch_config = SoccerPitchConfiguration()
        
        # Todos na própria metade
        positions = [
            [300, 200],  # Time 0: 3000 cm (própria metade)
            [400, 200],  # Time 0: 4000 cm (própria metade)
            [700, 210],  # Time 1: 7000 cm (metade adversária para time 0)
            [800, 210],  # Time 1: 8000 cm
        ]
        
        tracker_ids = [1, 2, 3, 4]
        team_ids = np.array([0, 0, 1, 1])
        
        detections = create_mock_detections(positions, tracker_ids)
        ball_detections = create_mock_detections([[350, 200]], [99], [0])
        transformer = MockTransformer()
        
        offside_ids = detector.detect(
            detections=detections,
            players_team_id=team_ids,
            ball_detections=ball_detections,
            transformer=transformer,
            config=pitch_config
        )
        
        assert len(offside_ids) == 0, "Não deveria haver impedimento na própria metade"
    
    def test_debounce_mechanism(self):
        """Testa o mecanismo de debounce temporal."""
        config = OffsideConfig(
            debounce_frames=3,
            min_defenders=2,
            depth_axis='x'
        )
        detector = OffsideDetector(config=config)
        pitch_config = SoccerPitchConfiguration()
        
        # Posições que indicam impedimento
        positions = [
            [400, 200],  # Time 0
            [900, 200],  # Time 0, atacante em impedimento
            [700, 210],  # Time 1
            [800, 210],  # Time 1
        ]
        
        tracker_ids = [1, 2, 3, 4]
        team_ids = np.array([0, 0, 1, 1])
        
        detections = create_mock_detections(positions, tracker_ids)
        ball_detections = create_mock_detections([[600, 200]], [99], [0])
        transformer = MockTransformer()
        
        # Frame 1: não deve detectar (buffer insuficiente)
        offside_ids = detector.detect(
            detections, team_ids, ball_detections, transformer, pitch_config
        )
        assert len(offside_ids) == 0, "Frame 1: buffer insuficiente"
        
        # Frame 2: ainda não deve detectar
        offside_ids = detector.detect(
            detections, team_ids, ball_detections, transformer, pitch_config
        )
        assert len(offside_ids) == 0, "Frame 2: buffer insuficiente"
        
        # Frame 3: agora deve detectar
        offside_ids = detector.detect(
            detections, team_ids, ball_detections, transformer, pitch_config
        )
        assert 2 in offside_ids, "Frame 3: deve detectar impedimento"
    
    def test_insufficient_defenders(self):
        """Testa que não avalia impedimento com poucos defensores."""
        config = OffsideConfig(
            debounce_frames=1,
            min_defenders=2,
            depth_axis='x'
        )
        detector = OffsideDetector(config=config)
        pitch_config = SoccerPitchConfiguration()
        
        # Apenas 1 defensor
        positions = [
            [400, 200],  # Time 0
            [900, 200],  # Time 0, atacante
            [800, 210],  # Time 1, único defensor
        ]
        
        tracker_ids = [1, 2, 3]
        team_ids = np.array([0, 0, 1])
        
        detections = create_mock_detections(positions, tracker_ids)
        ball_detections = create_mock_detections([[600, 200]], [99], [0])
        transformer = MockTransformer()
        
        offside_ids = detector.detect(
            detections, team_ids, ball_detections, transformer, pitch_config
        )
        
        assert len(offside_ids) == 0, "Não deveria avaliar com menos de 2 defensores"
    
    def test_offside_behind_ball(self):
        """Testa que não há impedimento se atacante está atrás da bola."""
        config = OffsideConfig(debounce_frames=1, min_defenders=2, depth_axis='x')
        detector = OffsideDetector(config=config)
        pitch_config = SoccerPitchConfiguration()
        
        # Atacante além dos defensores mas atrás da bola
        positions = [
            [400, 200],  # Time 0
            [900, 200],  # Time 0, atacante
            [700, 210],  # Time 1
            [800, 210],  # Time 1
        ]
        
        tracker_ids = [1, 2, 3, 4]
        team_ids = np.array([0, 0, 1, 1])
        
        detections = create_mock_detections(positions, tracker_ids)
        
        # Bola mais à frente que o atacante: 10000 cm
        ball_detections = create_mock_detections([[1000, 200]], [99], [0])
        transformer = MockTransformer()
        
        offside_ids = detector.detect(
            detections, team_ids, ball_detections, transformer, pitch_config
        )
        
        assert len(offside_ids) == 0, "Não deveria haver impedimento se atacante está atrás da bola"
    
    def test_annotation(self):
        """Testa a anotação visual de impedimento."""
        config = OffsideConfig(enable_annotations=True, circle_radius=30)
        detector = OffsideDetector(config=config)
        
        # Cria frame mock
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        positions = [[320, 240], [400, 300]]
        tracker_ids = [1, 2]
        detections = create_mock_detections(positions, tracker_ids)
        
        offside_ids = [2]  # Jogador 2 em impedimento
        
        annotated_frame = detector.annotate(frame, detections, offside_ids)
        
        # Verifica que o frame foi modificado (não é igual ao original)
        assert not np.array_equal(frame, annotated_frame), "Frame deveria ter sido anotado"
        
        # Verifica que há pixels não-pretos (círculo e texto foram desenhados)
        assert np.any(annotated_frame > 0), "Deveria haver anotações visuais no frame"


class TestDetectAndAnnotateOffside:
    """Testes para a função de conveniência."""
    
    def test_function_without_state(self):
        """Testa a função sem manter estado."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        positions = [
            [400, 200],
            [900, 200],  # Em impedimento
            [700, 210],
            [800, 210],
        ]
        
        tracker_ids = [1, 2, 3, 4]
        team_ids = np.array([0, 0, 1, 1])
        
        detections = create_mock_detections(positions, tracker_ids)
        ball_detections = create_mock_detections([[600, 200]], [99], [0])
        transformer = MockTransformer()
        pitch_config = SoccerPitchConfiguration()
        
        offside_config = OffsideConfig(debounce_frames=1, depth_axis='x')
        
        annotated_frame, offside_ids = detect_and_annotate_offside(
            frame=frame,
            detections=detections,
            players_team_id=team_ids,
            ball_detections=ball_detections,
            transformer=transformer,
            config=pitch_config,
            offside_config=offside_config
        )
        
        assert isinstance(annotated_frame, np.ndarray), "Deveria retornar frame anotado"
        assert isinstance(offside_ids, list), "Deveria retornar lista de IDs"
        assert 2 in offside_ids, "Jogador 2 deveria estar em impedimento"
    
    def test_empty_detections(self):
        """Testa com detecções vazias."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = sv.Detections.empty()
        team_ids = np.array([])
        ball_detections = sv.Detections.empty()
        transformer = MockTransformer()
        pitch_config = SoccerPitchConfiguration()
        
        annotated_frame, offside_ids = detect_and_annotate_offside(
            frame, detections, team_ids, ball_detections, transformer, pitch_config
        )
        
        assert len(offside_ids) == 0, "Não deveria detectar impedimento sem jogadores"
        assert np.array_equal(frame, annotated_frame), "Frame não deveria ser modificado"


class TestOffsideConfig:
    """Testes para a configuração de impedimento."""
    
    def test_default_config(self):
        """Testa configuração padrão."""
        config = OffsideConfig()
        
        assert config.debounce_frames == 5
        assert config.min_defenders == 2
        assert config.depth_axis == 'x'
        assert config.enable_annotations is True
    
    def test_custom_config(self):
        """Testa configuração customizada."""
        config = OffsideConfig(
            debounce_frames=10,
            min_defenders=1,
            depth_axis='y',
            position_tolerance_cm=100.0,
            enable_annotations=False
        )
        
        assert config.debounce_frames == 10
        assert config.min_defenders == 1
        assert config.depth_axis == 'y'
        assert config.position_tolerance_cm == 100.0
        assert config.enable_annotations is False


if __name__ == "__main__":
    # Permite executar os testes diretamente
    pytest.main([__file__, "-v"])



