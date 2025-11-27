"""
Módulo para detecção e anotação de impedimento em jogos de futebol.

Este módulo implementa a lógica simplificada de detecção de impedimento,
considerando as posições dos jogadores, times e bola no campo.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration


@dataclass
class OffsideConfig:
    """Configurações para detecção de impedimento."""
    
    # Número de frames consecutivos necessários para confirmar impedimento
    debounce_frames: int = 5
    
    # Número mínimo de defensores necessários para avaliar impedimento
    min_defenders: int = 2
    
    # Eixo de profundidade do campo ('x' para horizontal, 'y' para vertical)
    depth_axis: str = 'x'
    
    # Tolerância em cm para considerar posições próximas
    position_tolerance_cm: float = 50.0
    
    # Flag para ativar anotações visuais
    enable_annotations: bool = True
    
    # Cor para marcação de impedimento (formato BGR)
    offside_color: Tuple[int, int, int] = (0, 0, 255)  # Vermelho
    
    # Tamanho do círculo de marcação
    circle_radius: int = 30
    
    # Espessura da linha do círculo
    circle_thickness: int = 3


class OffsideDetector:
    """
    Detector de impedimento que mantém estado entre frames.
    """
    
    def __init__(self, config: Optional[OffsideConfig] = None):
        """
        Inicializa o detector de impedimento.
        
        Args:
            config: Configurações do detector. Se None, usa configurações padrão.
        """
        self.config = config or OffsideConfig()
        
        # Buffer para debounce: {tracker_id: deque de bools indicando impedimento}
        self._offside_buffer: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.config.debounce_frames)
        )
        
        # Último jogador que tocou a bola (para casos ambíguos)
        self._last_ball_player: Optional[int] = None
        
        # Histórico de posições da bola
        self._ball_positions: deque = deque(maxlen=30)
    
    def _get_depth_coordinate(self, point: np.ndarray) -> float:
        """
        Extrai a coordenada de profundidade baseada no eixo configurado.
        
        Args:
            point: Ponto 2D [x, y] em coordenadas do campo (cm).
            
        Returns:
            Coordenada de profundidade.
        """
        if self.config.depth_axis == 'x':
            return point[0]
        else:
            return point[1]
    
    def _determine_attacking_direction(
        self,
        players_xy_cm: np.ndarray,
        players_team_id: np.ndarray,
        ball_xy_cm: Optional[np.ndarray],
        pitch_config: SoccerPitchConfiguration
    ) -> Dict[int, int]:
        """
        Determina a direção de ataque para cada time.
        
        Args:
            players_xy_cm: Posições dos jogadores em cm no campo.
            players_team_id: IDs dos times dos jogadores.
            ball_xy_cm: Posição da bola em cm (se disponível).
            pitch_config: Configuração do campo.
            
        Returns:
            Dict mapeando team_id para direção: 1 = ataca para x crescente, -1 = ataca para x decrescente
        """
        if len(players_xy_cm) == 0:
            return {}
        
        # Calcula centro de massa de cada time
        team_0_positions = players_xy_cm[players_team_id == 0]
        team_1_positions = players_xy_cm[players_team_id == 1]
        
        if len(team_0_positions) == 0 or len(team_1_positions) == 0:
            return {}
        
        team_0_center = np.mean(team_0_positions[:, 0])  # média em x
        team_1_center = np.mean(team_1_positions[:, 0])
        
        # Time com menor x médio ataca para a direita (+1), maior ataca para esquerda (-1)
        if team_0_center < team_1_center:
            return {0: 1, 1: -1}
        else:
            return {0: -1, 1: 1}
    
    def _find_attacking_team(
        self,
        ball_xy_cm: Optional[np.ndarray],
        players_xy_cm: np.ndarray,
        players_team_id: np.ndarray
    ) -> Optional[int]:
        """
        Identifica o time atacante baseado na proximidade com a bola.
        
        Args:
            ball_xy_cm: Posição da bola em cm.
            players_xy_cm: Posições dos jogadores em cm.
            players_team_id: IDs dos times.
            
        Returns:
            ID do time atacante (0 ou 1) ou None se indeterminado.
        """
        if ball_xy_cm is None or len(players_xy_cm) == 0:
            return None
        
        # Encontra jogador mais próximo da bola
        distances = np.linalg.norm(players_xy_cm - ball_xy_cm, axis=1)
        closest_player_idx = np.argmin(distances)
        
        return int(players_team_id[closest_player_idx])
    
    def _check_offside_condition(
        self,
        attacker_pos: np.ndarray,
        defenders_pos: np.ndarray,
        ball_pos: Optional[np.ndarray],
        attacking_direction: int,
        pitch_config: SoccerPitchConfiguration
    ) -> bool:
        """
        Verifica se um atacante está em posição de impedimento.
        
        Condições simplificadas:
        1. Atacante está na metade adversária
        2. Atacante está mais próximo da linha de gol que o penúltimo defensor
        3. Atacante está mais próximo da linha de gol que a bola
        
        Args:
            attacker_pos: Posição do atacante em cm.
            defenders_pos: Posições dos defensores em cm.
            ball_pos: Posição da bola em cm.
            attacking_direction: Direção de ataque (+1 ou -1).
            pitch_config: Configuração do campo.
            
        Returns:
            True se o atacante está em impedimento.
        """
        if len(defenders_pos) < self.config.min_defenders:
            # Não há defensores suficientes para avaliar
            return False
        
        depth = self._get_depth_coordinate(attacker_pos)
        field_center = pitch_config.length / 2
        
        # Verifica se está na metade adversária
        if attacking_direction > 0:
            in_opponent_half = depth > field_center
            goal_line = pitch_config.length
        else:
            in_opponent_half = depth < field_center
            goal_line = 0
        
        if not in_opponent_half:
            return False
        
        # Encontra o penúltimo defensor
        defenders_depth = np.array([self._get_depth_coordinate(pos) for pos in defenders_pos])
        
        if attacking_direction > 0:
            # Atacando para a direita: defensores com maior x estão mais próximos do gol
            sorted_defenders = np.sort(defenders_depth)[::-1]
        else:
            # Atacando para a esquerda: defensores com menor x estão mais próximos do gol
            sorted_defenders = np.sort(defenders_depth)
        
        # Penúltimo defensor (segundo mais próximo do gol)
        if len(sorted_defenders) >= 2:
            second_last_defender = sorted_defenders[1]
        else:
            second_last_defender = sorted_defenders[0]
        
        # Verifica se atacante está mais próximo do gol que o penúltimo defensor
        if attacking_direction > 0:
            beyond_defenders = depth > second_last_defender + self.config.position_tolerance_cm
        else:
            beyond_defenders = depth < second_last_defender - self.config.position_tolerance_cm
        
        if not beyond_defenders:
            return False
        
        # Verifica se atacante está mais próximo do gol que a bola
        if ball_pos is not None:
            ball_depth = self._get_depth_coordinate(ball_pos)
            if attacking_direction > 0:
                beyond_ball = depth > ball_depth + self.config.position_tolerance_cm
            else:
                beyond_ball = depth < ball_depth - self.config.position_tolerance_cm
            
            if not beyond_ball:
                return False
        
        return True
    
    def detect(
        self,
        detections: sv.Detections,
        players_team_id: np.ndarray,
        ball_detections: sv.Detections,
        transformer: ViewTransformer,
        config: SoccerPitchConfiguration
    ) -> List[int]:
        """
        Detecta jogadores em impedimento no frame atual.
        
        Args:
            detections: Detecções de todos os jogadores.
            players_team_id: Array com IDs dos times dos jogadores.
            ball_detections: Detecções da bola.
            transformer: Transformador de visão para converter coordenadas.
            config: Configuração do campo de futebol.
            
        Returns:
            Lista de tracker_ids dos jogadores em impedimento confirmado.
        """
        if len(detections) == 0 or len(players_team_id) == 0:
            return []
        
        # Obtém posições dos jogadores em coordenadas de tela
        players_xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        
        # Transforma para coordenadas do campo (cm)
        players_xy_cm = transformer.transform_points(points=players_xy)
        
        # Obtém posição da bola se disponível
        ball_xy_cm = None
        if len(ball_detections) > 0:
            ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            ball_xy_cm = transformer.transform_points(points=ball_xy)[0]
            self._ball_positions.append(ball_xy_cm)
        elif len(self._ball_positions) > 0:
            # Usa última posição conhecida
            ball_xy_cm = self._ball_positions[-1]
        
        # Determina direção de ataque de cada time
        attack_directions = self._determine_attacking_direction(
            players_xy_cm, players_team_id, ball_xy_cm, config
        )
        
        if not attack_directions:
            return []
        
        # Identifica time atacante
        attacking_team = self._find_attacking_team(ball_xy_cm, players_xy_cm, players_team_id)
        
        # Se não conseguir determinar pelo contato com a bola, assume ambos times atacando
        teams_to_check = [attacking_team] if attacking_team is not None else [0, 1]
        
        # Verifica impedimento para cada jogador
        offside_candidates = []
        
        for team_id in teams_to_check:
            if team_id not in attack_directions:
                continue
            
            attackers_mask = players_team_id == team_id
            defenders_mask = players_team_id != team_id
            
            attackers_indices = np.where(attackers_mask)[0]
            defenders_xy_cm = players_xy_cm[defenders_mask]
            
            attack_direction = attack_directions[team_id]
            
            for attacker_idx in attackers_indices:
                attacker_pos = players_xy_cm[attacker_idx]
                tracker_id = detections.tracker_id[attacker_idx]
                
                is_offside = self._check_offside_condition(
                    attacker_pos,
                    defenders_xy_cm,
                    ball_xy_cm,
                    attack_direction,
                    config
                )
                
                # Adiciona ao buffer de debounce
                self._offside_buffer[tracker_id].append(is_offside)
                
                # Confirma impedimento se a maioria dos frames recentes indica impedimento
                if len(self._offside_buffer[tracker_id]) >= self.config.debounce_frames:
                    offside_count = sum(self._offside_buffer[tracker_id])
                    if offside_count >= self.config.debounce_frames * 0.7:  # 70% dos frames
                        offside_candidates.append(tracker_id)
        
        return offside_candidates
    
    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        offside_ids: List[int]
    ) -> np.ndarray:
        """
        Anota o frame com marcações de impedimento.
        
        Args:
            frame: Frame a ser anotado.
            detections: Detecções dos jogadores.
            offside_ids: Lista de tracker_ids em impedimento.
            
        Returns:
            Frame anotado.
        """
        if not self.config.enable_annotations or len(offside_ids) == 0:
            return frame
        
        annotated_frame = frame.copy()
        
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id in offside_ids:
                # Obtém posição do jogador
                xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)[i]
                x, y = int(xy[0]), int(xy[1])
                
                # Desenha círculo vermelho ao redor do jogador
                cv2.circle(
                    annotated_frame,
                    (x, y),
                    self.config.circle_radius,
                    self.config.offside_color,
                    self.config.circle_thickness
                )
                
                # Adiciona texto "IMP" acima do jogador
                text = "IMP"
                font = cv2.FONT_HERSHEY_BOLD
                font_scale = 0.8
                thickness = 2
                
                # Calcula tamanho do texto para criar fundo
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, thickness
                )
                
                # Posição do texto (acima do círculo)
                text_x = x - text_width // 2
                text_y = y - self.config.circle_radius - 10
                
                # Desenha fundo preto para o texto
                cv2.rectangle(
                    annotated_frame,
                    (text_x - 5, text_y - text_height - 5),
                    (text_x + text_width + 5, text_y + baseline + 5),
                    (0, 0, 0),
                    -1
                )
                
                # Desenha o texto
                cv2.putText(
                    annotated_frame,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    self.config.offside_color,
                    thickness
                )
        
        return annotated_frame


def detect_and_annotate_offside(
    frame: np.ndarray,
    detections: sv.Detections,
    players_team_id: np.ndarray,
    ball_detections: sv.Detections,
    transformer: ViewTransformer,
    config: SoccerPitchConfiguration,
    detector: Optional[OffsideDetector] = None,
    offside_config: Optional[OffsideConfig] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Função principal para detectar e anotar impedimento.
    
    Esta é uma função de conveniência que pode ser usada sem manter estado.
    Para uso com estado persistente, crie uma instância de OffsideDetector.
    
    Args:
        frame: Frame do vídeo a ser anotado.
        detections: Detecções dos jogadores (sv.Detections).
        players_team_id: Array numpy com IDs dos times (0 ou 1).
        ball_detections: Detecções da bola.
        transformer: ViewTransformer para conversão de coordenadas.
        config: Configuração do campo de futebol.
        detector: Detector de impedimento existente (mantém estado). Se None, cria novo.
        offside_config: Configurações do detector de impedimento.
        
    Returns:
        Tupla contendo:
        - Frame anotado com marcações de impedimento
        - Lista de tracker_ids dos jogadores em impedimento
    """
    if detector is None:
        detector = OffsideDetector(config=offside_config)
    
    # Detecta impedimento
    offside_ids = detector.detect(
        detections=detections,
        players_team_id=players_team_id,
        ball_detections=ball_detections,
        transformer=transformer,
        config=config
    )
    
    # Anota frame
    annotated_frame = detector.annotate(
        frame=frame,
        detections=detections,
        offside_ids=offside_ids
    )
    
    return annotated_frame, offside_ids



