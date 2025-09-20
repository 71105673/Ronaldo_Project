import pygame
from Config import WHITE, BUTTON_COLOR, HOVER_COLOR

class ImageButton:
    """이미지를 사용하는 버튼 클래스"""
    def __init__(self, image_path: str, x: int, y: int, width: int = None, height: int = None, action=None, sound=None):
        self.action = action
        self.sound = sound
        self.is_hovered = False

        try:
            original_image = pygame.image.load(image_path).convert_alpha()
            
            # 기본 이미지 설정
            self.image = pygame.transform.scale(original_image, (width, height)) if width and height else original_image
            self.rect = self.image.get_rect(center=(x, y))

            # 호버 이미지 설정 (105% 크기)
            hover_width = int(self.rect.width * 1.05)
            hover_height = int(self.rect.height * 1.05)
            self.hover_image = pygame.transform.scale(original_image, (hover_width, hover_height))

        except pygame.error as e:
            print(f"이미지 로드 오류: {image_path} - {e}")
            # 이미지가 없을 경우 대체 사각형 생성
            w, h = width or 100, height or 50
            self.image = pygame.Surface((w, h)); self.image.fill(BUTTON_COLOR)
            self.hover_image = pygame.Surface((w, h)); self.hover_image.fill(HOVER_COLOR)
            self.rect = self.image.get_rect(center=(x, y))

    def update(self):
        """마우스 위치에 따라 호버 상태를 업데이트합니다."""
        self.is_hovered = self.rect.collidepoint(pygame.mouse.get_pos())

    def draw(self, screen: pygame.Surface):
        """화면에 버튼을 그립니다."""
        current_image = self.hover_image if self.is_hovered else self.image
        # 호버 시 이미지가 커지므로 중앙을 다시 계산하여 그립니다.
        draw_rect = current_image.get_rect(center=self.rect.center)
        screen.blit(current_image, draw_rect)

    def handle_event(self, event: pygame.event.Event):
        """클릭 이벤트를 처리합니다."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.is_hovered:
            if self.sound:
                self.sound.play()
            if self.action:
                self.action()
                return True
        return False


class MenuButton:
    """텍스트를 사용하는 메뉴 버튼 클래스"""
    def __init__(self, text: str, x: int, y: int, width: int, height: int, font: pygame.font.Font, action=None, sound=None):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.font = font
        self.action = action
        self.sound = sound
        self.is_hovered = False

    def update(self):
        """마우스 위치에 따라 호버 상태를 업데이트합니다."""
        self.is_hovered = self.rect.collidepoint(pygame.mouse.get_pos())
    
    def draw(self, surface: pygame.Surface):
        """화면에 텍스트 버튼과 밑줄을 그립니다."""
        color = HOVER_COLOR if self.is_hovered else WHITE
        text_surf = self.font.render(self.text, True, color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

        # 밑줄 그리기
        line_start = (self.rect.left, self.rect.bottom)
        line_end = (self.rect.right, self.rect.bottom)
        pygame.draw.line(surface, WHITE, line_start, line_end, 1)

    def handle_event(self, event: pygame.event.Event):
        """클릭 이벤트를 처리합니다."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.is_hovered:
            if self.sound:
                self.sound.play()
            if self.action:
                self.action()
                return True
        return False