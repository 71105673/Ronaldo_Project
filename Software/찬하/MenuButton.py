import pygame

# 필요한 색상 정의 (WHITE가 없어서 추가했습니다)
WHITE = (255, 255, 255)
BUTTON_COLOR = (100, 100, 100)
# 마우스를 올렸을 때 색상 변경을 위해 HOVER_COLOR도 정의하면 좋습니다.
HOVER_COLOR = (200, 200, 200) 

class MenuButton:
    # 1. __init__ 메서드에 sound=None 인자 추가
    def __init__(self, text, x, y, width, height, font, action=None, sound=None):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.font = font
        self.action = action
        self.sound = sound  # ★ 추가: sound 인자를 self.sound에 저장
        self.text_surf = self.font.render(self.text, True, WHITE)
        self.text_rect = self.text_surf.get_rect(center=self.rect.center)

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        is_hover = self.rect.collidepoint(mouse_pos)

        # 기존 코드보다 마우스를 올렸을 때 색이 바뀌는 것이 더 직관적입니다.
        color = HOVER_COLOR if is_hover else WHITE
        self.text_surf = self.font.render(self.text, True, color)
        surface.blit(self.text_surf, self.text_rect)

        line_start = (self.rect.left, self.rect.bottom)
        line_end = (self.rect.right, self.rect.bottom)
        pygame.draw.line(surface, WHITE, line_start, line_end, 1)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                # 2. handle_event 메서드에 사운드 재생 코드 추가
                if self.sound:      # ★ 추가: self.sound가 존재하면
                    self.sound.play() # ★ 추가: 재생한다
                
                if self.action:
                    self.action()
                    return True
        return False