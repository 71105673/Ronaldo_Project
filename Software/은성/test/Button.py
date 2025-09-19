import pygame

WHITE = (255, 255, 255)
BUTTON_COLOR = (100, 100, 100)
HOVER_COLOR = (200, 200, 200) 

class ImageButton:
    def __init__(self, image_path, x, y, width=None, height=None, action=None, sound=None):
        self.action, self.sound, self.is_hovered = action, sound, False
        try:
            self.original_image = pygame.image.load(image_path).convert_alpha()
            scale_factor = 1.05
            self.image = pygame.transform.scale(self.original_image, (width, height)) if width and height else self.original_image
            hover_width = int(self.image.get_width() * scale_factor)
            hover_height = int(self.image.get_height() * scale_factor)
            self.hover_image = pygame.transform.scale(self.original_image, (hover_width, hover_height))
            self.rect = self.image.get_rect(center=(x, y))
        except pygame.error as e:
            print(f"이미지 로드 오류: {image_path} - {e}")
            self.image = pygame.Surface((width or 100, height or 50)); self.image.fill(BUTTON_COLOR)
            self.hover_image = pygame.Surface((width or 100, height or 50)); self.hover_image.fill((150,150,150))
            self.rect = self.image.get_rect(center=(x, y))

    def update(self):
        self.is_hovered = self.rect.collidepoint(pygame.mouse.get_pos())

    def draw(self, screen):
        current_image = self.hover_image if self.is_hovered else self.image
        screen.blit(current_image, current_image.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.is_hovered:
            if self.sound: self.sound.play()
            if self.action: self.action()

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
    
    def update(self):
        self.is_hovered = self.rect.collidepoint(pygame.mouse.get_pos())