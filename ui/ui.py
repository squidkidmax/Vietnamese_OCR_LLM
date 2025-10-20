import pygame
import sys
import tkinter as tk
from tkinter import filedialog
import threading
from pathlib import Path
import sys

# Add project root to path to import tool module
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from tool.vintern_ocr import VinternOCR

# --- Initialize ---
pygame.init()
pygame.display.set_caption("OCR Application Menu")

# --- Window setup ---
WIDTH, HEIGHT = 800, 500
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
FONT = pygame.font.SysFont("arial", 30)
SMALL_FONT = pygame.font.SysFont("arial", 20)

# --- Colors ---
WHITE = (255, 255, 255)
LIGHT_BLUE = (100, 149, 237)
DARK_BLUE = (30, 60, 150)
GRAY = (220, 220, 220)
BLACK = (0, 0, 0)

# --- App State ---
STATE = "menu"  # 'menu', 'convert', or 'showing_result'
attached_image = None
ocr_result = None
is_processing = False


# --- Button class ---
class Button:
    def __init__(self, text, pos, size):
        self.text = text
        self.rect = pygame.Rect(pos, size)
        self.color = LIGHT_BLUE
        self.hover_color = DARK_BLUE

    def draw(self, surface):
        mouse = pygame.mouse.get_pos()
        color = self.hover_color if self.rect.collidepoint(mouse) else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        text_surf = FONT.render(self.text, True, WHITE)
        surface.blit(
            text_surf,
            (
                self.rect.centerx - text_surf.get_width() // 2,
                self.rect.centery - text_surf.get_height() // 2,
            ),
        )

# --- File Dialog Function ---
def open_file_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide main Tk window
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")],
    )
    root.destroy()
    return file_path

# --- Buttons ---
attach_button = Button("Attach Image", (WIDTH//2 - 150, 180), (300, 60))
convert_button = Button("Convert to Text", (WIDTH//2 - 150, 280), (300, 60))

# --- Main Menu ---
def draw_menu():
    SCREEN.fill(WHITE)
    title = FONT.render("OCR Application", True, BLACK)
    SCREEN.blit(title, (WIDTH//2 - title.get_width()//2, 80))
    attach_button.draw(SCREEN)
    convert_button.draw(SCREEN)

    if attached_image:
        name_text = SMALL_FONT.render("Attached: " + attached_image.split("/")[-1], True, BLACK)
        SCREEN.blit(name_text, (WIDTH//2 - name_text.get_width()//2, 380))

# --- Converting Screen ---
def process_image():
    ocr = VinternOCR()
    global ocr_result, is_processing, STATE
    try:
        ocr_result = ocr.process_image(attached_image)
        STATE = "showing_result"
    except Exception as e:
        ocr_result = f"Error: {str(e)}"
    finally:
        is_processing = False

def draw_convert_screen():
    SCREEN.fill(WHITE)
    
    if is_processing:
        title = FONT.render("Converting Image to Text...", True, BLACK)
    else:
        title = FONT.render("Press SPACE to start conversion", True, BLACK)
    SCREEN.blit(title, (WIDTH//2 - title.get_width()//2, 40))

    if attached_image:
        try:
            img = pygame.image.load(attached_image)
            img = pygame.transform.scale(img, (400, 300))
            SCREEN.blit(img, (WIDTH//2 - 200, 100))
        except Exception as e:
            err_text = SMALL_FONT.render(f"Error loading image: {e}", True, (255, 0, 0))
            SCREEN.blit(err_text, (WIDTH//2 - err_text.get_width()//2, HEIGHT//2))

    back_text = SMALL_FONT.render("Press [ESC] to go back", True, DARK_BLUE)
    SCREEN.blit(back_text, (20, HEIGHT - 40))

def draw_result_screen():
    SCREEN.fill(WHITE)
    title = FONT.render("OCR Result", True, BLACK)
    SCREEN.blit(title, (WIDTH//2 - title.get_width()//2, 40))

    # Display the OCR result
    if ocr_result:
        # Split the result into lines to fit the screen
        words = ocr_result.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            test_line = " ".join(current_line)
            if SMALL_FONT.size(test_line)[0] > WIDTH - 60:
                lines.append(" ".join(current_line[:-1]))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))

        # Display lines
        y_pos = 100
        for line in lines[:15]:  # Limit to 15 lines to prevent overflow
            text_surface = SMALL_FONT.render(line, True, BLACK)
            SCREEN.blit(text_surface, (30, y_pos))
            y_pos += 25

    back_text = SMALL_FONT.render("Press [ESC] to go back to menu", True, DARK_BLUE)
    SCREEN.blit(back_text, (20, HEIGHT - 40))


# --- Main Loop ---
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if STATE == "menu":
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if attach_button.rect.collidepoint(event.pos):
                    attached_image = open_file_dialog()
                elif convert_button.rect.collidepoint(event.pos):
                    if attached_image:
                        STATE = "convert"

        elif STATE == "convert":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    STATE = "menu"
                elif event.key == pygame.K_SPACE and not is_processing:
                    is_processing = True
                    threading.Thread(target=process_image, daemon=True).start()
        
        elif STATE == "showing_result":
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                STATE = "menu"
                ocr_result = None

    if STATE == "menu":
        draw_menu()
    elif STATE == "convert":
        draw_convert_screen()
    elif STATE == "showing_result":
        draw_result_screen()

    pygame.display.flip()
