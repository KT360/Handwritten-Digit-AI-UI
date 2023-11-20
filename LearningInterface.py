import pygame
import sys
import numpy as np
import matplotlib.pyplot as plt
from MatureNetwork import MatureNetwork

# Initialize Pygame
pygame.init()

# Constants for the window size
WIDTH, HEIGHT = 560, 560  # Upscaled 28x28 grid
CELL_SIZE = 20  # Each cell is 20x20 pixels
GRID_WIDTH, GRID_HEIGHT = WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE
LABEL_WIDTH = 200  # Width for the label area
# Create the window
screen = pygame.display.set_mode((WIDTH + LABEL_WIDTH, HEIGHT))
pygame.display.set_caption("Draw and Guess")

drawing_surface = pygame.Surface((WIDTH, HEIGHT))
# Create a separate surface for the label
label_surface = pygame.Surface((LABEL_WIDTH, HEIGHT))

# Set the background color
screen.fill((255, 255, 255))

drawing_surface.fill((0,0,0))

# Set up font
font = pygame.font.Font(None, 36)

num_font = pygame.font.Font(None, 50)

solver = MatureNetwork()

guessed_value = 0

##TODO: Something about the drawing timing is making it so that the center black cell is getting drawn over
##Also reduce strength when mouse is first clicked
def draw_antialiased_pixel(pos, strength):
    x, y = pos[0], pos[1]
    # Draw the main cell
    pygame.draw.rect(drawing_surface, (255, 255, 255), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    
    # Draw anti-aliased edges
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:  # Skip the center cell
                continue
            edge_strength = (1 - strength) * 255
            new_x, new_y = (x + dx) * CELL_SIZE, (y + dy) * CELL_SIZE
            pygame.draw.rect(drawing_surface, (edge_strength, edge_strength, edge_strength), (new_x, new_y, CELL_SIZE, CELL_SIZE))
    



def get_cell_index_from_pos(pos):
    # Convert pixel position to cell indices
    cell_x, cell_y = pos[0] // CELL_SIZE, pos[1] // CELL_SIZE
    return cell_x, cell_y



def update_label(new_text):
    global label_text, text_surface
    label_text = new_text
    text_surface = font.render(label_text, True, (0, 0, 0))


#R,G,B values are going to be the same, so can treat first value as grey scale
#Simply get the first color value within a cell
def get_pixel_data():
    
    data = np.zeros((28,28))
    dy = 0
    dx = 0 

    for y in range(0, GRID_HEIGHT):

        for x in range(0, GRID_WIDTH):

            value = drawing_surface.get_at((dx, dy))[0]/255 #normalize
            data[y][x] = value

            dx+=CELL_SIZE
        
        dx = 0    
        dy += CELL_SIZE

    return data






running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                # Start drawing
                mouse_x, mouse_y = pygame.mouse.get_pos()
                cell_x, cell_y = get_cell_index_from_pos((mouse_x, mouse_y))
    
                draw_antialiased_pixel((cell_x, cell_y), strength=1.0)  # Full strength for center cell
       
                
        
        if event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:  # Left button held down
                mouse_x, mouse_y = pygame.mouse.get_pos()
                cell_x, cell_y = get_cell_index_from_pos((mouse_x, mouse_y))
                draw_antialiased_pixel((cell_x, cell_y), strength=0.5)  # Partial strength for edges
    
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            cell_x, cell_y = get_cell_index_from_pos((mouse_x, mouse_y))

            pixel_array = get_pixel_data()
            flattened = pixel_array.reshape(pixel_array.size, -1).reshape((784)) ##Have to reshape twice in order to get a truly flattened array

            guessed_value = solver.guessNumber(flattened)

            

            

            

    
    text = font.render("Number Entered: ", False, (0, 0, 0))
    number = num_font.render(str(guessed_value), False, (127,35,40))
    label_surface.fill((255, 255, 255))  # Clear the label surface
    label_surface.blit(text, (10, 10))
    label_surface.blit(number, (10, 30))


    screen.blit(drawing_surface, (0, 0))  # Blit the drawing surface onto the screen
    screen.blit(label_surface, (WIDTH, 0))

    pygame.display.flip()

pygame.quit()
sys.exit()