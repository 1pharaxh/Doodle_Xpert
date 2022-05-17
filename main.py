from PIL import Image, ImageEnhance, ImageFilter
import pygame
from pygame.locals import *
import cv2 as cv
import numpy as np
import tensorflow as tf
import NeuralNetwork
"""
    Description: This function is used to load the model and 
                 run prediction on the screenshot from doodle
                 pad area. It also takes care of the distinct
                 needs of MLP and CNN models
    Parameters: argv - 0 for CNN and 1 for MLP
    Return:     Prediction of the class.    
"""


def machineLearning(argv):
    # load the model
    if argv == 0:
        model = tf.keras.models.load_model('models/CNN.h5')
    else:
        model = tf.keras.models.load_model('models/MLP.h5')
    # read the image
    img = cv.imread("files/img.png", 0)
    # convert it into a numpy array
    img = np.invert(np.array([img]))
    # if the argv is for MLP the reshape / flatten the
    # image into 1D array
    if argv == 1:
        img = img.reshape(1, 784)
    # predict the image
    prediction = model.predict(img)
    # Dictionary of classes
    CLASSES = {0: "Apple", 1: "Banana", 2: "Book", 3: "Cup", 4: "Ladder"}
    # return the value of the highest probability class
    print(np.argmax(prediction))
    return CLASSES[np.argmax(prediction)]

    # for debugging uncomment the following lines
    # and paste before return.
    # plt.imshow(img[0], cmap=plt.cm.binary)
    # plt.show()


"""
    Description: This function is used to capture the doodle
                 pad area and scale it down to 28x28 pixels 
                 from 560x560 pixels.
    Parameters:  Pygame object, topleft and bottomright coordinates of
                 capture area and the file name to save the image
    Return:      None
"""


def screenshot(obj, file_name, topleft, bottomright):
    # Calculate the region of interest
    size = bottomright[0] - topleft[0], bottomright[1] - topleft[1]
    # Create a new image of the appropriate size
    img = pygame.Surface(size)
    img.blit(obj, (0, 0), (topleft, size))
    # Save the image
    pygame.image.save(img, file_name)
    # Open the image with PIL library
    im = Image.open("files/img.png")
    # Resize the image and also sharpen it
    im_resized = im.resize((28, 28), Image.ANTIALIAS).filter(
        ImageFilter.SHARPEN)
    # Increase image brightness
    enhancer = ImageEnhance.Brightness(im_resized)
    # Values > 1 increase the brightness
    factor = 1.2
    im_resized = enhancer.enhance(factor)
    # Save the image
    im_resized.save("files/img.png")


"""
    Description: This function is the gameloop for the mainmenu
    Parameters:  None
    Return:      None
"""


def mainmenu():
    # Intialize pygame
    pygame.init()
    # Set the screen size
    screen = pygame.display.set_mode((800, 800), 0, 32)
    # Set the BG color
    background = (181, 254, 131)
    screen.fill(background)
    # Clock to keep track of FPS
    clock = pygame.time.Clock()

    while True:
        # X and Y coordinates of the cursor
        mx, my = pygame.mouse.get_pos()
        # Creating 2 buttons, 1 for using MLP and 1 for using CNN
        button_1 = pygame.Rect(50, 300, 325, 190)
        button_2 = pygame.Rect(425, 300, 325, 190)
        # Drawing the buttons
        pygame.draw.rect(screen, (249, 255, 164), button_1)
        pygame.draw.rect(screen, (249, 255, 164), button_2)
        # Loading the sticker computer image
        image = pygame.image.load('files\computer.png')
        # Downsizing the image
        image = pygame.transform.scale(image, (240, 190))
        # Blitting the image
        screen.blit(image, (510, 590))
        # Loading the font with size 24
        font = pygame.font.Font('files\Early GameBoy.ttf', 24)
        # Drawing text
        screen.blit(font.render('Choose a Machine Learning Model !',
                    True, (255, 161, 161)), (50, 250))
        # Loading font of different size for drawing the note
        font3 = pygame.font.Font('files\Early GameBoy.ttf', 18)

        screen.blit(font3.render('NOTE : Training AI may take ',
                                 True, (255, 161, 161)), (50, 600))
        screen.blit(font3.render('a while ! Depending on ',
                                 True, (255, 161, 161)), (50, 630))
        screen.blit(font3.render('your system ...',
                                 True, (255, 161, 161)), (50, 660))
        # Change color of the MLP button when cursor hovers over it
        if button_1.collidepoint((mx, my)):
            pygame.draw.rect(screen, (255, 213, 158), button_1)

        # Change color of the CNN button when cursor hovers over it
        if button_2.collidepoint((mx, my)):
            pygame.draw.rect(screen, (255, 213, 158), button_2)
        # Handle events (mouse click, key press, etc)
        for event in pygame.event.get():
            # When X is pressed, quit the game
            if event.type == pygame.QUIT:
                pygame.quit()
                break
            # When mouse is clicked
            if event.type == MOUSEBUTTONDOWN:
                # If left mouse button is clicked
                if event.button == 1:
                    # If mouse is hovering over MLP button
                    if button_1.collidepoint((mx, my)):
                        # Call NeuralNetwork.py with argv of MLP
                        NeuralNetwork.execute(1)
                        # Call game loop with argv of MLP
                        game(1)
                    # If mouse is hovering over CNN button
                    if button_2.collidepoint((mx, my)):
                        # Call NeuralNetwork.py with argv of CNN
                        NeuralNetwork.execute(0)
                        # Call game loop with argv of CNN
                        game(0)
        pygame.init()
        # Font for drawing the title
        font2 = pygame.font.Font('files\Early GameBoy.ttf', 48)
        screen.blit(font2.render("Doodle X-Perts", True, (0, 0, 0)), (80, 0))
        screen.blit(font.render('MULTI-LAYER ',
                    True, (255, 161, 161)), (83, 350))
        # Drawing button in-text
        screen.blit(font.render('PERCEPTRON',
                    True, (255, 161, 161)), (95, 400))

        screen.blit(font.render('CONVOLUTIONAL',
                    True, (255, 161, 161)), (435, 350))
        screen.blit(font.render('NEURAL NETWORK',
                    True, (255, 161, 161)), (430, 400))
        # Loading the brain image below title
        image2 = pygame.image.load('files/brain.gif')
        # Blitting the image
        screen.blit(image2, (0, 30))
        pygame.display.update()
        # Limit FPS to 60
        clock.tick(60)


"""
    Description: This function is the gameloop for the game, It makes the 
                 doodle pad, Brush and displays prediction to user.
    Parameters:  argv of MLP or CNN
    Return:      None
"""


def game(argv):
    # Initialize pygame
    pygame.init()
    # Set brush color to black
    BLACK = (0, 0, 0)
    # Setting the screen size
    screen = pygame.display.set_mode((800, 800), 0, 32)
    # Setting the BG color
    background = (247, 247, 247)
    screen.fill(background)
    # Clock to keep track of FPS
    clock = pygame.time.Clock()

    while True:
        # Getting mouse input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                break
            # if mouse is pressed
            elif event.type == pygame.MOUSEMOTION:
                # If left mouse button is pressed
                if event.buttons[0]:
                    # Calculating mouse position
                    last = (event.pos[0]-event.rel[0],
                            event.pos[1]-event.rel[1])
                    # Drawing a line from last position to current position
                    # 15 is the thickness of the line
                    pygame.draw.line(screen, BLACK, last, event.pos, 20)
                # If right mouse button is pressed
                if event.buttons[2]:
                    # Clear the screen
                    screen.fill(background)
            # If keyboard key is pressed
            elif event.type == pygame.KEYDOWN:
                # If 'Enter' key is pressed
                if event.key == pygame.K_RETURN:
                    # Take screen shot of the doodle pad area
                    screenshot(screen, "files/img.png", (240, 0),
                               (800, 560))
                    if argv == 1:
                        prediction = machineLearning(1)
                    if argv == 0:
                        prediction = machineLearning(0)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if button_3.collidepoint((mx, my)):
                        mainmenu()
        # Draw rectangle prediction area
        pygame.draw.rect(screen, (204, 242, 244), pygame.Rect(0, 0, 240, 800))
        # Draw rectangle for the prompt area
        pygame.draw.rect(screen, (164, 235, 243),
                         pygame.Rect(240, 560, 560, 560))
        button_3 = pygame.Rect(40, 610, 160, 140)
        pygame.draw.rect(screen, (183, 229, 221), button_3)

        mx, my = pygame.mouse.get_pos()
        if button_3.collidepoint((mx, my)):
            pygame.draw.rect(screen, (151, 196, 184), button_3)
        font3 = pygame.font.Font('files\Early GameBoy.ttf', 34)
        screen.blit(font3.render('BACK',
                                 True, (255, 161, 161)), (55, 655))
        # Text stored in an array because pygame won't accept a
        # newline character
        text = ['Start doodling on the', 'pad and press Enter',
                'For AI to guess',
                'your Drawing !']
        # Variable to store the text position
        y_value = 0
        font = pygame.font.Font('files\Early GameBoy.ttf', 24)
        # Loop to draw the text
        for i in range(len(text)):
            # Centering "Draw and press" text and then
            # Adjusting the position of "Enter to recognize" text
            screen.blit(font.render(
                text[i], True, (170, 170, 170)), (280, 600+y_value))
            y_value += 50
        # Exception Handling to catch any exception rendered during prediction
        try:
            if prediction:
                screen.blit(font.render('I think', True,
                            (206, 148, 97)), (40, 50))
                screen.blit(font.render('you drew', True,
                            (206, 148, 97)), (30, 100))
                screen.blit(font.render(f'{prediction}!', True,
                            (206, 148, 97)), (30, 150))
        except:
            pass

        pygame.display.update()
        # Limit FPS to 60
        clock.tick(60)


# Calling main function
if __name__ == "__main__":
    mainmenu()
