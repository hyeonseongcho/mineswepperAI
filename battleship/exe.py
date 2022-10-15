import pygame

### pygame 전역변수 선언 ###


# colors
BLACK  = (0, 0, 0)
WHITE  = (255, 255, 255)
RED    = (255, 0, 0)
GREEN  = (0, 255, 0)
BLUE   = (0, 0, 255)

# screen
size   = [1280, 720]

# 시스템 관리
state = 0 # 0: 로비, 1: 설정, 2: 게임
done = False

# 게임 관리
turn = 0 # 0: P1, 1: P2

# Grids
grid = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
        ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']

grid_p1 = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']

grid_p1_origin = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']

grid_p2 = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']

grid_p2_origin = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
           ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']


def is_grid(grid, position):
    if grid[position] == ' ':
        return True
    else:
        return False

def is_winner(grid_1, grid_2):
    end_1 = not 'O' in grid_1 # 'O'가 있으면 == 안 끝남 == False
    end_2 = not 'O' in grid_2
    return [end_1, end_2]


pygame.init() # pygame 초기화

screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

# BGM
bgm = pygame.mixer.Sound('./assets/bgm/03_Game_Theme.mp3')
bgm.play(-1) # -1 시 무한반복


# font
font_title  = pygame.font.SysFont('arial', 80, True, True) # (font name, size, bold=False, italic=False)

title_lobby = font_title.render('BATTLESHIP', True, BLACK) # (Text, antialias, color, background=None)
start_lobby = font_title.render('START', True, BLACK) # (Text, antialias, color, background=None)

title_p1_setting = font_title.render('Player 1 mapping', True, BLACK)
title_p2_setting = font_title.render('Player 2 mapping', True, BLACK)
done_setting = font_title.render('DONE', True, BLACK) # (Text, antialias, color, background=None)

notice_p1_turn = font_title.render('Player 1 Turn', True, BLACK) # (Text, antialias, color, background=None)
notice_p2_turn = font_title.render('Player 2 Turn', True, BLACK) # (Text, antialias, color, background=None)

notice_p1_win = font_title.render('Game Over! P1 WIN!', True, BLACK) # (Text, antialias, color, background=None)
notice_p2_win = font_title.render('Game Over! P2 WIN!', True, BLACK) # (Text, antialias, color, background=None)
notice_draw   = font_title.render('Game Over! DRAW!', True, BLACK) # (Text, antialias, color, background=None)


### pygame 루프 ###
def runGame():
    cell_size  = 30
    num_column = 16
    num_row    = 16
    grid_offset = [400, 150]

    small_cell_size = 10
    
    P1_WIN = 1
    P2_WIN = 2
    DRAW = 3
    game_over = 0
    is_game_over = 0

    turn_set_map = 0
    turn_in_game = 0

    global state, done
    global grid, grid_p1, grid_p2, grid_p1, grid_p2
    while not done:
        clock.tick(10) # clock, tick으로 FPS(Frame per Second) 개념
        screen.fill(WHITE)
        
        ### 유저 입력 감지 및 처리 ###
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # 종료
                done = True
            
            if state == 0: # 로비
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if (event.pos[0] > 540) and (event.pos[0] < 740) and (event.pos[1] > 310) and (event.pos[1] < 410):
                        state = 1

            elif state == 1: # 설정
                if event.type == pygame.MOUSEBUTTONDOWN:
                    column_idx = (event.pos[0] - grid_offset[0]) // cell_size
                    row_idx    = (event.pos[1] - grid_offset[1]) // cell_size
                    position   = column_idx + 16 * row_idx

                    if turn_set_map == 0: # P1 전함 배치
                        if not (column_idx < 0) and not (column_idx > 15) and not (row_idx < 0) and not (row_idx > 15): # grid 외부 오류 방지
                            if is_grid(grid, position):
                                grid_p1[position] = 'O'
                                grid_p1_origin[position] = 'O'

                        if (event.pos[0] > 940) and (event.pos[0] < 1140) and (event.pos[1] > 310) and (event.pos[1] < 410): # set 완료
                            turn_set_map = 1

                    elif turn_set_map == 1: # P2 전함 배치
                        if not (column_idx < 0) and not (column_idx > 15) and not (row_idx < 0) and not (row_idx > 15): # grid 외부 오류 방지
                            if is_grid(grid, position):
                                grid_p2[position] = 'O'
                                grid_p2_origin[position] = 'O'

                        if (event.pos[0] > 940) and (event.pos[0] < 1140) and (event.pos[1] > 310) and (event.pos[1] < 410): # set 완료
                            state = 2

            elif state == 2: # 인게임
                
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    column_idx = (event.pos[0] - grid_offset[0]) // cell_size
                    row_idx    = (event.pos[1] - grid_offset[1]) // cell_size
                    position   = column_idx + 16 * row_idx

                    if not (column_idx < 0) and not (column_idx > 15) and not (row_idx < 0) and not (row_idx > 15): # grid 외부 오류 방지
                        if is_grid(grid, position):
                            grid[position] = 'X'
                            grid_p1[position] = 'X'
                            grid_p2[position] = 'X'

                            if turn_in_game == 0:
                                turn_in_game = 1
                            elif turn_in_game == 1:
                                turn_in_game = 0

                            # 승리 조건
                            if is_game_over == 0:
                                if is_winner(grid_p1, grid_p2) == [True, True]:
                                    game_over = DRAW
                                    is_game_over = 1
                                elif is_winner(grid_p1, grid_p2)[0]:
                                    game_over = P2_WIN
                                    is_game_over = 1
                                elif is_winner(grid_p1, grid_p2)[1]:
                                    game_over = P1_WIN
                                    is_game_over = 1


        ### 화면 표시 ###
        
        if state == 0: # 로비
            screen.blit(title_lobby, (440, 100))
            screen.blit(start_lobby, (540, 310))
            start_button_xywh = (540, 310, 200, 100) # (1280, 720) -> (640, 360), size: 200 x 100
            pygame.draw.rect(screen, BLACK, start_button_xywh, 1)
        elif state == 1: # 설정
            screen.blit(done_setting, (940, 310)) # DONE 글씨
            done_button_xywh = (940, 310, 200, 100) # (1280, 720) -> (640, 360), size: 200 x 100
            pygame.draw.rect(screen, BLACK, done_button_xywh, 1)

            # grid
            if turn_set_map == 0: # P1 mapping
                screen.blit(title_p1_setting, (440, 30))
                for column_idx in range(num_column):
                    for row_idx in range(num_row):
                        position = column_idx + 16 * row_idx
                        if grid_p1[position] == ' ': # 배치 안된 cell: WHITE
                            rect = (grid_offset[0] + cell_size * column_idx, grid_offset[1] + cell_size * row_idx, cell_size, cell_size)
                            pygame.draw.rect(screen, BLACK, rect, 1) # 1은 width인데, 0 이상이면 color의 테두리 rect / 0이면 테두리 없이 color로 fill된 rect
                        if grid_p1[position] == 'O': # 배치된 cell: RED
                            rect_object = pygame.Rect(grid_offset[0] + cell_size * column_idx, grid_offset[1] + cell_size * row_idx, cell_size, cell_size)
                            pygame.draw.rect(screen, GREEN, rect_object)

            if turn_set_map == 1: # P2 mapping
                screen.blit(title_p2_setting, (440, 30))
                for column_idx in range(num_column):
                    for row_idx in range(num_row):
                        position = column_idx + 16 * row_idx
                        if grid_p2[position] == ' ': # 배치 안된 cell: WHITE
                            rect = (grid_offset[0] + cell_size * column_idx, grid_offset[1] + cell_size * row_idx, cell_size, cell_size)
                            pygame.draw.rect(screen, BLACK, rect, 1) # 1은 width인데, 0 이상이면 color의 테두리 rect / 0이면 테두리 없이 color로 fill된 rect
                        if grid_p2[position] == 'O': # 배치된 cell: RED
                            rect_object = pygame.Rect(grid_offset[0] + cell_size * column_idx, grid_offset[1] + cell_size * row_idx, cell_size, cell_size)
                            pygame.draw.rect(screen, BLUE, rect_object)


        elif state == 2: # 인게임
            
            if turn_in_game == 0:
                screen.blit(notice_p1_turn, (440, 30))
            elif turn_in_game == 1:
                screen.blit(notice_p2_turn, (440, 30))

            # grid
            for column_idx in range(num_column):
                for row_idx in range(num_row):
                    position = column_idx + 16 * row_idx
                    if grid[position] == ' ': # 포격 안된 cell: WHITE
                        rect = (grid_offset[0] + cell_size * column_idx, grid_offset[1] + cell_size * row_idx, cell_size, cell_size)
                        pygame.draw.rect(screen, BLACK, rect, 1) # 1은 width인데, 0 이상이면 color의 테두리 rect / 0이면 테두리 없이 color로 fill된 rect
                    if grid[position] == 'X': # 포격 된 cell: RED
                        rect_object = pygame.Rect(grid_offset[0] + cell_size * column_idx, grid_offset[1] + cell_size * row_idx, cell_size, cell_size)
                        pygame.draw.rect(screen, RED, rect_object)
                        if grid_p1_origin[position] == 'O': # 포격 된 cell 중 p1 전함이 있는 cell: 우측 상단에 GREEN
                            rect_object = pygame.Rect(grid_offset[0] + cell_size  * column_idx + 20, grid_offset[1] + cell_size * row_idx, small_cell_size, small_cell_size)
                            pygame.draw.rect(screen, GREEN, rect_object)
                        if grid_p2_origin[position] == 'O': # 포격 된 cell 중 p2 전함이 있는 cell: 우측 하단에 BLUE
                            rect_object = pygame.Rect(grid_offset[0] + cell_size  * column_idx + 20, grid_offset[1] + cell_size * row_idx + 20, small_cell_size, small_cell_size)
                            pygame.draw.rect(screen, BLUE, rect_object)


            if game_over == DRAW:
                screen.blit(notice_draw, (340, 310))
            elif game_over == P1_WIN:
                screen.blit(notice_p1_win, (340, 310))
            elif game_over == P2_WIN:
                screen.blit(notice_p2_win, (340, 310))



        pygame.display.update()

runGame()
pygame.quit()
