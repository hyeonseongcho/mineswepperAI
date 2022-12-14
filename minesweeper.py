import numpy as np

class Minesweeper():
    def __init__(self, num_mine, sizeX, sizeY):
        
        self.num_mine = num_mine # The number of mine
        self.sizeX = sizeX # X-axis size
        self.sizeY = sizeY # Y-axis size
        self.num_cell = self.sizeX * self.sizeY # The number of cell
        self.num_actions = 2 * self.num_cell # first ~self.num_cell: OPEN / second: POINT

        self.win  = 0
        self.dead = 0

        self.make_game()

    def new_game(self, num_mine, sizeX, sizeY):
        self.__init__(num_mine, sizeX, sizeY)
    
    def make_game(self):
        # Make mines, initial status
        # mines doesn't change during game
        # An action of agent make a change of status

        self.pos_mine = np.random.choice(self.num_cell, size=self.num_mine, replace=False) # 1-D vector
        self.mines  = np.zeros((self.sizeX, self.sizeY)) # 0: not mine, 1: mine # 2-D vector
        self.status = np.zeros((self.sizeX, self.sizeY)) # 0: unknown(closed & not pointed), 1: opened, -1: pointed(as mine)
        for i in self.pos_mine: # Fill mines
            posX, posY = self.num2pos(i)
            self.mines[posY][posX] = 1

        # Fill num_mine_around
        # Apply Conv2d  to mines and 0 to mines filter
        self.num_mine_around = np.zeros((self.sizeX, self.sizeY))
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
        padded = np.pad(self.mines, ((1,1),(1,1)), 'constant', constant_values = 0)
        for i in range(self.sizeX):
            for j in range(self.sizeY):
                self.num_mine_around[i,j] = (padded[i:i+3, j:j+3] * kernel).sum()
    
    # pos_mine: 1-D vector (sizeX*sizeY)
    # mines: 2-D vector (sizeX, sizeY)
    def num2pos(self, num): # Convert 1-D idx to 2-D coordinate
        posY = num // self.sizeX
        posX = num % self.sizeX
        return posX, posY

    def pos2num(self, posX, posY): # Convert 1-D idx to 2-D coordinate
        num = posY * self.sizeX + posX
        return num

    def get_state(self):
        # Agent gets information through state. State is one 2-D matrix
        # Each cell has one status among 'unknown', 'pointed', 'opened': based on self.status
        # If cell 'opened', it will show 'num_mine_around'
        state = np.where(self.status==0,-5,self.status)
        for i in range(self.sizeX):
            for j in range(self.sizeY):
                if state[i,j] == 1: # opened
                    state[i,j] = self.num_mine_around[i,j]
        return state

    def action_mask(self): # Mask for invalid action - 0:Valid Action, 1:Invalid Action
        # Invalid action
        # 1) Open 'opened' 2) Point 'opened' 3) Point 'pointed'

        # Action Number
        # Let's assume 10*10 grid / self.num_cell = 100 / self.num_actions = 200
        # IDX 0~99: OPEN / IDX 100~199: POINT

        flatten_status = self.status.copy().flatten()
        # 1) Open 'opened': OPEN????????? point??? ?????? ????????????????????? ???
        cond_1 = np.where(flatten_status==-1, 0, flatten_status)
        # 2,3) Point 'opened'??? Point 'pointed': Point????????? -1???(point?????????) 1???(invalid???) ???????????? ???
        cond_2 = np.where(flatten_status==-1,1,flatten_status)

        self.mask = np.concatenate((cond_1,cond_2),axis=0)

        return self.mask

    def action(self, num):
        # All possible action

        # 1) OPEN: Open 'unknown' or 'pointed'. Can't open 'opened'
        # 2) POINT: Point 'unknown' as mine. 'Unknown' becomes 'pointed'
        #
        # Reward: choose one way
        # 1) DEAD: -1, ALIVE: 1, WIN: 1000(big reward)
        # 2) DEAD: -1, WIN: 1 
        #
        # action: make action -> change status -> give reward to agent
        
        if num > self.num_cell - 1: # POINT aciton
            num_sub = num - self.num_cell
            posX, posY = self.num2pos(num_sub)
            if self.status[posY][posX] == 0: # allow only point 'unknown'
                self.status[posY][posX] = -1
            elif self.status[posY][posX] == 1:
                print('invalid action | point "opened" cell')
            else:
                print('invalid action | point "pointed" cell')

        else: # OPEN action
            posX, posY = self.num2pos(num)
            if (self.status[posY][posX] == 0) or (self.status[posY][posX] == -1):
                self.open(num)
            else:
                print('invalid action | open "opened" cell')

        self.win_or_dead_or_alive()
        
        if self.dead == 1:
            #print('dead')
            done = 1
            reward = -1
        elif self.win == 1:
            #print('win')
            reward = 100 # big reward
            done = 1
        else: # alive, continue
            # point??? reward??? ?????????????????? point??? ??? ?????? ???????????? ????????? ?????????
            # point??? reward ????????? ???
            if num > self.num_cell - 1:
                reward = 0
                done = 0
            else:
                reward = 1
                done = 0
            # ?????? ????????? ??????????????? ?????? ????????? ??? ??????
            # ?????? ??????, open????????? ?????? point?????? open?????? ????????? ???????????????
            # ????????? ????????? win ??? reward??? (big reward)??? ????????? (big reward - alpha * turn) ?????? ???????????? ?????? ?????? ???

        return reward, done

    def open(self, num):
        # "0 ?????? ??????"
        # ?????? ??????: open??? cell??? mine??? ?????????, num_mine_around??? 0??? ??????
        # ??????: ?????? 8 cell??? ?????? ????????????. num_mine_around??? 0????????? ????????? ?????? ????????? ?????? ?????? ?????????.
        # ????????? ?????????: 8 cell ??? ????????? ?????? ???????????? ?????? ?????? ?????? ????????? ??????
        
        # ?????? ??????
        posX, posY = self.num2pos(num)
        #print('self', posX, posY)
        self.status[posY][posX] = 1

        if self.num_mine_around[posY][posX] == 0: # ????????? 0?????? ?????? ??????
            look = [-1, 0, 1]
            for i in look:
                for j in look:
                    if not [i,j] == [0,0]: # ????????? ??????
                         # index negative or overflow ??????
                        if (posX+i >= 0) and (posX+i < self.sizeX) and (posY+j >= 0) and (posY+j < self.sizeY):
                            before = self.status[posY+j][posX+i]
                            self.status[posY+j][posX+i] = 1 # ??????
                            #print('side', posX+i, posY+j)
                            # ?????? ?????? 0??????
                            if self.num_mine_around[posY+j][posX+i] == 0 and before == 0:
                                new_num = self.pos2num(posX+i, posY+j)
                                self.open(new_num) # ????????????`


    def win_or_dead_or_alive(self):
        # DEAD?: when agent opens mine
        # HOW JUDGE 'DEAD'?: if 1 in mine*status / when agent opens(status 1) mine(mine 1)
        #
        # WIN?: when agent points all mine
        # HOW JUDGE 'WIN'?: when -1 of num_mine in mine*status
        # 'pointed'(status -1) 'mine'(mines 1) = -1
        #
        # BUT, one more condition: -1 of num_mine in status
        # Without this condition, Agent can win by just pointing all cells
        
        times = self.mines * self.status
        if 1 in times:
            self.dead = 1
        else:
            if np.count_nonzero(self.status==-1) == self.num_mine: # -1 of num_mine in status
                if np.count_nonzero(times==-1) == self.num_mine: # all mine found
                    self.win = 1