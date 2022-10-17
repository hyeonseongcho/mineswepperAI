import torch
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt


class Circuit:
    def __init__(self, numIn, numOut, numN, numP, data, gpu):
        self.gpu = gpu

        self.numIn = numIn
        self.numOut = numOut
        self.numN = numN
        self.numP = numP
        
        self.data = data
        
        self.frontSize = 2 + numIn + numOut
        self.spaSize = 2 + numIn + numOut + (numN + numP)*3
        self.chSize = 6 # vdd,gnd,in,out,n,p
        
        self.errType = 0
        
        self.features = np.zeros((1, self.spaSize, self.chSize))
        
        self.features[0,0,0] = 1 # vdd
        self.features[0,1,1] = 1 # gnd
        
        index = 2
        for i in range(numIn):
            self.features[0,index+i,2] = 1 # inputs
        
        index += numIn
        for i in range(numOut):
            self.features[0,index+i,3] = 1 # outputs
        
        index += numOut
        for i in range(numN):
            self.features[0,index+3*i:index+3*(i+1),4] = 1 # NMOS
            
        index += 3*numN
        for i in range(numP):
            self.features[0,index+3*i:index+3*(i+1),5] = 1 # NMOS
        
        
        # Circuit graph
        self.cir = np.zeros((self.spaSize, self.spaSize, 1+2*self.chSize))
        
        self.cir[:,:,1:self.chSize+1] = self.features
        self.cir[:,:,self.chSize+1:] = np.swapaxes(self.features,0,1)
        
    def inputGen(self):
        
        inputNet = []
        
        for i in range(self.spaSize-1):
            if (i < self.frontSize):
                inputNet = np.concatenate((inputNet,self.cir[self.frontSize:self.spaSize,i,0]),axis=None)
            else:
                inputNet = np.concatenate((inputNet,self.cir[i+1:self.spaSize,i,0]), axis=None)
                
        return 2*np.array(inputNet) - 1
            
    def maskGen(self, data):
        
        mask = []
        
        # (1) The existing connections should be skipped
        maskCir = self.cir[:,:,0].copy()
        
        maskInOut = np.array(np.nonzero(np.sum(np.abs(data),axis=0) == 0)).flatten()
        
        if (len(maskInOut) > 0):
            for i in maskInOut:
                maskCir[:,2+i] = 1
                maskCir[2+i,:] = 1
                
        
        # (2) From VDD to ouputs must not be connected to one node
        for i in range(3*(self.numN+self.numP)):
            index = self.frontSize + i
            if (np.sum(self.cir[0:self.frontSize,index,0]) > 0):
                for j in range(self.frontSize):
                    maskCir = np.array(np.logical_or(maskCir,self._fullMatrix(j,index)),dtype=np.int32)
        
        for i in range(self.numN+self.numP):
            index = self.frontSize+3*i
            
            # (3) The same signal must not be connected to S and D of a MOS.
            maskCir = np.array(np.logical_or(maskCir,self._fullMatrix(index,index+2)),dtype=np.int32)
            
            # (4) From VDD to inputs must not be connexted to S and D of a MOS
            if (np.sum(self.cir[0:self.frontSize-self.numOut,index,0] + self.cir[0:self.frontSize-self.numOut,index+2,0]) > 0):
                for j in range(self.frontSize-self.numOut):
                    maskCir = np.array(np.logical_or(maskCir,self._fullMatrix(j,index)),dtype=np.int32)
                    maskCir = np.array(np.logical_or(maskCir,self._fullMatrix(j,index+2)),dtype=np.int32)
                
            # (5) The same signal must not be connexted to S and D of a MOS
            for j in range(self.numOut+3*i):
                indey = self.frontSize-self.numOut+j
                if (self.cir[indey,index,0] + self.cir[indey,index+2,0] > 0):
                    maskCir = np.array(np.logical_or(maskCir,self._fullMatrix(indey,index)),dtype=np.int32)
                    maskCir = np.array(np.logical_or(maskCir,self._fullMatrix(indey,index+2)),dtype=np.int32)
                    
        maskCir = np.array(np.logical_or(maskCir, np.transpose(maskCir)),dtype=np.int32)
        
        for i in range(self.spaSize-1):
            if (i < self.frontSize):
                mask = np.concatenate((mask,maskCir[i,self.frontSize:self.spaSize]),axis=None)
            else:
                mask = np.concatenate((mask,maskCir[i,i+1:self.spaSize]), axis=None)
                
        self.mask = mask
                    
        return mask, maskCir
            
    def _fullMatrix(self, indy, indx):
        
        tempCir = self.cir[:,:,0].copy()
        
        tempCir[indy,indx] = 1
        
        pos = np.array(np.nonzero(tempCir[indy,:])).flatten()
        for i in pos:
            for j in pos:
                if (i != j):
                    tempCir[i,j] = 1
                    
        pos = np.array(np.nonzero(tempCir[:,indx])).flatten()
        
        for i in pos:
            for j in pos:
                if (i != j):
                    tempCir[i,j] = 1
        
        return tempCir
    
    def reset(self):
        self.cir = np.zeros((self.spaSize, self.spaSize, 1+2*self.chSize))
        
        self.cir[:,:,1:self.chSize+1] = self.features
        self.cir[:,:,self.chSize+1:] = np.swapaxes(self.features,0,1)
        
    def _num2pos(self, num):
        
        col = num
        
        for i in range(self.spaSize-1):
            if (i < self.frontSize):
                div = self.spaSize - self.frontSize
                offset = self.frontSize
            else:
                div = self.spaSize-1-i
                offset = i + 1
                
            if col < div:
                row = i
                break
            else:
                col = col - div
                
        col = col + offset
                
        return row, col   
            
    def _connect(self):
        
        col = self.col
        row = self.row
        
        self.cir[row,col,0] = 1
        
        pos = np.array(np.nonzero(self.cir[row,:,0])).flatten()
        
        for i in pos:
            for j in pos:
                if (i != j):
                    self.cir[i,j,0] = 1
                    
        pos = np.array(np.nonzero(self.cir[:,col,0])).flatten()
        
        for i in pos:
            for j in pos:
                if (i != j):
                    self.cir[i,j,0] = 1
            
        temp = np.logical_or(self.cir[:,:,0],np.transpose(self.cir[:,:,0]))
        
        self.cir[:,:,0] = np.array(temp, dtype=np.int32)
        
    def step(self, num):
        
        self.row, self.col = self._num2pos(num)
        
        col = self.col
        row = self.row
        
        self.errType = 0
        
        if (len(np.array(np.nonzero(self.mask==0)).flatten()) == 0):
            self.errType = 4
            reward = torch.Tensor([-1]).to(self.gpu)
            done = 1
            #print('# Error Type = 4')
        elif (self.cir[row,col,0] == 0):
            self._connect()
            reward, done = self._getReward()
        else:
            reward = torch.Tensor([-1]).to(self.gpu)
            self.errType = 1
            print('# Error Type = 1')
            #self.reset()
            done = 0
        #    print("####already connected")
            
        return reward, done
            
    def _getReward(self):
        
        if self._nodeShort():
            reward = torch.Tensor([-1]).to(self.gpu)
            done = 1 # 0
            self.errType = 2
            print('# Error Type = 2')
        else:
            result = self._simCheck()
            if result == 1:
                reward = torch.Tensor([1]).to(self.gpu)
                done = 1
        #        print("####Simulation Success")
            else:
                reward = torch.Tensor([-1]).to(self.gpu)
                done = 0
                self.errType = 3
        #        print("####Wrong Results")
            
        return reward, done
        
    def _simCheck(self):
        
        simResult = 1
        
        for (inLogic, outLogic) in self.data:
            out = self._simLogic(inLogic)
            if (out != outLogic):
                simResult = 0
                break
            
            
        return simResult
            
    
    def _partialConCheck(self):
        
        partialCon = 0
        index = 2 + self.numIn + self.numOut
        
        # no connect for input and output
        for i in range(self.numIn+self.numOut):
            if len(np.array(np.nonzero(self.cir[:,2+i,0])).flatten()) == 0:
                partialCon = 1
   
        return partialCon
        
    def _nodeShort(self):
        
        # vdd, gnd, input, output
        ndShort = 0
        
        index = 2 + self.numIn + self.numOut
        
        ## vdd, gnd short
        for i in range(index):
            numOne = len(np.array(np.nonzero(self.cir[i,0:index,0] == 1)).flatten())
            if (numOne > 0):
                ndShort = 1
                break

        # Source Drain Short Check
        
        sdShort = 0
        index = 2 + self.numIn + self.numOut
        
        for i in range(self.numN):
            if (self.cir[index,index+2,0] == 1) :
                sdShort = 1
            elif (self.cir[0,index,0] == 1):
                for j in range(2+self.numIn):
                    if (self.cir[j,index+2,0] == 1):
                        sdShort = 1
            elif (self.cir[1,index,0] == 1):
                for j in range(2+self.numIn):
                    if (self.cir[j,index+2,0] == 1):
                        sdShort = 1
            else:
                for j in range(self.numIn):
                    if (self.cir[2+j,index,0] == 1):
                        for k in range(2+self.numIn):
                            if (self.cir[k,index+2,0] == 1):
                                sdShort = 1
                
            index += 3
            
        for i in range(self.numP):
            
            if (self.cir[index,index+2,0] == 1) :
                sdShort = 1
            elif (self.cir[0,index,0] == 1):
                for j in range(2+self.numIn):
                    if (self.cir[j,index+2,0] == 1):
                        sdShort = 1
            elif (self.cir[1,index,0] == 1):
                for j in range(2+self.numIn):
                    if (self.cir[j,index+2,0] == 1):
                        sdShort = 1
            else:
                for j in range(self.numIn):
                    if (self.cir[2+j,index,0] == 1):
                        for k in range(2+self.numIn):
                            if (self.cir[k,index+2,0] == 1):
                                sdShort = 1
                
        return ndShort | sdShort
    
    def _updateLogic(self,pos,offset):
        con = np.array(np.nonzero(self.cir[:,pos,0])).flatten()
        for i in con:
            if (i >= offset):
                self.simCir[i-offset] = self.simCir[pos-offset]
    
    def _updateSD(self):
        update = 0
        offset = 2+self.numIn+self.numOut
        for n in range(self.numN):
            if (self.simCir[3*n+1] == 1):
                if (self.simCir[3*n] == 0) & (self.simCir[3*n+2] != 0):
                    self.simCir[3*n] = self.simCir[3*n+2]
                    self._updateLogic(3*n+offset,offset)
                    update = 1
                elif (self.simCir[3*n+2] == 0) & (self.simCir[3*n] != 0):
                    self.simCir[3*n+2] = self.simCir[3*n]
                    self._updateLogic(3*n+2+offset,offset)
                    update = 1
                    
        offsetP = 3*self.numN
        for p in range(self.numP):
            if (self.simCir[offsetP+3*p+1] == -1):
                if (self.simCir[offsetP+3*p] == 0) & (self.simCir[offsetP+3*p+2] != 0):
                    self.simCir[offsetP+3*p] = self.simCir[offsetP+3*p+2]
                    self._updateLogic(3*p+offset+offsetP,offset)
                    update = 1
                elif (self.simCir[offsetP+3*p+2] == 0) & (self.simCir[offsetP+3*p] != 0):
                    self.simCir[offsetP+3*p+2] = self.simCir[offsetP+3*p]
                    self._updateLogic(3*p+2+offset+offsetP,offset)
                    update = 1
        
        return update
    
    def _simLogic(self, inData):
        
        numNoTr = 2 + self.numIn + self.numOut
        numTr = (self.numN+self.numP)*3
        self.simCir = np.zeros((numTr))
        
        #VDD = 1
        pos = np.array(np.nonzero(self.cir[0,:,0])).flatten()
        for i in pos:
            self.simCir[i-numNoTr] = 1
            
            self._updateLogic(i,numNoTr)
            
            update = 1
            
            while update:
                update = self._updateSD()
        
        #GND = -1
        pos = np.array(np.nonzero(self.cir[1,:,0])).flatten()
        for i in pos:
            self.simCir[i-numNoTr] = -1
            
            self._updateLogic(i,numNoTr)
            
            update = 1
            
            while update:
                update = self._updateSD()
        
      
        #input
        for n in range(self.numIn):
            pos = np.array(np.nonzero(self.cir[2+n,:,0])).flatten()
            for i in pos:
                self.simCir[i-numNoTr] = inData[n]
                
                self._updateLogic(i,numNoTr)

                update = 1

                while update:
                    update = self._updateSD()
        
        con = np.array(np.nonzero(self.cir[:,numNoTr-1,0])).flatten()
        if (len(con) > 0) :
            out = self.simCir[con[0]-numNoTr]
        else:
            out = 0
        
        return out
                            
class Designer:
    def __init__(self, l_rate, sizeF, sizeX, sizeCh, gpu):
        
        self.gpu = gpu
        
        self.mainDQ = GDNet(sizeF, sizeX, sizeCh).to(self.gpu)
        self.targetDQ = GDNet(sizeF, sizeX, sizeCh).to(self.gpu)
        self.targetDQ.load_state_dict(self.mainDQ.state_dict())
        self.targetDQ.eval()
        
        self.criterion = torch.nn.MSELoss()
        
        self.opt_mainDQ = torch.optim.Adam(self.mainDQ.parameters(), l_rate)
        self.opt_targetDQ = torch.optim.Adam(self.targetDQ.parameters(), l_rate)
        
        self.sizeQ = self.mainDQ.sizeOut
        self.sizeBatch = 64
        self.gamma = 0.95
        
    def update(self): # targetDQ의 weight를 mainDQ의 weight로 바꿔줌
        self.targetDQ.load_state_dict(self.mainDQ.state_dict())
        
    def learn(self, mem):
        
        self.opt_mainDQ.zero_grad()
        
        inputs = []
        targetQ = []
        
        for i, (state, mask, action, ret) in enumerate(mem):
            inputs.append(state)
            
            tarQ = self.mainDQ(state)
            tarQ[action] = ret # 현재 tarQ는 mainDQ에서의 Q_value를 담고있음
            # action을 취했을 때, targetDP에서의 기대 reward를 부여해서 나중에 fitting시 그쪽으로 되게함
            targetQ.append(tarQ)
            
        inputs  = torch.stack(inputs)
        targetQ = torch.stack(targetQ)
        
        mainQ = self.mainDQ(inputs)
        loss  = self.criterion(mainQ, targetQ)
        loss.backward()
        self.opt_mainDQ.step()
            
    def design(self, eps, state, mask, learn=True):
        
        if learn:
            
            Q = self.mainDQ(state)
            #print(Q.max())
            if eps > torch.rand([1]):
                pos = torch.nonzero(mask == 0).flatten()
                if (len(pos) > 0):
                    tmp = torch.randint(0, len(pos), [1])
                    num = pos[tmp]
                else:
                    num = torch.randint(0, self.sizeQ, [1]).to(self.gpu)
            else:
                num = torch.argmax(Q - 10000*mask).unsqueeze(dim=0).to(self.gpu)


        else:
            Q = self.mainDQ(state)
            num = torch.argmax(Q - 10000*mask)
        
        return num, Q
        
class GDNet(torch.nn.Module):
    def __init__(self, sizeFront, sizeX, sizeCh):
        super().__init__()
        
        self.sizeOut = 0
        
        for i in range(sizeX-1):
            if (i < sizeFront):
                self.sizeOut += (sizeX - sizeFront)
            else:
                self.sizeOut += (sizeX - 1 - i)
        
        self.sizeIn = self.sizeOut + 12
        
        self.fc1 = torch.nn.Linear(self.sizeIn, self.sizeIn*2)
        self.fc2 = torch.nn.Linear(self.sizeIn*2, self.sizeIn*2)
        self.fc3 = torch.nn.Linear(self.sizeIn*2, self.sizeIn*2)
        self.fc4 = torch.nn.Linear(self.sizeIn*2, self.sizeOut)
        
        self.norm1 = torch.nn.LayerNorm(self.sizeIn*2)
        self.norm2 = torch.nn.LayerNorm(self.sizeIn*2)
        self.norm3 = torch.nn.LayerNorm(self.sizeIn*2)
        

        self.relu = torch.nn.ReLU()
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = self.fc4(x)

        return x
    
