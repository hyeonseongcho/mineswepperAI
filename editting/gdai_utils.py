import torch
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

def make_dataIn(data):
    dataIn = []
    for (inCir,outCir) in data:
        dataIn.append(np.concatenate((inCir,[outCir])))
    return dataIn

def softmax(a) :
    temperature = 0.01
    maxA = np.max(a)
    exp_a = np.exp(temperature*(a-maxA))
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def set_data_sel(sel, cir, datas, dataIns):
    cir.data = datas[sel]
    data = np.array(dataIns[sel]).flatten()
    return data

def decrease_epsilon(epsilon, sim, period):
    if (((sim+1) % int(period)) == 0) & (epsilon > 0.1):
        epsilon = epsilon / 2
        print('epsilon =' + str(epsilon))
    return epsilon

def rec_Bufs(turn, reward, sel, Bufs, stM, mkM, acM, rdM):
    ret = 0
    for i in range(turn):
        index = turn -1 -i
        ret = 0.95 * ret + rdM[index]
        if (reward == 1):
            Bufs[sel].append((stM[index], mkM[index], acM[index], ret))
        else:
            Bufs[8].append((stM[index], mkM[index], acM[index], ret))
    return Bufs

def make_replayBuf(Bufs):
    replayBuf = deque()
    for i in range(9):
        replayBuf.extend(Bufs[8-i])
    return replayBuf

def plot_result(is_train, sim, sim_plot, sc_plot, num_sim, sc_sim, sc_acc, moving):
    '''
    ## INPUT ##
    is_train: True(Train) or False(Test)
    sim: int (이번 sim)
    sim_plot: [] : sim 계속 append (plt x axis)
    sc_plot: [[], [], [], [], [], [], [], []] (_tr, _ts) (plt y axis)
    num_sim: [0, 0, 0, 0, 0, 0, 0, 0] / 이번 sim에서 시도 수 (_tr, _ts)
    sc_sim: [0, 0, 0, 0, 0, 0, 0, 0] / 이번 sim에서 성공 수 (_tr, _ts)
    sc_acc: [0, 0, 0, 0, 0, 0, 0, 0] / 여태까지 누적 성공 수 (_tr, _ts)
    moving: [[], [], [], [], [], [], [], []]
    
    ### OUTPUT ### : 저장해야하는 것들
    sim_plot, sc_plot, sc_acc, moving
    '''

    if is_train == True:
        print(f'sim: {sim}\n========== TRAIN ==========')
        sim_plot.append(sim)
    elif is_train == False:
        print(f'========== TEST ==========')

    for i in range(8):
        sc_acc[i] += sc_sim[i] # sc_sim을 sc_acc에 누적
        sc_plot[i].append(sc_acc[i]) # sc_acc을 sc_plot에 추가

    print('<This sim>')
    print(f'INV1: {sc_sim[0]}/{num_sim[0]} INV2: {sc_sim[1]}/{num_sim[1]} NAND: {sc_sim[2]}/{num_sim[2]} NOR: {sc_sim[3]}/{num_sim[3]}')
    print(f'BUF1: {sc_sim[4]}/{num_sim[4]} BUF2: {sc_sim[5]}/{num_sim[5]} AND: {sc_sim[6]}/{num_sim[6]} OR: {sc_sim[7]}/{num_sim[7]}')

    print('<Accumulated Success>')
    print(f'INV1: {sc_acc[0]} INV2: {sc_acc[1]} NAND: {sc_acc[2]} NOR: {sc_acc[3]}')
    print(f'BUF1: {sc_acc[4]} BUF2: {sc_acc[5]} AND: {sc_acc[6]} OR: {sc_acc[7]}')
    for i in range(8):
        plt.plot(sim_plot, sc_plot[i])
    plt.title('Number of Accumulated Successes in Training')
    plt.legend(['INV1', 'INV2', 'NAND', 'NOR', 'BUF1', 'BUF2', 'AND', 'OR'],loc='upper left')
    plt.show()

    if (sim > 100) & (is_train == False) :
        print('<moving success rate>')
        for i in range(8):
            moving[i].append((sc_plot[i][-1] - sc_plot[i][-100])/100)
        for i in range(8):
            plt.plot(sim_plot[101:], moving[i])
        plt.title('Moving success rate in last 100 sim')
        plt.legend(['INV1', 'INV2', 'NAND', 'NOR', 'BUF1', 'BUF2', 'AND', 'OR'],loc='upper left')
        plt.show()

    return sim_plot, sc_plot, sc_acc, moving

def record_ep(reward, sel, num_sim, sc_sim):
    num_sim[sel] += 1
    if reward == 1:
        sc_sim[sel] += 1
    return num_sim, sc_sim