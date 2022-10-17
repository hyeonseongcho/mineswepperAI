import torch
import numpy as np
import matplotlib.pyplot as plt

def decrease_epsilon(epsilon, sim, period):
    if (((sim+1) % int(period)) == 0) & (epsilon > 0.1):
        epsilon = epsilon / 2
        print('epsilon =' + str(epsilon))
    return epsilon

def plot_result(is_train, plot, sim, rd, sim_plot, rd_mean_plot, rd_mean_plot_smooth):
    '''
    ## INPUT ##
    is_train: True(Train) or False(Test)
    sim: int (이번 sim)
    sim_plot: [] : sim 계속 append (plt x axis)
    rd_mean_plot: [] (_tr, _ts) (plt y axis, mean reward)
    rd_mean_plot_smooth: [] (_tr, _ts) (plt y axis, mean reward): smooth ver.
    
    ### OUTPUT ### : 저장해야하는 것들
    sim_plot, rd_plot
    '''

    if is_train == True:
        sim_plot.append(sim)

    if is_train == True:
        rd_mean = torch.mean(torch.Tensor(rd))
    else:
        rd_mean = rd

    rd_mean_plot.append(rd_mean) # rd_mean을 rd_plot에 추가
    
    if sim > 10:
        smooth = torch.mean(torch.Tensor((rd_mean_plot[-10:])))
    else:
        smooth = rd_mean
    rd_mean_plot_smooth.append(smooth)

    if plot:
        if is_train == True:
            print(f'sim: {sim}\n========== TRAIN ==========')
            plt.figure(figsize=(16, 4))
            plt.subplot(1,2,1)
            plt.plot(sim_plot, rd_mean_plot)
            plt.title('Mean of accumulated rewards in Training')
            
            plt.subplot(1,2,2)
            plt.plot(sim_plot, rd_mean_plot_smooth)
            plt.title('Smooth ver.')
        elif is_train == False:
            print(f'========== TEST ==========')
            plt.figure(figsize=(16, 4))
            plt.subplot(1,2,1)
            plt.plot(sim_plot, rd_mean_plot)
            plt.title('Accumulated reward in Test')

            plt.subplot(1,2,2)
            plt.plot(sim_plot, rd_mean_plot_smooth)
            plt.title('Smooth ver.')

        #print('Accumulated_rewards in this sim')
        #print(rd)
        #plt.plot(sim_plot, rd_mean_plot)
        
        #if is_train == True:
        #    plt.title('Mean of accumulated rewards in Training')
        #else:
        #    plt.title('Accumulated rewards in Test')
        
        plt.show()

    return sim_plot, rd_mean_plot, rd_mean_plot_smooth