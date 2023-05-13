def mjrFormatter(x, pos):
    return "$10^{{{0}}}$".format(x)

def main():
    import matplotlib
    import matplotlib.pyplot as plt
    import math

    fig, ax = plt.subplots(figsize=(15, 10))
    radius = 9.5
    notation_size = 21
    notation_small_size = 21
    
    '''huge models'''
    # IMDN RFDN ECBSR-M16C64
    x = [265.20, 132.67]
    log_x = [math.log(item, 10) for item in x]
    y = [32.17, 31.71]
    area = (40) * radius**2
    ax.scatter(log_x, y, s=area, alpha=0.8, marker='.', c='#4D96FF', edgecolors='white', linewidths=2.0)
    plt.annotate('IMDN', (log_x[0]+ 0.035, y[0] - 0.05), fontsize=notation_size)
    plt.annotate('ECBSR-M16C64', (log_x[1]+ 0.04, y[1] - 0.04), fontsize=notation_small_size)
    

    
    '''large models'''
    x = [200.89, 155.69]
    log_x = [math.log(item, 10) for item in x]
    y = [32.12, 32.18]
    area = (30) * radius**2
    plt.annotate('RFDN', (log_x[0]+ 0.02, y[0] - 0.15), fontsize=notation_size)
    # plt.annotate('PCAVAnet-L', (log_x[1] + 0.04, y[1] - 0.03), fontsize=notation_size)
    plt.annotate('PCEVAnet-L', (log_x[1] + 0.02, y[1] + 0.08), fontsize=notation_size)
    ax.scatter(log_x, y, s=area, alpha=0.8, marker='.', c='#95CD41', edgecolors='white', linewidths=2.0)

    '''medium models'''
    # ECBSR-M10C32  FASR-M
    x = [33.3, 33.08]
    log_x = [math.log(item, 10) for item in x]
    y = [31.25, 31.39]
    area = (20) * radius**2
    ax.scatter(log_x, y, s=area, alpha=0.8, marker='.', c='#FFD93D', edgecolors='white', linewidths=2.0)
    # # plt.annotate('BSRN(Ours)', (357 - 70, 32.35 + 0.10), fontsize=notation_size)
    plt.annotate('ECBSR-M10C32', (log_x[0] + 0.03, y[0] - 0.03), fontsize=notation_small_size)
    plt.annotate('PCEVAnet-M', (log_x[1] + 0.03, y[1]  - 0.01), fontsize=notation_small_size)
    # plt.annotate('PCAVAnet-M', (log_x[2], y[2] + 0.04), fontsize=notation_small_size)
    
    '''small models'''
    # ECBSR-M4C8  PCAVAnet-S
    x = [5.39, 4, 25.83]
    log_x = [math.log(item, 10) for item in x]
    y = [29.68, 30.48, 29.88]
    area = (15) * radius**2
    ax.scatter(log_x, y, s=area, alpha=0.8, marker='.', c='#FFB6C1', edgecolors='white', linewidths=2.0)
    # # plt.annotate('BSRN(Ours)', (357 - 70, 32.35 + 0.10), fontsize=notation_size)
    plt.annotate('ECBSR-M4C8', (log_x[0] + 0.0, y[0] + 0.04), fontsize=notation_small_size)
    plt.annotate('PCEVAnet-S', (log_x[1], y[1] + 0.04), fontsize=notation_small_size)
    plt.annotate('FSRCNN', (log_x[2], y[2] + 0.04), fontsize=notation_small_size)
    
    '''Ours marker'''
    x = [4, 33.08, 155.69]
    log_x = [math.log(item, 10) for item in x]
    y = [30.48, 31.39, 32.18]
    ax.scatter(log_x, y, alpha=1.0, marker='*', c='r', s=300)
    ax.plot(log_x, y, color='r', linewidth=1, linestyle='-.',alpha=0.6)

    # plt.xlim(0, 270)
    
    
    plt.ylim(29.5, 32.5)
    plt.xlabel('Latency(ms)', fontsize=25)
    plt.ylabel('PSNR(dB)', fontsize=25)
    # plt.title('PSNR vs. Latency', fontsize=35)

    h = [
        plt.plot([], [], color=c, marker='.', ms=i, alpha=a, ls='')[0] for i, c, a in zip(
            [30, 38, 40, 50], ['#FFB6C1', '#FFD93D', '#95CD41', '#4D96FF'], [0.8, 1.0, 0.6, 0.8])
    ]
    ax.legend(
        labelspacing=0.1,
        handles=h,
        handletextpad=1.0,
        markerscale=1.0,
        fontsize=16,
        title='FLOPs (G)',
        title_fontsize=22,
        labels=['< 20', '20 - 100', '100 - 250', '250 - 500'],
        scatteryoffsets=[0.0],
        loc='lower right',
        ncol=4,
        shadow=False,
        handleheight=4)

    for size in ax.get_xticklabels():  # Set fontsize for x-axis
        size.set_fontsize('25')
    for size in ax.get_yticklabels():  # Set fontsize for y-axis
        size.set_fontsize('25')

    # ax.grid(linestyle='-.', linewidth=0.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(mjrFormatter))

    X = [24, 30, 60, 120]
    X_FPS = [math.log(1000/item, 10) for item in X]
    
    plt.axvline(X_FPS[0], color='gray', linestyle="--")
    plt.axvline(X_FPS[1], color='gray', linestyle="--")
    plt.axvline(X_FPS[2], color='gray', linestyle="--")
    plt.axvline(X_FPS[3], color='gray', linestyle="--")
    ybar = 31.9
    plt.text(0.65, ybar, '120FPS',  fontweight='bold', fontsize=18)
    plt.text(1.0, ybar, '60FPS', fontweight='bold', fontsize=18)
    plt.text(1.3, ybar, '30FPS', fontweight='bold', fontsize=18)
    plt.text(1.50, ybar, '24FPS', fontweight='bold', fontsize=18)
    plt.text(1.80, ybar, '<24FPS', fontweight='bold', fontsize=18)
    plt.axvspan(X_FPS[0], 2.5, color=(155/255, 168/255, 147/255), alpha=0.1)
    plt.axvspan(X_FPS[1], X_FPS[0], color=(233/255, 210/255, 244/255), alpha=0.1)
    plt.axvspan(X_FPS[2], X_FPS[1], color="crimson", alpha=0.1)
    plt.axvspan(X_FPS[3], X_FPS[2], color="lightcyan", alpha=0.1)
    plt.axvspan(0.55,  X_FPS[3], color="gold", alpha=0.1)

    plt.show()
    fig.savefig('model_vis.png')
    fig.savefig('model_vis.pdf')

if __name__ == '__main__':
    main()