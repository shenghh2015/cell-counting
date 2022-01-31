from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# plot loss and mce
def plot_history(file_name, history, deeply, t_epochs):
    if not deeply:
        train_loss_name = 'loss'
        train_metric_name = 'mce'
        val_loss_name = 'val_loss'
        val_metric_name = 'val_mce'
    else:
        train_loss_name = 'original_loss'
        train_metric_name = 'original_mce'
        val_loss_name = 'val_original_loss'
        val_metric_name = 'val_original_mce'
    
    rows, cols, size = 1,2,5
    font_size = 15
    fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
    ax[0].plot(history.history[train_loss_name][t_epochs:])
    ax[0].plot(history.history[val_loss_name][t_epochs:])
    ax[0].set_ylabel('mse', fontsize = font_size)
    ax[0].set_xlabel('Epochs', fontsize = font_size)
    ax[0].legend(['train','valid'], fontsize = font_size)
    ax[1].plot(history.history[train_metric_name][t_epochs:])
    ax[1].plot(history.history[val_metric_name][t_epochs:])
    ax[1].set_ylabel('mce', fontsize = font_size)
    ax[1].set_xlabel('Epochs', fontsize = font_size)
    ax[1].legend(['train','valid'], fontsize = font_size)
    canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)
