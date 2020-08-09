import os
import matplotlib.pyplot as plt
import pickle

# read the results

file_name = 'training.pkl'

with open(file_name, 'rb') as f:
	result_dic = pickle.load(f)

start_idx = 4

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(result_dic['tr_loss'][start_idx:],'b-')
ax.plot(result_dic['te_loss'][start_idx:],'r-')

## plot the results


## plot and save the file
def plot_loss(loss,val_loss):
    f_out='loss_epochs.png'
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(loss,'b-',linewidth=1.3)
    ax.plot(val_loss,'r-',linewidth=1.3)
    ax.set_title('Model Loss')
    ax.set_ylabel('MSE')
    ax.set_xlabel('epochs')
    ax.legend(['train', 'test'], loc='upper left')  
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(f_out, dpi=80)

def plot_multi_loss(train_loss_dic,val_loss_dic):
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	f_out='loss_epochs.png'
	start_idx = 4
	if (len(train_loss_dic['loss'])>start_idx):
		fig = Figure(figsize=(12,8))
		ax = fig.add_subplot(2,3,1)
		ax.plot(train_loss_dic['loss'][start_idx:],'b-',linewidth=1.3)
		ax.plot(val_loss_dic['loss'][start_idx:],'r-',linewidth=1.3)
		ax.set_title('Total loss')
		ax.set_ylabel('MSE')
		ax.set_xlabel('epochs')
		ax.legend(['train', 'test'], loc='upper left')
		bx = fig.add_subplot(2,3,2)
		bx.plot(train_loss_dic['ori'][start_idx:],'b-',linewidth=1.3)
		bx.plot(val_loss_dic['ori'][start_idx:],'r-',linewidth=1.3)
		bx.set_title('Red_1 loss')
		bx.set_ylabel('MSE')
		bx.set_xlabel('epochs')
		bx.legend(['train', 'test'], loc='upper left')
		cx = fig.add_subplot(2,3,3)
		cx.plot(train_loss_dic['red2'][start_idx:],'b-',linewidth=1.3)
		cx.plot(val_loss_dic['red2'][start_idx:],'r-',linewidth=1.3)
		cx.set_title('Red_2 loss')
		cx.set_ylabel('MSE')
		cx.set_xlabel('epochs')
		cx.legend(['train', 'test'], loc='upper left')
		dx = fig.add_subplot(2,3,4)
		dx.plot(train_loss_dic['red4'][start_idx:],'b-',linewidth=1.3)
		dx.plot(val_loss_dic['red4'][start_idx:],'r-',linewidth=1.3)
		dx.set_title('Red_4 loss')
		dx.set_ylabel('MSE')
		dx.set_xlabel('epochs')
		dx.legend(['train', 'test'], loc='upper left')
		ex = fig.add_subplot(2,3,5)
		ex.plot(train_loss_dic['red8'][start_idx:],'b-',linewidth=1.3)
		ex.plot(val_loss_dic['red8'][start_idx:],'r-',linewidth=1.3)
		ex.set_title('Red_8 loss')
		ex.set_ylabel('MSE')
		ex.set_xlabel('epochs')
		ex.legend(['train', 'test'], loc='upper left')  
		canvas = FigureCanvasAgg(fig)
		canvas.print_figure(f_out, dpi=80)

# plot_loss(result_dic['tr_loss'],result_dic['te_loss'])
plot_multi_loss(result_dic['tr_loss'],result_dic['te_loss'])