import os
import sys
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.options.display.float_format = '{:.3g}'.format
sns.set(font_scale = 1.0, rc={"grid.linewidth": 1,'grid.color': '#b0b0b0', 'axes.edgecolor': 'black',"lines.linewidth": 3.0}, style = 'whitegrid')

import argparse
parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--Delta",type=float,default=1000.)
parser.add_argument("--fraction",type=float,default=0.005)
parser.add_argument("--gamma",type=float,default=8.0)
parser.add_argument("--rho",type=float,default=1.00001)
parser.add_argument("--symmetric",type=int,default=1)
parser.add_argument("--dataname",type=str,default="tests")
args = parser.parse_args()

Delta = args.Delta
fraction = args.fraction
gamma = args.gamma
rho = args.rho
symmetric = args.symmetric
dataname = args.dataname

symmetric_returns    = symmetric
state_dependent_xi   = 0
optimize_over_ell    = 0
compute_irfs         = 0                    # need to start julia with "-p 5"


if symmetric_returns == 1:
    if state_dependent_xi == 0:
        filename = "model_sym_HS.npz"
    elif state_dependent_xi == 1:
        filename = "model_sym_HSHS.npz"
    elif state_dependent_xi == 2:
        filename = "model_sym_HSHS2.npz"
elif symmetric_returns == 0:
    if state_dependent_xi == 0:
        filename = "model_asym_HS.npz"
    elif state_dependent_xi == 1:
        filename = "model_asym_HSHS.npz"
    elif state_dependent_xi == 2:
        filename = "model_asym_HSHS2.npz"

filename_ell = "./output/"+dataname+"/gamma_"+str(gamma)+"_rho_"+str(rho)+"/"
npz = np.load(filename_ell + filename)


figname = "./figure/"+dataname+"/gamma_"+str(gamma)+"_rho_"+str(rho)+"/"
os.makedirs(figname,exist_ok=True)

lscale = 1.0
llim = 18*lscale
lgrid = int(1000*lscale+1)
rscale = llim*2/(lgrid-1)
print(rscale)
plot_benchmark = False

def trans(x):
    return np.exp(x)/(np.exp(x)+1)
def transderi(x):
    return np.exp(x)/(np.exp(x)+1)**2
def read_csv(name):
    h1 = pd.DataFrame(npz[name])
    h1.index = trans(np.linspace(-llim,llim,lgrid))
    h1.columns = np.linspace(-1,1,201)
    return h1
d1 = read_csv('d1')
d2 = read_csv('d2')
h1 = read_csv('h1')
h2 = read_csv('h2')
hz = read_csv('hz')
V = read_csv('V')
g = read_csv('g')

if plot_benchmark == True:
    benchmark = np.load('./output/azt_-0005_ell_ex_0143_model_sym_HS.npz')
    def read_benchmark(name):
        h1 = pd.DataFrame(benchmark[name])
        h1.index = trans(np.linspace(-llim,llim,lgrid))
        h1.columns = np.linspace(-1,1,201)
        return h1
    d1b = read_benchmark('d1')
    d2b = read_benchmark('d2')
    h1b = read_benchmark('h1')
    h2b = read_benchmark('h2')
    hzb = read_benchmark('hz')
    Vb = read_benchmark('V')
    gb = read_benchmark('g')


fig, ax = plt.subplots(1,1,figsize = (4,4))
g_R = g.sum(axis=1)*0.01
newinterval = trans(np.linspace(-llim,llim,lgrid))[1:] - trans(np.linspace(-llim,llim,lgrid))[:-1]
g_R = (g_R*rscale).iloc[1:]/newinterval
sns.lineplot(data = g_R,label = r"$g_R$")
if rho<1.01 and rho > 0.99 and gamma == 8.0 and plot_benchmark == True:
    g_Rb = gb.sum(axis=1)*0.01
    g_Rb = (g_Rb*rscale).iloc[1:]/newinterval
    sns.lineplot(data = g_Rb,label = r"$g_R, \rho =1$", ls = '--')

ax.set_ylim([0.0,4.0])
ax.set_ylabel(r'$g_R$')
ax.set_xlabel(r'$R$')
ax.set_title(r'R density, '+ '$\gamma=$'+str(gamma)+', '+'$\\rho$ ='+str(rho))
fig.tight_layout()
fig.savefig(figname+'/gR.png', dpi = 400)
plt.close()


fig, ax = plt.subplots(1,1,figsize = (4,4))
g_R = g.sum(axis=1)*0.01
g_Rc = g_R/transderi(np.linspace(-llim,llim,lgrid))
sns.lineplot(data = g_Rc,label = r"$g_R$")
ax.set_ylim([0.0,4.0])
ax.set_ylabel(r'$g_R$')
ax.set_xlabel(r'$R$')
ax.set_title(r'R density (change of variable), '+ '$\gamma=$'+str(gamma)+', '+'$\\rho$ ='+str(rho))
fig.tight_layout()
fig.savefig(figname+'/gRc.png', dpi = 400)
plt.close()

gl = pd.DataFrame(npz['g'])
gl.index = np.linspace(-llim,llim,lgrid)
gl.columns = np.linspace(-1,1,201)
fig, ax = plt.subplots(1,1,figsize = (4,4))
g_l = gl.sum(axis=1)*0.01
sns.lineplot(data = g_l,label = r"$g_l$")
if rho<1.01 and rho > 0.99 and gamma == 8.0 and plot_benchmark == True:
    glb = pd.DataFrame(benchmark['g'])
    glb.index = np.linspace(-llim,llim,lgrid)
    glb.columns = np.linspace(-1,1,201)
    g_lb = glb.sum(axis=1)*0.01
    sns.lineplot(data = g_lb,label = r"$g_l, \rho =1$", ls = '--')
ax.set_ylim([0.0,0.5])
ax.set_ylabel(r'$g_l$')
ax.set_xlabel(r'$l$')
ax.set_title(r'l density, '+ '$\gamma=$'+str(gamma)+', '+'$\\rho$ ='+str(rho))
fig.tight_layout()
fig.savefig(figname+'/gl.png', dpi = 400)
plt.close()


fig, ax = plt.subplots(1,1,figsize = (4,4))
g_Z = g.sum(axis=0)*rscale
sns.lineplot(data = g_Z,label = r"$g_Z$")
if rho<1.01 and rho > 0.99 and gamma == 8.0 and plot_benchmark == True:
    g_Zb = gb.sum(axis=0)*rscale
    sns.lineplot(data = g_Zb,label = r"$g_Z, \rho =1$", ls = '--')
ax.set_ylim([0.0,3.0])
ax.set_ylabel(r'$g_Z$')
ax.set_xlabel(r'$Z$')
ax.set_title(r'Z density, '+ '$\gamma=$'+str(gamma)+', '+'$\\rho$ ='+str(rho))
fig.tight_layout()
fig.savefig(figname+'/gZ.png', dpi = 400)
plt.close()

fig, ax = plt.subplots(1,1,figsize = (4,4))
sns.lineplot(data = h1[0],label = r"$-H_1$")
sns.lineplot(data = h2[0],label = r"$-H_2$")
sns.lineplot(data = hz[0],label = r"$-H_z$")
ax.set_ylim([-0.01,0.18])
ax.set_ylabel(r'$-H$')
ax.set_xlabel(r'$R$')
ax.set_title(r'H, '+ '$\gamma=$'+str(gamma)+', '+'$\\rho$ ='+str(rho))
fig.tight_layout()
fig.savefig(figname+'/h.png', dpi = 400)
plt.close()

fig, ax = plt.subplots(1,1,figsize = (4,4))
sns.lineplot(data = d1[0],label = r"$d_1$")
sns.lineplot(data = d2[0],label = r"$d_2$")
if rho<1.01 and rho > 0.99 and gamma == 8.0 and plot_benchmark == True:
    sns.lineplot(data = d1b[0],label = r"$d_1, \rho =1$", ls = '--')
    sns.lineplot(data = d2b[0],label = r"$d_2, \rho =1$", ls = '--')
ax.set_ylim([0.027,0.037])
# ax.set_ylim([-0.01,0.05])
ax.set_ylabel(r'$d$')
ax.set_xlabel(r'$R$')
ax.set_title(r'd, '+ '$\gamma=$'+str(gamma)+', '+'$\\rho$'+'='+str(rho))
fig.tight_layout()
fig.savefig(figname+'/d.png', dpi = 400)
plt.close()

fig, ax = plt.subplots(1,1,figsize = (4,4))
sns.lineplot(data = V[0],label = r"$V$")
if rho<1.01 and rho > 0.99 and gamma == 8.0 and plot_benchmark == True:
    sns.lineplot(data = Vb[0],label = r"$V, \rho =1$", ls = '--')
ax.set_ylim([-12.0,-0.0])
ax.set_ylabel(r'$V$')
ax.set_xlabel(r'$R$')
ax.set_title(r'V, '+ '$\gamma=$'+str(gamma)+', '+'$\\rho$'+'='+str(rho))
fig.tight_layout()

fig.savefig(figname+'/v.png', dpi = 400)
plt.close()


res = npz
W1 = trans(np.linspace(-llim,llim,lgrid))
W2 = np.linspace(-1,1,201)
var_name = ['Investment over Capital', 'Consumption over Capital', 'Log Value Function']


plot_row_dims      = 1
plot_col_dims      = 3

plot_color_style   = ['blues','reds', 'greens']

subplot_titles = []
subplot_types = []
for row in range(plot_row_dims):
    subplot_type = []
    for col in range(plot_col_dims):
        subplot_titles.append(var_name[col])
        subplot_type.append({'type': 'surface'})
    subplot_types.append(subplot_type)
spacing = 0.1
fig = make_subplots(rows=plot_row_dims, cols=plot_col_dims, horizontal_spacing=spacing, vertical_spacing=spacing, subplot_titles=(subplot_titles), specs=subplot_types)
fig.add_trace(go.Surface(z=res['d1'].T[5:-5,5:-5], x=W1, y=W2, colorscale=plot_color_style[0], showscale=False, name= 'd1', showlegend=True), row = 1, col = 1)
fig.add_trace(go.Surface(z=res['d2'].T[5:-5,5:-5], x=W1, y=W2, colorscale=plot_color_style[1], showscale=False, name= 'd2', showlegend=True), row = 1, col = 1)
fig.update_scenes(dict(xaxis_title='r', yaxis_title='z', zaxis_title='d', zaxis = dict(nticks=4, tickformat= ".4f")), row = 1, col = 1)

fig.add_trace(go.Surface(z=res['cons'].T[5:-5,5:-5], x=W1, y=W2, colorscale=plot_color_style[2], showscale=False, name= 'c', showlegend=True), row = 1, col = 2)
fig.update_scenes(dict(xaxis_title='r', yaxis_title='z', zaxis_title='c', zaxis = dict(nticks=4, tickformat= ".4f")), row = 1, col = 2)
fig.update_scenes(dict(aspectmode = 'cube'), row = 1, col = 2)

fig.add_trace(go.Surface(z=res['V'].T[5:-5,5:-5], x=W1, y=W2, colorscale=plot_color_style[2], showscale=False, name= 'V', showlegend=True), row = 1, col = 3)
fig.update_scenes(dict(xaxis_title='r', yaxis_title='z', zaxis_title='V', zaxis = dict(nticks=4, tickformat= ".2f")), row = 1, col = 3)
fig.update_scenes(dict(aspectmode = 'cube'), row = 1, col = 3)
fig.update_layout(title= 'Policy Function, Value Function <br><span style="font-size: 12px;"> gamma = '+ str(gamma)+', rho = '+ str(rho)+'</span>',\
              title_x = 0.5, title_y = 0.97, height=500, width=1200, title_yanchor = 'top')
fig.update_layout(margin=dict(t=75))
fig.write_json(figname+"/3dw.json")
fig.write_image(figname+"/3dw.png")
        

plot_row_dims      = 1
plot_col_dims      = 3

plot_color_style   = ['Viridis', 'Plasma']
plot_color_style   = ['blues','reds', 'greens']

subplot_titles = []
subplot_types = []
for row in range(plot_row_dims):
    subplot_type = []
    for col in range(plot_col_dims):
        subplot_titles.append(var_name[col])
        subplot_type.append({'type': 'surface'})
    subplot_types.append(subplot_type)
spacing = 0.1
fig = make_subplots(rows=plot_row_dims, cols=plot_col_dims, horizontal_spacing=spacing, vertical_spacing=spacing, subplot_titles=(subplot_titles), specs=subplot_types)
fig.add_trace(go.Surface(z=res['d1'].T, x=W1, y=W2, colorscale=plot_color_style[0], showscale=False, name= 'd1', showlegend=True), row = 1, col = 1)
fig.add_trace(go.Surface(z=res['d2'].T, x=W1, y=W2, colorscale=plot_color_style[1], showscale=False, name= 'd2', showlegend=True), row = 1, col = 1)
fig.update_scenes(dict(xaxis_title='r', yaxis_title='z', zaxis_title='d', zaxis = dict(nticks=4, range=[0.0,0.06], tickformat= ".2f")), row = 1, col = 1)

fig.add_trace(go.Surface(z=res['cons'].T, x=W1, y=W2, colorscale=plot_color_style[2], showscale=False, name= 'c', showlegend=True), row = 1, col = 2)
fig.update_scenes(dict(xaxis_title='r', yaxis_title='z', zaxis_title='c', zaxis = dict(nticks=4, range=[0.0,0.06], tickformat= ".2f")), row = 1, col = 2)
fig.update_scenes(dict(aspectmode = 'cube'), row = 1, col = 2)

fig.add_trace(go.Surface(z=res['V'].T, x=W1, y=W2, colorscale=plot_color_style[2], showscale=False, name= 'V', showlegend=True), row = 1, col = 3)
fig.update_scenes(dict(xaxis_title='r', yaxis_title='z', zaxis_title='V', zaxis = dict(nticks=4, tickformat= ".2f")), row = 1, col = 3)
fig.update_scenes(dict(aspectmode = 'cube'), row = 1, col = 3)
fig.update_layout(title= 'Policy Function, Value Function <br><span style="font-size: 12px;"> gamma = '+ str(gamma)+', rho = '+ str(rho)+'</span>',\
              title_x = 0.5, title_y = 0.97, height=500, width=1200, title_yanchor = 'top')
fig.update_layout(margin=dict(t=75))
fig.write_json(figname+"/3d.json")
fig.write_image(figname+"/3d.png")

