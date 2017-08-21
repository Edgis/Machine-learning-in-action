#coding=utf-8
'''
Created on Aug 21, 2017
Tree-Based Regression Methods
@author: Edgis
'''
from numpy import *
from tkinter import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import regTrees
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

"""
def reDraw(tolS, tolN):
    pass

def drawNewTree():
    pass
"""


def reDraw(tolS, tolN):
    reDraw.f.clf() #清空之前的图像
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN <2:
            #构建模型树
            tolN = 2
            myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf,regTrees.modelErr,(tolS,tolN))
            yHat = regTrees.createForeCast(myTree,reDraw.testDat,regTrees.modelTreeEval)
        else:
            #构建回归树
            myTree = regTrees.createTree(reDraw.rawDat,ops=(tolS,tolN))
            yHat = regTrees.createForeCast(myTree,reDraw.testDat)
        reDraw.a.scatter(reDraw.rawDat[:,0],reDraw.rawDat[:,1],s=5)
        reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
        reDraw.canvas.show()


def getInput():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN")
        #清除错误的输入，用默认值代替
        tolNentry.delete(0, END)
        tolNentry.insert(0,'10')
    try:
        tolS = float(tolNentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolNentry.delete(0,END)
        tolNentry.insert(0,'1.0')
    return tolN, tolS

def drawNewTree():
    tolN, tolS = getInput()
    reDraw(tolS ,tolN)


root = Tk()

Label(root, text ='Plot Place Holder').grid(row=0,columnspan=3)

Label(root, text='tolN').grid(row = 1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1,column=1)
tolNentry.insert(0,'10')
Label(root,text='tolS').grid(row=2,column=0)
tolNentry = Entry(root)
tolNentry.grid(row=2,column=1)
tolNentry.insert(0,'1.0')
Button(root,text='ReDraw',command=drawNewTree).grid(row =1, column=2, rowspan=3)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text='Model Tree',variable = chkBtnVar) #复选框
chkBtn.grid(row=3,column=0,columnspan=2)

reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:,0]), max(reDraw.rawDat[:,0]), 0.01)

reDraw.f = Figure(figsize=(5,4),dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row = 0,columnspan=3)

#reDraw(1.0,10)


root.mainloop()
