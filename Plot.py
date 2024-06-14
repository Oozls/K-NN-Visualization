# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import os

plt.rcParams['font.family'] ='AppleSDGothicNeoB00'
plt.rcParams['axes.unicode_minus'] = False

def Test1():
    global blue, red, target

    fig, ax = plt.subplots()

    if os.path.exists('./arrays/blue.npy') and os.path.exists('./arrays/red.npy') and os.path.exists('./arrays/target.npy'):
        blue = np.load('./arrays/blue.npy')
        red = np.load('./arrays/red.npy')
        target = np.load('./arrays/target.npy')
    else:
        blue = np.random.randint(10,30,size=(5,2))
        red = np.random.randint(20,50,size=(5,2))
        target = np.random.randint(10,50,size=(1,2))

        np.save('./arrays/blue.npy', blue)
        np.save('./arrays/red.npy', red)
        np.save('./arrays/target.npy', target)

    plt.scatter(*zip(*blue), c='lightskyblue', marker='s')
    plt.scatter(*zip(*red), c='tomato', marker='o')

    plt.xlabel("속성값 1")
    plt.ylabel("속성값 2")

    plt.savefig('./photos/1-1.png')

    plt.scatter(*zip(*target), c='orange', marker='^')

    plt.savefig('./photos/1-2.png')

    for I in range(3):
        ar = (blue, red, target)[I]
        for i in range(len(ar)): 
            plt.annotate(f'{str(ar[i][0])}, {str(ar[i][1])}', (ar[i][0] + 0.2, ar[i][1] + 0.2))
    
    plt.savefig('./photos/1-3.png')

    circle1 = plt.Circle((21, 18), 7, color='orange', alpha=0.5)
    ax.add_patch(circle1)
    fig.savefig('./photos/1-4.png')

def Test2():
    csv = pd.read_csv('./datas/gender.csv')
    train = csv.iloc[0:400]
    male_train = train[train['Gender'] == 'Male']
    female_train = train[train['Gender'] == 'Female']
    test = csv.iloc[400:500]

    plt.scatter(male_train['Weight'], male_train['Height'], c='lightskyblue', marker='s')
    plt.scatter(female_train['Weight'], female_train['Height'], c='tomato', marker='o')

    plt.xlabel("몸무게")
    plt.ylabel("신장")

    plt.savefig('./photos/2-1.png')
    
    plt.scatter(test['Weight'], test['Height'], c='greenyellow', marker='D')

    weight_height = np.column_stack((csv['Weight'], csv['Height']))
    genders = csv['Gender'].map({'Male': 0, 'Female': 1})
    wh_train = weight_height[0:400,:]
    g_train = genders.iloc[0:400]
    wh_test = weight_height[400:500,:]
    g_test = genders.iloc[400:500]

    kn = KNeighborsClassifier(n_neighbors=100)
    kn.fit(wh_train, g_train)
    print(kn.score(wh_test, g_test))

    plt.savefig('./photos/2-2.png')

def Test2_1():
    csv = pd.read_csv('./datas/diabetes.csv')
    train = csv.iloc[0:100]
    normal_train = train[train['Outcome'] == 0]
    diabetes_train = train[train['Outcome'] == 1]
    test = csv.iloc[100:120]

    plt.scatter(normal_train['Glucose'], normal_train['BMI'], c='lightskyblue', marker='s')
    plt.scatter(diabetes_train['Glucose'], diabetes_train['BMI'], c='tomato', marker='o')

    plt.xlabel("글루코스 (mg/dL)")
    plt.ylabel("BMI")

    plt.savefig('./photos/2-3.png')
    
    plt.scatter(test['Glucose'], test['BMI'], c='greenyellow', marker='D')

    glu_bmi = np.column_stack((csv['Glucose'], csv['BMI']))
    outc = csv['Outcome']
    gb_train = glu_bmi[0:100,:]
    oc_train = outc.iloc[0:100]
    gb_test = glu_bmi[100:120,:]
    oc_test = outc.iloc[100:120]

    kn = KNeighborsClassifier(n_neighbors=10)
    kn.fit(gb_train, oc_train)
    print(kn.score(gb_test, oc_test))

    plt.savefig('./photos/2-4.png')

    ar1 = (np.column_stack((train['Glucose'], train['BMI'], train['BloodPressure'])))
    for i in range(len(ar1)): 
        plt.annotate(f'{str(ar1[i][2])}', (ar1[i][0]-3.5, ar1[i][1]-0.7), fontsize=5)

    ar = (np.column_stack((test['Glucose'], test['BMI'], test['BloodPressure'])))
    for i in range(len(ar)): 
        plt.annotate(f'{str(ar[i][2])}', (ar[i][0]-3.5, ar[i][1]-0.7), c='red', fontsize=8)

    plt.savefig('./photos/2-5.png')

    x0, y0 = 125, 22.5
    distances = np.sqrt((train['Glucose'] - x0)**2 + (train['BMI'] - y0)**2)
    closest_indices = np.argsort(distances)[:5]
    closest_points = train.iloc[closest_indices]
    print(np.column_stack((closest_points['Glucose'], closest_points['BMI'], closest_points['BloodPressure'])))

    knr = KNeighborsRegressor(n_neighbors=5)
    knr.fit(np.column_stack((train['Glucose'], train['BMI'])), train['BloodPressure'])
    prd = knr.predict(np.column_stack((test['Glucose'], test['BMI'])))
    print(mean_absolute_error(prd, test['BloodPressure']))

def Test3():
    fig, axis = plt.subplots(subplot_kw={"projection":"3d"})

    global blue, red
    if os.path.exists('./arrays/blue3D.npy') and os.path.exists('./arrays/red3D.npy'):
        blue = np.load('./arrays/blue3D.npy')
        red = np.load('./arrays/red3D.npy')
    else:
        blue = np.random.randint(10,30,size=(20,3))
        red = np.random.randint(20,50,size=(20,3))

        np.save('./arrays/blue3D.npy', blue)
        np.save('./arrays/red3D.npy', red)

    target = np.array([[18, 33, 23]])
    axis.scatter(*zip(*blue), c='lightskyblue', marker='s')
    axis.scatter(*zip(*red), c='tomato', marker='o')

    def animate(i):
        axis.view_init(elev=30., azim=i)
        return fig,

    anim = animation.FuncAnimation(fig, animate,
                                frames=360, interval=20, blit=True)
    anim.save('./photos/3-1.gif', fps=30)

    plt.scatter(*zip(*target), c='orange', marker='^')
    anim.save('./photos/3-2.gif', fps=30)

def Test4():
    fig, axis = plt.subplots(subplot_kw={"projection":"3d"})
    
    axis.scatter([0, 3, 3, 3], [0, 0, 4, 4], zs=[0, 0, 0, 5])

    axis.plot([0, 3], [0, 0], zs=[0, 0])
    axis.plot([3, 3], [0, 4], zs=[0, 0])
    axis.plot([3, 3], [4, 4], zs=[0, 5])
    axis.plot([0, 3], [0, 4], zs=[0, 5], c="tomato")

    axis.text(1.5, 0, 0.25, "3")
    axis.text(3, 2, 0.25, "4")
    axis.text(3, 3.75, 2.5, "5")

    axis.text(0, 0, 0.5, "A")
    axis.text(3, 4, 5.5, "B")
    
    def animate(i):
        axis.view_init(elev=30., azim=i)
        return fig,

    anim = animation.FuncAnimation(fig, animate,
                                frames=360, interval=20, blit=True)
    anim.save('./photos/4-1.gif', fps=30)

def Test5():
    #fig, axis = plt.subplots(subplot_kw={"projection":"3d"})
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')


    csv = pd.read_csv('./datas/diabetes.csv')
    normal_train = csv[csv['Outcome'] == 0].iloc[0:100]
    diabetes_train = csv[csv['Outcome'] == 1].iloc[0:100]
    train = csv.iloc[0:100]
    
    axis.scatter(normal_train['Glucose'], normal_train['BMI'], normal_train['Insulin'], c='lightskyblue', marker='s')
    axis.scatter(diabetes_train['Glucose'], diabetes_train['BMI'], diabetes_train['Insulin'], c='tomato', marker='o')

    axis.set_xlabel("글루코스")
    axis.set_ylabel("BMI")
    axis.set_zlabel("인슐린")
    #axis.zaxis.set_rotate_label(False) 

    '''plt.savefig('./photos/2-3.png')
    
    plt.scatter(test['Glucose'], test['BMI'], c='greenyellow', marker='D')

    glu_bmi = np.column_stack((csv['Glucose'], csv['BMI']))
    outc = csv['Outcome']
    gb_train = glu_bmi[0:100,:]
    oc_train = outc.iloc[0:100]
    gb_test = glu_bmi[100:120,:]
    oc_test = outc.iloc[100:120]

    kn = KNeighborsClassifier(n_neighbors=10)
    kn.fit(gb_train, oc_train)
    print(kn.score(gb_test, oc_test))'''

    
    def animate(i):
        axis.view_init(elev=30., azim=i)
        return fig,

    anim = animation.FuncAnimation(fig, animate,
                                frames=360, interval=20, blit=True)
    anim.save('./photos/5-1.gif', fps=30)

    x0, y0, z0 = 170, 225, 34.5
    distances = np.sqrt((train['Glucose'] - x0)**2 + (train['BMI'] - y0)**2 + (train['Insulin'] - z0)**2)
    closest_indices = np.argsort(distances)[:11]
    closest_points = train.iloc[closest_indices]
    print(np.column_stack((closest_points['Glucose'], closest_points['BMI'], closest_points['Insulin'], closest_points['Outcome'])))

if __name__ == '__main__':
    Test1() # Change on your own