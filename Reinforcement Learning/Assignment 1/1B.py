import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table


# gridsize
WORLD_SIZE_width = 4
WORLD_SIZE_height = 3

A_POS = [1, 1]
B_POS = [2, 0]
C_POS = [2, 2]

DISCOUNT = 0.9

# action = {up, down, right, left}
ACTIONS = [np.array([-1, 0]),
           np.array([1, 0]),
           np.array([0, 1]),
           np.array([0, -1]),
]

ACTION_PROB = 0.25



def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows


    for (i,j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')


    for i, label in enumerate(range(len(image))):
        tb.add_cell(i, -1, width, height,  loc='right',
                    edgecolor='none', facecolor='none')


    for j, label in enumerate(range(len(image))):
        tb.add_cell(WORLD_SIZE_width, j, width, height/2, loc='center',
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    

"""
Define the wall and the additional rewards

"""

def step(state, action):


    reward = -1.0
    state1 = np.array(state)
   
    next_state = (state1 + action).tolist()
    
    x, y = next_state

    if x < 0 or x >= WORLD_SIZE_width or y < 0 or y >= WORLD_SIZE_height:
        
        reward = -1.0
        next_state = state
        
        
    
    if state == [3,2]:
        reward = 0
        

####################################

    if state == A_POS and action.tolist() == [0, 1] or action.tolist() == [0, -1]:
        
        reward = -1.0
        next_state = state
    
    if state == [1,0] and action.tolist() == [0, 1]:
        
        reward = -1.0
        next_state = state
        
    if state == [1,2] and action.tolist() == [0, -1]:
        
        reward = -1.0
        next_state = state
        
#####################################

    if state  == C_POS and action.tolist() == [1, 0]:
        
        reward = -1.0
        next_state = state
    
    if state == [3,2] and action.tolist() == [-1, 0]:

        next_state = state
        
        
#####################################
        
    if state == [3,0] and action.tolist() == [0, 1]:
        
        reward = -1.0
        next_state = state
    
    if state == [3,1] and action.tolist() == [0, -1]:
        
        reward = -1.0
        next_state = state
        

#####################################
        
    if state != A_POS and next_state == A_POS :
        reward = -6.0
        
    if state != C_POS and next_state == C_POS:
        reward = -6.0
        
    if state != B_POS and next_state == B_POS:
        
        reward = -11.0
    
        
    return next_state, reward


"""
V^rand
"""

def bellman_equation():

    value = np.zeros((WORLD_SIZE_width, WORLD_SIZE_height))  # initial value = 0
    
    while True:
        
        new_value = np.zeros(value.shape)
        
        for i in range(0, WORLD_SIZE_width):
            
            for j in range(0, WORLD_SIZE_height):

                for action in ACTIONS:
                   
                    (next_i, next_j), reward = step([i, j], action)
                    new_value[i, j] += ACTION_PROB * (reward + DISCOUNT * value[next_i, next_j])

#        print(np.sum(np.abs(value - new_value)))
        print(new_value)
       
        if np.sum(np.abs(value - new_value)) < 1e-5:
            draw_image(np.round(new_value, decimals=2))
            plt.show()
            plt.close()
            break
       
        value = new_value


"""
V^*
"""

def bellman_optimal_equation():

    value = np.zeros((WORLD_SIZE_width, WORLD_SIZE_height))  # initial value = 0
    
    while True:
        
        new_value = np.zeros(value.shape)

        for i in range(0, WORLD_SIZE_width):
            
            for j in range(0, WORLD_SIZE_height):
                
                values = []
             
                for action in ACTIONS:

                    (next_i, next_j), reward = step([i, j], action)
                    values.append(reward + DISCOUNT * value[next_i, next_j])


                new_value[i, j] = np.max(values)

#        print(np.sum(np.abs(new_value - value)))
        if np.sum(np.abs(new_value - value)) < 1e-5:
            draw_image(np.round(new_value, decimals=2))
            plt.show()
            plt.close()
            break
        value = new_value


bellman_equation()
bellman_optimal_equation()
