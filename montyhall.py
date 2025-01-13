# %%
import random as rd

# %%
winning = [0, 0, 0]

# %%
for i in range(10000):
    #Three doors, only one contains the prize
    doors = ["Prize", "Wrong1", "Wrong2"]

    #Shuffling the doors
    rd.shuffle(doors)
    doors

    #The player choose one door 
    choice = rd.choice(doors)
    choice

    aux_doors = doors.copy()

    #Removing one door (which it was neither chosen or not contains the prize)
    if(choice == "Prize"): 
        aux_doors.remove(choice)
        aux_doors = [rd.sample(aux_doors, 1)[0], "Prize"]
        rd.shuffle(aux_doors)

    else:
        aux_doors = []
        aux_doors.append("Prize")
        aux_doors.append(choice)
        rd.shuffle(aux_doors)

    winning[2] += 1

    #Stay with the doors
    if(choice == "Prize"):
        winning[0] += 1

    # Change the door
    else:
        winning[1] += 1


#%%
winning[0]/winning[2]

#%%
winning[1]/winning[2]

